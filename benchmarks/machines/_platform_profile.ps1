# Shared helpers for the Windows platform-benchmark machine profiles (#737):
# benchmarks\machines\<machine>-platform.ps1. The PowerShell counterpart of
# _platform_profile.sh, and the same contract:
#
# A profile is the whole invocation for one machine: it identifies the machine,
# derives the CPUs to measure from the topology (never hard-coded), applies and
# records the clock policy, and calls benchmarks\platform.ps1 with -Cpus and
# -ClockPolicy. Everything it changes is put back when it ends.
#
# The clock policy is a power scheme: a copy of the active one with the
# processor's minimum and maximum state at 100% and the energy-performance
# preference at 0 (performance) -- the Windows counterpart of the Linux
# `performance` governor. Turbo (boost mode) is left as the scheme has it and
# recorded. The copy is made active for the session; afterwards the previous
# scheme is re-activated and the copy deleted. powercfg needs an administrator.
#
#   $env:FORCE = '1'             run on a machine the profile doesn't recognize.
#   $env:NO_CLOCK_POLICY = '1'   measure without applying the policy; the
#                                sidecars then record clock_policy=not applied,
#                                and the card has to be published as PRELIMINARY.
#   $env:PLATFORM_OUT = DIR      results directory, passed to platform.ps1 -Out.
#   $env:CLASS = C               small, medium (default) or large.
#   $env:INTERLEAVE = '1'        after the session, still under the clock policy,
#                                run bench_interleave_rows (#780) on the first
#                                measured CPU: REPS runs (default 3) with MSVC,
#                                and as many with clang-cl where Visual Studio
#                                has it. Written to interleave-<compiler>-<i>.txt.
#
# Usage, from a profile:
#     . (Join-Path $PSScriptRoot '_platform_profile.ps1')
#     Assert-Machine '8845HS'
#     $cpus = Get-OneCpuPerCore
#     exit (Invoke-PlatformProfile ryzen-7-8845hs $cpus $args)

$ErrorActionPreference = 'Continue'
$ProfileRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path

function Assert-Machine([string]$pattern) {
    $name = (Get-CimInstance Win32_Processor | Select-Object -First 1).Name
    if ($name -notmatch [regex]::Escape($pattern)) {
        Write-Host "this profile is for a $pattern; this machine reports: $name"
        if ($env:FORCE -ne '1') { Write-Host "set `$env:FORCE = '1' if this really is one"; exit 2 }
    }
}

# One logical CPU per physical core, the first SMT sibling of each. Windows
# numbers the logical processors of a core consecutively, so with SMT they are
# 0,2,4,...; checked against the core count rather than assumed.
function Get-OneCpuPerCore {
    $p = @(Get-CimInstance Win32_Processor)
    $cores = ($p | Measure-Object NumberOfCores -Sum).Sum
    $lps = ($p | Measure-Object NumberOfLogicalProcessors -Sum).Sum
    if ($lps -eq $cores) { return (0..($cores - 1)) -join ',' }
    if ($lps -eq 2 * $cores) { return (0..($cores - 1) | ForEach-Object { 2 * $_ }) -join ',' }
    Write-Host "cannot derive one CPU per core: $cores cores, $lps logical processors"
    exit 2
}

# OpenBLAS for the blas variant, from an unpacked official Windows release
# (OpenBLAS-<ver>-x64.zip) under ~\opt. CMake reads CMAKE_PREFIX_PATH and
# BLA_VENDOR from the environment; the DLL has to be on PATH for both the
# configure-time ?gemmt probe and the measurements.
function Use-OpenBlas {
    $dir = Get-ChildItem -Directory (Join-Path $env:USERPROFILE 'opt') -Filter 'OpenBLAS-*' -ErrorAction SilentlyContinue |
           Where-Object { Test-Path (Join-Path $_.FullName 'lib\libopenblas.lib') } |
           Sort-Object { [version]($_.Name -replace '^OpenBLAS-', '') } | Select-Object -Last 1
    if (-not $dir) { Write-Host "   no OpenBLAS under ~\opt: the blas variant will be skipped"; return $false }
    $env:CMAKE_PREFIX_PATH = $dir.FullName
    $env:BLA_VENDOR = 'OpenBLAS'
    $env:PATH = (Join-Path $dir.FullName 'bin') + ';' + $env:PATH
    Write-Host "   BLAS: $($dir.Name)"
    return $true
}

# One Visual Studio for every build of the session. With several installed,
# CMake's default generator choice is not stable: on the Ryzen 9 (VS 2022 and
# VS 2026) one session's mtl5_bench configured for VS 2022 and its interleave
# build, minutes later, for VS 2026. The newest instance with the C++ tools is
# pinned through CMAKE_GENERATOR and CMAKE_GENERATOR_INSTANCE, which CMake reads
# from the environment.
function Use-VisualStudio {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) { Write-Host "no Visual Studio (vswhere.exe not found)"; exit 2 }
    $req = 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64'
    $path = & $vswhere -latest -products * -requires $req -property installationPath
    $ver  = & $vswhere -latest -products * -requires $req -property installationVersion
    $year = & $vswhere -latest -products * -requires $req -property catalog_productLineVersion
    if (-not $path -or -not $ver -or -not $year) { Write-Host "no Visual Studio with the C++ tools"; exit 2 }
    $env:CMAKE_GENERATOR = "Visual Studio $(([version]$ver).Major) $year"
    $env:CMAKE_GENERATOR_INSTANCE = $path
    Write-Host "   compiler: $env:CMAKE_GENERATOR ($ver)"
}

$script:PowerRestore = $null
function Get-ActiveScheme {
    if ((powercfg /getactivescheme) -match '([0-9a-fA-F-]{36})') { return $Matches[1] }
    return $null
}
function Get-BoostMode([string]$scheme) {
    # /qh: PERFBOOSTMODE is a hidden setting, which /query leaves out.
    $q = powercfg /qh $scheme SUB_PROCESSOR PERFBOOSTMODE 2>$null
    $ac = $q | Select-String 'Current AC Power Setting Index: 0x([0-9a-fA-F]+)' | Select-Object -First 1
    if (-not $ac) { return 'unknown' }
    switch ([Convert]::ToInt32($ac.Matches[0].Groups[1].Value, 16)) {
        0 { 'off' } 1 { 'on' } 2 { 'aggressive' } 3 { 'efficient' } 4 { 'efficient aggressive' }
        5 { 'aggressive at guaranteed' } 6 { 'efficient aggressive at guaranteed' } default { "mode $_" }
    }
}
# Apply the session scheme and return the policy text for the sidecars.
function Enter-ClockPolicy {
    $orig = Get-ActiveScheme
    if (-not $orig) { Write-Host "cannot read the active power scheme"; exit 2 }
    $origName = ((powercfg /getactivescheme) -replace '.*\((.*)\).*', '$1')
    $copy = $null
    if ((powercfg /duplicatescheme $orig) -match '([0-9a-fA-F-]{36})') { $copy = $Matches[1] }
    if (-not $copy) { Write-Host "powercfg /duplicatescheme failed (needs an administrator)"; exit 2 }
    $script:PowerRestore = @{ orig = $orig; copy = $copy }
    # The name says what it is; preflight.ps1 records it as the governor and
    # looks for "performance" in it (and reads the name up to the first ")").
    powercfg /changename $copy "mtl5 performance - temporary copy of $origName" | Out-Null
    foreach ($mode in '/setacvalueindex', '/setdcvalueindex') {
        powercfg $mode $copy SUB_PROCESSOR PROCTHROTTLEMIN 100 | Out-Null
        powercfg $mode $copy SUB_PROCESSOR PROCTHROTTLEMAX 100 | Out-Null
        powercfg $mode $copy SUB_PROCESSOR PERFEPP 0 | Out-Null
    }
    powercfg /setactive $copy | Out-Null
    if ((Get-ActiveScheme) -ne $copy) { Exit-ClockPolicy; Write-Host "could not activate the session power scheme"; exit 2 }
    $ac = if ((Get-CimInstance Win32_Battery -ErrorAction SilentlyContinue)) { 'battery present' } else { 'AC' }
    return "power scheme copy of $origName with processor state 100-100% and EPP 0, boost $(Get-BoostMode $copy); $ac"
}
function Exit-ClockPolicy {
    if (-not $script:PowerRestore) { return }
    powercfg /setactive $script:PowerRestore.orig | Out-Null
    powercfg /delete $script:PowerRestore.copy | Out-Null
    $script:PowerRestore = $null
}

function Get-ClangClToolset {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) { return $false }
    # The pinned instance (Use-VisualStudio), not merely the newest one.
    [bool](& $vswhere -path $env:CMAKE_GENERATOR_INSTANCE -products * -requires Microsoft.VisualStudio.Component.VC.Llvm.ClangToolset -property installationPath)
}

# The interleave-width sweep (#780) on one CPU, under the policy still applied.
function Invoke-Interleave([int]$cpu, [string]$policy, [string]$out) {
    New-Item -ItemType Directory -Force $out | Out-Null
    $status = 0
    $commit = (git -C $ProfileRoot rev-parse HEAD)
    $arms = @(@{ name = 'msvc'; toolset = @() })
    if (Get-ClangClToolset) { $arms += @{ name = 'clang-cl'; toolset = @('-T', 'ClangCL') } }
    else { Write-Host "== bench_interleave_rows (clang-cl): skipped, Visual Studio has no C++ Clang tools" }
    foreach ($arm in $arms) {
        $dir = Join-Path $ProfileRoot "build-platform\interleave-$($arm.name)"
        Write-Host "== bench_interleave_rows ($($arm.name))"
        & cmake -S $ProfileRoot -B $dir @($arm.toolset) -DMTL5_BUILD_BENCHMARKS=ON -DMTL5_BUILD_TESTS=OFF `
            -DMTL5_BUILD_EXAMPLES=OFF *>&1 | Out-File -Encoding ascii (Join-Path $out "interleave-$($arm.name).configure.log")
        if ($LASTEXITCODE -eq 0) {
            & cmake --build $dir --config Release --target bench_interleave_rows --parallel 4 `
                *>&1 | Out-File -Encoding ascii (Join-Path $out "interleave-$($arm.name).build.log")
        }
        if ($LASTEXITCODE -ne 0) { Write-Host "   build FAILED ($($arm.name))"; $status = 1; continue }
        $exe = Join-Path $dir 'benchmarks\Release\bench_interleave_rows.exe'
        $compiler = (Select-String -Path (Join-Path $out "interleave-$($arm.name).configure.log") `
                     -Pattern 'The CXX compiler identification is (.*)' | Select-Object -First 1).Matches.Groups[1].Value
        $self = [System.Diagnostics.Process]::GetCurrentProcess()
        $full = $self.ProcessorAffinity
        for ($i = 1; $i -le $(if ($env:REPS) { [int]$env:REPS } else { 3 }); $i++) {
            $file = Join-Path $out "interleave-$($arm.name)-$i.txt"
            $self.ProcessorAffinity = [IntPtr]([int64]1 -shl $cpu)
            try {
                $body = & $exe --reps 21 2>&1
                $rc = $LASTEXITCODE
            } finally { $self.ProcessorAffinity = $full }
            @("# bench_interleave_rows, $compiler, cpu $cpu, clock policy: $policy",
              "# commit $commit, $((Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ'))") + $body |
                Set-Content -Encoding ascii $file
            if ($rc -ne 0) { Write-Host "   run FAILED ($($arm.name) #$i)"; $status = 1; break }
        }
    }
    return $status
}

# Run the platform benchmark: 1 thread and one per listed CPU, pinned, under the
# clock policy; the policy is restored however this ends. Returns the exit code.
function Invoke-PlatformProfile([string]$machine, [string]$cpus, [object[]]$extra) {
    $n = @($cpus -split ',').Count
    $policy = 'not applied (NO_CLOCK_POLICY=1)'
    Use-VisualStudio
    $null = Use-OpenBlas
    $status = 0
    try {
        if ($env:NO_CLOCK_POLICY -ne '1') { $policy = Enter-ClockPolicy }
        Write-Host "== ${machine}: CPUs $cpus ($n), clock policy: $policy"
        $class = if ($env:CLASS) { $env:CLASS } else { 'medium' }
        $pargs = @{ Class = $class; Cpus = $cpus; Threads = "1,$n"; Machine = $machine; ClockPolicy = $policy
                    Variants = 'native,native-fast,blas' }
        if ($env:PLATFORM_OUT) { $pargs.Out = $env:PLATFORM_OUT }
        # Out-Host: only the exit status may reach this function's output.
        & (Join-Path $ProfileRoot 'benchmarks\platform.ps1') @pargs @extra | Out-Host
        $status = $LASTEXITCODE
        if ($env:INTERLEAVE -eq '1') {
            $out = if ($env:PLATFORM_OUT) { $env:PLATFORM_OUT } else { Join-Path $ProfileRoot 'build-platform\results\interleave' }
            if ((Invoke-Interleave ([int]($cpus -split ',')[0]) $policy $out) -ne 0) { $status = 1 }
        }
    } finally {
        Exit-ClockPolicy
    }
    return $status
}
