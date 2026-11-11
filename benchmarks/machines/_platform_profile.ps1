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
# build, minutes later, for VS 2026. The newest instance with the C++ tools
# that THIS CMake has a generator for is pinned through CMAKE_GENERATOR and
# CMAKE_GENERATOR_INSTANCE, which CMake reads from the environment. (CMake 3.31
# knows no VS 2026, and given its instance falls back to a default generator
# that then fails on the mismatched instance.)
function Use-VisualStudio {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) { Write-Host "no Visual Studio (vswhere.exe not found)"; exit 2 }
    $known = @(((& cmake -E capabilities) | ConvertFrom-Json).generators | ForEach-Object { $_.name })
    # CMake's generator names carry the product year, which vswhere doesn't
    # report reliably (catalog_productLineVersion is "18" for VS 2026).
    $years = @{ 15 = 2017; 16 = 2019; 17 = 2022; 18 = 2026 }
    # Complete instances only (no -all): an install or update in progress is
    # registered but unusable -- the Ryzen 9's VS 2026 mid-update was
    # isComplete=0, and CMake refused it. ForEach-Object unrolls the array,
    # which Windows PowerShell's ConvertFrom-Json emits as ONE object.
    $all = @((& $vswhere -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -format json) |
             Out-String | ConvertFrom-Json | ForEach-Object { $_ }) |
           Sort-Object { [version]$_.installationVersion } -Descending
    $skipped = @()
    foreach ($vs in $all) {
        $major = ([version]$vs.installationVersion).Major
        $gen = if ($years.ContainsKey($major)) { "Visual Studio $major $($years[$major])" } else { $null }
        if ($gen -and $known -contains $gen) {
            $env:CMAKE_GENERATOR = $gen
            $env:CMAKE_GENERATOR_INSTANCE = $vs.installationPath
            $note = if ($skipped.Count) { "; skipped $($skipped -join ', '), which this CMake has no generator for" } else { '' }
            Write-Host "   compiler: $gen ($($vs.installationVersion))$note"
            return
        }
        $skipped += "$($vs.installationVersion)"
    }
    Write-Host "no Visual Studio with the C++ tools that this CMake ($((cmake --version)[0])) can generate for"
    exit 2
}

$script:PowerRestore = $null
$script:RestoreFailed = $false
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
# powercfg, failing loudly: a native command's nonzero exit is not a PowerShell
# error, so an unchecked write that failed would still be reported as applied.
function Invoke-Powercfg {
    $out = & powercfg @args 2>&1
    if ($LASTEXITCODE -ne 0) { throw "powercfg $($args -join ' ') failed (exit $LASTEXITCODE): $out" }
    $out
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
    try {
        Invoke-Powercfg /changename $copy "mtl5 performance - temporary copy of $origName" | Out-Null
        foreach ($mode in '/setacvalueindex', '/setdcvalueindex') {
            Invoke-Powercfg $mode $copy SUB_PROCESSOR PROCTHROTTLEMIN 100 | Out-Null
            Invoke-Powercfg $mode $copy SUB_PROCESSOR PROCTHROTTLEMAX 100 | Out-Null
            Invoke-Powercfg $mode $copy SUB_PROCESSOR PERFEPP 0 | Out-Null
        }
        Invoke-Powercfg /setactive $copy | Out-Null
    } catch {
        Write-Host "clock policy not applied: $_"
        Exit-ClockPolicy; exit 2
    }
    if ((Get-ActiveScheme) -ne $copy) { Exit-ClockPolicy; Write-Host "could not activate the session power scheme"; exit 2 }
    $ac = if ((Get-CimInstance Win32_Battery -ErrorAction SilentlyContinue)) { 'battery present' } else { 'AC' }
    return "power scheme copy of $origName with processor state 100-100% and EPP 0, boost $(Get-BoostMode $copy); $ac"
}
function Exit-ClockPolicy {
    if (-not $script:PowerRestore) { return }
    # Both steps are attempted; a failure is reported, since it leaves the machine
    # on the session scheme (or with a stray copy) for whoever uses it next.
    foreach ($step in @(@('/setactive', $script:PowerRestore.orig), @('/delete', $script:PowerRestore.copy))) {
        try { Invoke-Powercfg @step | Out-Null }
        catch {
            Write-Host "WARNING: restoring the power scheme: $_ -- fix by hand with powercfg /setactive $($script:PowerRestore.orig)"
            $script:RestoreFailed = $true   # the session then fails (Invoke-PlatformProfile)
        }
    }
    $script:PowerRestore = $null
}

function Get-ClangClToolset {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) { return $false }
    # The pinned instance (Use-VisualStudio), and the compiler itself: the
    # component can be registered without its tools (the Ryzen 9 during a
    # Visual Studio update), and MSBuild then fails with MSB8020.
    if (-not (& $vswhere -path $env:CMAKE_GENERATOR_INSTANCE -products * -requires Microsoft.VisualStudio.Component.VC.Llvm.ClangToolset -property installationPath)) { return $false }
    Test-Path (Join-Path $env:CMAKE_GENERATOR_INSTANCE 'VC\Tools\Llvm\x64\bin\clang-cl.exe')
}

# The interleave-width sweep (#780) on one CPU, under the policy still applied.
function Invoke-Interleave([int]$cpu, [string]$policy, [string]$out) {
    New-Item -ItemType Directory -Force $out | Out-Null
    $status = 0
    $commit = (git -C $ProfileRoot rev-parse HEAD)
    $arms = @(@{ name = 'msvc'; toolset = @() })
    if (Get-ClangClToolset) { $arms += @{ name = 'clang-cl'; toolset = @('-T', 'ClangCL') } }
    else { Write-Host "== bench_interleave_rows (clang-cl): skipped, this Visual Studio has no clang-cl (C++ Clang tools for Windows)" }
    foreach ($arm in $arms) {
        $dir = Join-Path $ProfileRoot "build-platform\interleave-$($arm.name)"
        # Fresh, like platform.ps1's variants: a reused cache keeps its old
        # generator, and CMake prints the compiler it found only on a first configure.
        if (Test-Path $dir) { Remove-Item -Recurse -Force $dir }
        Write-Host "== bench_interleave_rows ($($arm.name))"
        & cmake -S $ProfileRoot -B $dir @($arm.toolset) -DMTL5_BUILD_BENCHMARKS=ON -DMTL5_BUILD_TESTS=OFF `
            -DMTL5_BUILD_EXAMPLES=OFF *>&1 | Out-File -Encoding ascii (Join-Path $out "interleave-$($arm.name).configure.log")
        if ($LASTEXITCODE -eq 0) {
            & cmake --build $dir --config Release --target bench_interleave_rows --parallel 4 `
                *>&1 | Out-File -Encoding ascii (Join-Path $out "interleave-$($arm.name).build.log")
        }
        if ($LASTEXITCODE -ne 0) { Write-Host "   build FAILED ($($arm.name))"; $status = 1; continue }
        $exe = Join-Path $dir 'benchmarks\Release\bench_interleave_rows.exe'
        $id = Select-String -Path (Join-Path $out "interleave-$($arm.name).configure.log") `
                  -Pattern 'The CXX compiler identification is (.*)' | Select-Object -First 1
        $compiler = if ($id) { $id.Matches[0].Groups[1].Value.Trim() } else { "$($arm.name), version not recorded" }
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
        # The results directory is always named here, so the gate's outcome can be
        # read from it below; the default is platform.ps1's own.
        $out = if ($env:PLATFORM_OUT) { $env:PLATFORM_OUT } else {
            Join-Path $ProfileRoot "build-platform\results\$($env:COMPUTERNAME)-$((Get-Date).ToUniversalTime().ToString("yyyyMMdd'T'HHmmss'Z'"))" }
        $pargs.Out = $out
        # Out-Host: only the exit status may reach this function's output.
        & (Join-Path $ProfileRoot 'benchmarks\platform.ps1') @pargs @extra | Out-Host
        $status = $LASTEXITCODE
        if ($env:INTERLEAVE -eq '1') {
            # Only on a machine the session accepted: platform.ps1 writes
            # logs\preflight.txt once both preflight gates have passed (or -NoGate
            # made them report-only). A gate failure, or no variant built, leaves
            # none, and the sweep would measure a machine the session rejected.
            if (Test-Path (Join-Path $out 'logs\preflight.txt')) {
                if ((Invoke-Interleave -cpu ([int]($cpus -split ',')[0]) -policy $policy -out $out) -ne 0) { $status = 1 }
            } else {
                Write-Host "== bench_interleave_rows: skipped, the session did not pass preflight"
                $status = 1
            }
        }
    } finally {
        Exit-ClockPolicy
        # A machine left on the session scheme is not a clean finish.
        if ($script:RestoreFailed) { $status = 1 }
    }
    return $status
}
