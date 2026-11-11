<#
.SYNOPSIS
Characterize this machine with mtl5_bench: the one command, Windows (#721 M2).

.DESCRIPTION
The PowerShell counterpart of benchmarks/platform.sh, and the same contract:

  1. preflight.ps1 gates the session and records the machine state. -NoGate
     records without gating (CI, or a first look at a machine).
  2. Configures and builds mtl5_bench once per variant, each in its own clean
     build directory under build-platform\. Every variant is built before any
     is measured.
       native       generic C++
       native-fast  Highway SIMD + MTL5's blocked GEMM (MSVC has no
                    -march=native; the Highway baseline target is used)
       blas         system BLAS/LAPACK, if CMake finds one (else skipped)
  3. preflight again, after the builds.
  4. Runs every variant at every thread count, writing <variant>-t<T>.json with
     its .sysinfo (machine state appended) and <variant>-t<T>.md (the card), and
     summary.md with every card.

.PARAMETER Class
small, medium (default) or large.
.PARAMETER Variants
Comma list; default native,native-fast. blas is opt-in on Windows, where a
system BLAS is rarely installed.
.PARAMETER Threads
Comma list of thread counts; default 1 and the number of logical CPUs.
.PARAMETER Jobs
Parallel compile jobs; default half the logical CPUs, at least 1.
.PARAMETER Out
Results directory; default build-platform\results\<host>-<UTC time>.
.PARAMETER Machine
Machine name for the cards; default the CPU brand.
.PARAMETER NoGate
Record the machine state but never stop on it.
.PARAMETER Cpus
Pin the measurements (#737): a run at T threads gets the processor affinity of
the first T logical CPUs of this list. Comma-separated ids or ranges ("0,2,4"
or "0-3"), one per core to measure, in the order threads should take them.
Every measured child process inherits the affinity from this script, which
sets it on itself just for the run, so it is in force before the benchmark
starts its threads. The builds are not pinned. Each run's sidecar records
bench_cpus=. Without -Cpus the runs are unpinned (bench_cpus=unpinned), and a
machine with SMT gets a warning: its threads may land on siblings.
.PARAMETER ClockPolicy
The clock policy the caller applied, recorded in every sidecar as
clock_policy= (the machine profiles benchmarks\machines\*-platform.ps1 set it).

.NOTES
Exit: 0 if every run passed every check; 1 if a gate, a build or a check
failed; 2 on a usage error.
#>
param(
    [ValidateSet('small', 'medium', 'large')][string]$Class = 'medium',
    # [string[]], not [string]: PowerShell parses `-Threads 1,2` as an ARRAY, and
    # binding that to a [string] joins it with spaces ("1 2"). Either form --
    # an array or one comma-separated string -- is accepted below.
    [string[]]$Variants = @('native', 'native-fast'),
    [string[]]$Threads = @(),
    [int]$Jobs = 0,
    [string]$Out = '',
    [string]$Machine = '',
    [switch]$NoGate,
    [string[]]$Cpus = @(),
    [string]$ClockPolicy = ''
)

$ErrorActionPreference = 'Continue'
$RepoRoot  = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$BuildRoot = Join-Path $RepoRoot 'build-platform'
$ncpu = [Environment]::ProcessorCount
if ($Jobs -le 0) { $Jobs = [Math]::Max(1, [int]($ncpu / 2)) }
function Split-List([string[]]$items) {
    @($items | ForEach-Object { $_ -split '[,\s]+' } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
}
if (-not $Threads -or $Threads.Count -eq 0) { $Threads = if ($ncpu -gt 1) { @('1', "$ncpu") } else { @('1') } }
$Threads = Split-List $Threads
$Variants = Split-List $Variants

$ThreadList = @()
foreach ($t in $Threads) {
    $n = 0
    if (-not [int]::TryParse($t.Trim(), [ref]$n) -or $n -lt 1) {
        Write-Host "platform.ps1: bad thread count '$t'"; exit 2
    }
    $ThreadList += $n
}
# Ascending and de-duplicated: the 1-thread run, when requested, comes first,
# so every later run has this session's baseline (see platform.sh).
$ThreadList = @($ThreadList | Sort-Object -Unique)
$MaxThreads = ($ThreadList | Measure-Object -Maximum).Maximum
$VariantList = $Variants

# -- CPU pinning (#737) ---------------------------------------------------------
# CpuList: the -Cpus ids, expanded and in order. Pin-Mask T is the affinity mask
# of the first T of them, Pin-List T the same as a comma list for the sidecar.
$CpuList = @()
foreach ($item in (Split-List $Cpus)) {
    if ($item -match '^(\d+)-(\d+)$') {
        $lo = [int]$Matches[1]; $hi = [int]$Matches[2]
        if ($lo -gt $hi) { Write-Host "platform.ps1: bad -Cpus range '$item'"; exit 2 }
        $CpuList += $lo..$hi
    } elseif ($item -match '^\d+$') {
        $CpuList += [int]$item
    } else { Write-Host "platform.ps1: bad -Cpus item '$item'"; exit 2 }
}
if ($CpuList.Count -gt 0) {
    $dups = @($CpuList | Group-Object | Where-Object { $_.Count -gt 1 } | ForEach-Object { $_.Name })
    # Affinity is a set: "0,0" pins a 2-thread run to one CPU.
    if ($dups.Count -gt 0) { Write-Host "platform.ps1: -Cpus repeats CPU(s) $($dups -join ',')"; exit 2 }
    foreach ($c in $CpuList) {
        # ProcessorAffinity is one 64-bit mask: processor group 0 only.
        if ($c -ge $ncpu -or $c -ge 64) { Write-Host "platform.ps1: -Cpus names CPU $c; this machine has CPUs 0-$($ncpu - 1)"; exit 2 }
    }
    if ($CpuList.Count -lt $MaxThreads) {
        Write-Host "platform.ps1: -Cpus lists $($CpuList.Count) CPUs but -Threads asks for $MaxThreads"; exit 2
    }
} else {
    $cpu0 = Get-CimInstance Win32_Processor | Select-Object -First 1
    if ($cpu0 -and $cpu0.NumberOfLogicalProcessors -gt $cpu0.NumberOfCores) {
        Write-Host "WARNING: unpinned run on a machine with SMT: threads may land on SMT siblings."
        Write-Host "         Pass -Cpus, or use the machine's profile benchmarks\machines\*-platform.ps1 (#737)."
    }
}
function Pin-List([int]$t) { ($CpuList | Select-Object -First $t) -join ',' }
function Pin-Mask([int]$t) {
    [int64]$m = 0
    foreach ($c in ($CpuList | Select-Object -First $t)) { $m = $m -bor ([int64]1 -shl $c) }
    return $m
}
$Self = [System.Diagnostics.Process]::GetCurrentProcess()
$FullAffinity = $Self.ProcessorAffinity
# Run a script block with this process's affinity set to mask (children inherit
# it at creation), then put the full set back.
function With-Affinity([int64]$mask, [scriptblock]$body) {
    if ($mask -ne 0) { $Self.ProcessorAffinity = [IntPtr]$mask }
    try { & $body } finally { if ($mask -ne 0) { $Self.ProcessorAffinity = $FullAffinity } }
}
foreach ($v in $VariantList) {
    if ($v -notin @('native', 'native-fast', 'blas')) { Write-Host "platform.ps1: unknown variant '$v'"; exit 2 }
}

if (-not $Out) {
    $stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMdd'T'HHmmss'Z'")
    $Out = Join-Path $BuildRoot "results\$($env:COMPUTERNAME)-$stamp"
}
New-Item -ItemType Directory -Force -Path (Join-Path $Out 'logs') | Out-Null
$Out = (Resolve-Path $Out).Path
$Logs = Join-Path $Out 'logs'

$CpusText = if ($CpuList.Count -gt 0) { $CpuList -join ',' } else { 'unpinned' }
$PolicyText = if ($ClockPolicy) { $ClockPolicy } else { 'not applied' }
Write-Host "mtl5 platform benchmark: class=$Class variants=$($Variants -join ',') threads=$($Threads -join ',') jobs=$Jobs cpus=$CpusText"
Write-Host "results: $Out"

# Run a native tool with stdout+stderr to files, returning the exit code.
# Arguments with spaces are quoted: a repo under "C:\Users\Some Name\..."
# otherwise breaks the configure (#438).
#
# NOT Start-Process -Wait. -Wait waits for the process AND ITS DESCENDANTS, and
# an MSVC build leaves descendants behind: MSBuild keeps reusable worker nodes
# alive for about 15 minutes after a build. Under -Wait every `cmake --build`
# blocked until those idle nodes timed out, and the Windows CI job sat in this
# script for 13-30 minutes while Linux and macOS took one. Waiting on the
# process itself, plus disabling node reuse below, removes both halves.
# Touching .Handle before waiting keeps ExitCode readable in Windows
# PowerShell 5.1, which otherwise loses it once the process exits -- the reason
# run_sweeps.ps1 used -Wait.
function Invoke-Native {
    param([string]$Exe, [string[]]$NativeArgs, [string]$LogBase)
    $quoted = $NativeArgs | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }
    $p = Start-Process -FilePath $Exe -ArgumentList $quoted -PassThru -NoNewWindow `
                       -RedirectStandardOutput "$LogBase.out.log" `
                       -RedirectStandardError  "$LogBase.err.log"
    $null = $p.Handle
    $p.WaitForExit()
    return $p.ExitCode
}

# No lingering MSBuild worker nodes after this script's builds.
$env:MSBUILDDISABLENODEREUSE = '1'

# --- 1. preflight, before the builds ----------------------------------------
$PreflightScript = Join-Path $PSScriptRoot 'preflight.ps1'
function Invoke-Preflight {
    param([string]$Phase = 'before')
    if ($Phase -eq 'after') { return (& $PreflightScript -Phase after) }
    $kv = With-Affinity (Pin-Mask $CpuList.Count) { & $PreflightScript -Threads $MaxThreads -Repo $RepoRoot -ReportOnly:$NoGate }
    if ($LASTEXITCODE -ne 0) {
        Write-Host "preflight failed -- not measuring. Fix the above, pass -NoGate to record"
        Write-Host "without gating, or set `$env:ALLOW_DIRTY = '1' to accept a dirty tree (it is recorded)."
        exit 1
    }
    return $kv
}
Invoke-Preflight | Set-Content -Encoding ascii (Join-Path $Logs 'preflight-before-build.txt')

# --- 2. build every variant ---------------------------------------------------
$Built = @(); $Skipped = @(); $Failed = 0
foreach ($v in $VariantList) {
    $dir = Join-Path $BuildRoot $v
    if (Test-Path $dir) { Remove-Item -Recurse -Force $dir }
    $cfg = @('-S', $RepoRoot, '-B', $dir, '-DCMAKE_BUILD_TYPE=Release', '-DMTL5_BUILD_BENCHMARKS=ON',
             '-DMTL5_BUILD_TESTS=OFF', '-DMTL5_BUILD_EXAMPLES=OFF')
    if ($v -eq 'native-fast') { $cfg += @('-DMTL5_WITH_HIGHWAY=ON', '-DMTL5_NATIVE_FAST_GEMM=ON') }
    if ($v -eq 'blas')        { $cfg += @('-DMTL5_WITH_BLAS=ON', '-DMTL5_WITH_LAPACK=ON') }
    Write-Host "== build $v"
    if ((Invoke-Native 'cmake' $cfg (Join-Path $Logs "$v.configure")) -ne 0) {
        if ($v -eq 'blas') {
            Write-Host "   skipped: no BLAS/LAPACK found (see logs\$v.configure.err.log)"
            $Skipped += "${v}: no BLAS/LAPACK found"; continue
        }
        Write-Host "   configure FAILED (see $Logs\$v.configure.err.log)"
        $Skipped += "${v}: configure failed"; $Failed = 1; continue
    }
    $bld = @('--build', $dir, '--config', 'Release', '--target', 'mtl5_bench', '--parallel', "$Jobs")
    if ((Invoke-Native 'cmake' $bld (Join-Path $Logs "$v.build")) -ne 0) {
        Write-Host "   build FAILED (see $Logs\$v.build.out.log)"
        $Skipped += "${v}: build failed"; $Failed = 1; continue
    }
    $Built += $v
}
if ($Built.Count -eq 0) { Write-Host 'nothing built -- no measurements'; exit 1 }

# --- 3. preflight, after the builds -------------------------------------------
$PreflightKv = Invoke-Preflight
$PreflightKv | Set-Content -Encoding ascii (Join-Path $Logs 'preflight.txt')

# --- 4. measure ----------------------------------------------------------------
$Runs = @()
foreach ($v in $Built) {
    # Visual Studio generators put the binary under Release\; Ninja does not.
    $exe = Join-Path $BuildRoot "$v\benchmarks\Release\mtl5_bench.exe"
    if (-not (Test-Path $exe)) { $exe = Join-Path $BuildRoot "$v\benchmarks\mtl5_bench.exe" }
    $t1Json = $null   # this invocation's 1-thread result for $v, once it exists
    foreach ($t in $ThreadList) {
        $base = Join-Path $Out "$v-t$t"
        Write-Host "== run $v, $t thread(s)"
        # Remove a previous session's files for this run (a reused -Out).
        foreach ($f in "$base.json", "$base.json.sysinfo", "$base.md") {
            if (Test-Path $f) { Remove-Item -Force $f }
        }
        # Every library's pool pinned to the same count (see platform.sh).
        foreach ($name in 'MTL5_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                          'BLIS_NUM_THREADS', 'OMP_NUM_THREADS') {
            Set-Item -Path "env:$name" -Value "$t"
        }
        $runArgs = @('--class', $Class, '--label', $v, '--json', "$base.json", '--card', "$base.md")
        if ($Machine) { $runArgs += @('--machine', $Machine) }
        # Thread scaling against THIS invocation's 1-thread run of the variant.
        if ($t -ne 1 -and $t1Json) { $runArgs += @('--baseline', $t1Json) }
        # Pinned (#737): this run's threads get the first t CPUs of -Cpus.
        $benchCpus = if ($CpuList.Count -gt 0) { Pin-List $t } else { 'unpinned' }
        $rc = With-Affinity (Pin-Mask $(if ($CpuList.Count -gt 0) { $t } else { 0 })) {
            Invoke-Native $exe $runArgs (Join-Path $Logs "$v-t$t")
        }
        $last = Get-Content (Join-Path $Logs "$v-t$t.out.log") -Tail 1
        if ($rc -eq 0) { Write-Host "   $last" }
        else { Write-Host "   FAILED (exit $rc): $last (see logs\$v-t$t.*.log)"; $Failed = 1 }
        if (Test-Path "$base.json.sysinfo") {
            # Machine state into the sidecar, so it is IN the data (#442).
            $after = Invoke-Preflight -Phase after
            if (-not ($after -match 'thermal_after_c=')) { $after = 'thermal_after_c=unavailable' }
            Add-Content -Encoding ascii -Path "$base.json.sysinfo" -Value ($PreflightKv + $after +
                @("bench_cpus=$benchCpus", "clock_policy=$PolicyText"))
            $Runs += $base
            if ($t -eq 1 -and (Test-Path "$base.json")) { $t1Json = "$base.json" }
        }
    }
}

# --- summary --------------------------------------------------------------------
$summary = @('# mtl5_bench platform characterization', '',
             "Class ``$Class``, variants $($Built -join ' '), threads $($Threads -join ','), CPUs $CpusText, clock policy $PolicyText, $((Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')).")
if ($Skipped.Count -gt 0) { $summary += @('', 'Not measured:') + ($Skipped | ForEach-Object { "- $_" }) }
$summary += ''
foreach ($r in $Runs) { if (Test-Path "$r.md") { $summary += Get-Content "$r.md" } }
$summary | Set-Content -Encoding ascii (Join-Path $Out 'summary.md')

Write-Host ''
Write-Host "summary: $(Join-Path $Out 'summary.md')"
exit $Failed
