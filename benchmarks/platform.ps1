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

.NOTES
Exit: 0 if every run passed every check; 1 if a gate, a build or a check
failed; 2 on a usage error.
#>
param(
    [ValidateSet('small', 'medium', 'large')][string]$Class = 'medium',
    [string]$Variants = 'native,native-fast',
    [string]$Threads = '',
    [int]$Jobs = 0,
    [string]$Out = '',
    [string]$Machine = '',
    [switch]$NoGate
)

$ErrorActionPreference = 'Continue'
$RepoRoot  = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$BuildRoot = Join-Path $RepoRoot 'build-platform'
$ncpu = [Environment]::ProcessorCount
if ($Jobs -le 0) { $Jobs = [Math]::Max(1, [int]($ncpu / 2)) }
if (-not $Threads) { $Threads = if ($ncpu -gt 1) { "1,$ncpu" } else { '1' } }

$ThreadList = @()
foreach ($t in ($Threads -split ',')) {
    $n = 0
    if (-not [int]::TryParse($t.Trim(), [ref]$n) -or $n -lt 1) {
        Write-Host "platform.ps1: bad thread count '$t'"; exit 2
    }
    $ThreadList += $n
}
$MaxThreads = ($ThreadList | Measure-Object -Maximum).Maximum
$VariantList = @($Variants -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ })
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

Write-Host "mtl5 platform benchmark: class=$Class variants=$Variants threads=$Threads jobs=$Jobs"
Write-Host "results: $Out"

# Run a native tool with stdout+stderr to files, returning the exit code. Same
# pattern as run_sweeps.ps1: Start-Process -Wait, arguments with spaces quoted
# (a repo under "C:\Users\Some Name\..." otherwise breaks the configure, #438).
function Invoke-Native {
    param([string]$Exe, [string[]]$NativeArgs, [string]$LogBase)
    $quoted = $NativeArgs | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }
    $p = Start-Process -FilePath $Exe -ArgumentList $quoted -PassThru -NoNewWindow -Wait `
                       -RedirectStandardOutput "$LogBase.out.log" `
                       -RedirectStandardError  "$LogBase.err.log"
    return $p.ExitCode
}

# --- 1. preflight, before the builds ----------------------------------------
$PreflightScript = Join-Path $PSScriptRoot 'preflight.ps1'
function Invoke-Preflight {
    param([string]$Phase = 'before')
    if ($Phase -eq 'after') { return (& $PreflightScript -Phase after) }
    $kv = & $PreflightScript -Threads $MaxThreads -Repo $RepoRoot -ReportOnly:$NoGate
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
    foreach ($t in $ThreadList) {
        $base = Join-Path $Out "$v-t$t"
        Write-Host "== run $v, $t thread(s)"
        # Every library's pool pinned to the same count (see platform.sh).
        foreach ($name in 'MTL5_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                          'BLIS_NUM_THREADS', 'OMP_NUM_THREADS') {
            Set-Item -Path "env:$name" -Value "$t"
        }
        $runArgs = @('--class', $Class, '--label', $v, '--json', "$base.json", '--card', "$base.md")
        if ($Machine) { $runArgs += @('--machine', $Machine) }
        $rc = Invoke-Native $exe $runArgs (Join-Path $Logs "$v-t$t")
        $last = Get-Content (Join-Path $Logs "$v-t$t.out.log") -Tail 1
        if ($rc -eq 0) { Write-Host "   $last" }
        else { Write-Host "   FAILED (exit $rc): $last (see logs\$v-t$t.*.log)"; $Failed = 1 }
        if (Test-Path "$base.json.sysinfo") {
            # Machine state into the sidecar, so it is IN the data (#442).
            $after = Invoke-Preflight -Phase after
            if (-not ($after -match 'thermal_after_c=')) { $after = 'thermal_after_c=unavailable' }
            Add-Content -Encoding ascii -Path "$base.json.sysinfo" -Value ($PreflightKv + $after)
            $Runs += $base
        }
    }
}

# --- summary --------------------------------------------------------------------
$summary = @('# mtl5_bench platform characterization', '',
             "Class ``$Class``, variants $($Built -join ' '), threads $Threads, $((Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')).")
if ($Skipped.Count -gt 0) { $summary += @('', 'Not measured:') + ($Skipped | ForEach-Object { "- $_" }) }
$summary += ''
foreach ($r in $Runs) { if (Test-Path "$r.md") { $summary += Get-Content "$r.md" } }
$summary | Set-Content -Encoding ascii (Join-Path $Out 'summary.md')

Write-Host ''
Write-Host "summary: $(Join-Path $Out 'summary.md')"
exit $Failed
