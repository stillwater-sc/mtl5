<#
.SYNOPSIS
    Windows/MSVC port of run_scaling.sh: multi-core GEMM scaling (#108) across
    native-fast / openblas / mkl, one CSV per backend.

.DESCRIPTION
    Threading is a runtime axis of the same per-backend binary:
      native-fast : MTL5_NUM_THREADS=T
      openblas    : OPENBLAS_NUM_THREADS=T
      mkl         : MKL_NUM_THREADS=T
    For T threads we pin to the first T physical cores -- one logical id per core,
    SMT siblings excluded -- so scaling reflects cores, not hyperthreads. Pinning
    is a Windows affinity BITMASK (OR of one bit per physical core), not taskset.

    BLIS is not built (no native MSVC support). Each CSV's `backend` column is
    labelled "<backend>-t<T>" so analyze_scaling.py can recover (backend, T).

.PARAMETER PCores
    First-logical-processor id of each physical core, in the order to fill. The
    default 0,2,4,... matches an SMT machine whose sibling threads are adjacent
    (verify with topology_probe: physical core -> logical mask).

.PARAMETER Threads
    Thread counts to sweep (default 1,2,4,8).

.PARAMETER Sizes
    GEMM sizes (default "1024,2048").

.EXAMPLE
    pwsh benchmarks/run_scaling.ps1 -Threads 1,2,4,8 -Sizes "1024,2048"
#>
[CmdletBinding()]
param(
    # Comma strings, not [int[]]: passing "1,2,4,8" to an [int[]] param via
    # `powershell -File` coerces it to the single integer 1248 (commas read as
    # digit-group separators). Parse the strings into int lists ourselves.
    [string]$PCores = "0,2,4,6,8,10,12,14",
    [string]$Threads = "1,2,4,8",
    [string]$Sizes = "1024,2048",
    [string[]]$Variants = @("native-fast", "openblas", "mkl"),
    [string]$VcpkgToolchain = $(if ($env:VCPKG_ROOT) { Join-Path $env:VCPKG_ROOT "scripts\buildsystems\vcpkg.cmake" } else { "" }),
    [string]$MklRoot = "C:\Users\tomtz\dev\mkl-venv\Library",
    [string]$OpenBlasRoot = "C:\Users\tomtz\dev\openblas-win",
    # REQUIRED, one directory PER MACHINE (e.g. benchmarks\data\ryzen-9-8945hs).
    # There is deliberately no default: the CSVs are named by backend, not by
    # machine, so a shared default silently overwrites another machine's
    # committed results -- which is exactly what happened once (#439).
    [Parameter(Mandatory = $true)][string]$OutDir,
    [int]$Jobs = [Environment]::ProcessorCount,

    # Permit a dirty working tree (still recorded as tree_git_dirty=1). A switch,
    # not only the ALLOW_DIRTY environment variable, because `set ALLOW_DIRTY=1`
    # is cmd.exe syntax -- in PowerShell `set` aliases Set-Variable, so the run
    # fails again with the same message and no clue why.
    [switch]$AllowDirty
)

if ($Variants.Count -eq 1 -and $Variants[0] -match ',') { $Variants = $Variants[0] -split ',' }
[int[]]$PCoreList  = $PCores  -split ',' | ForEach-Object { [int]$_ }
[int[]]$ThreadList = $Threads -split ',' | ForEach-Object { [int]$_ }

# See run_sweeps.ps1 for why $ErrorActionPreference is left at Continue.
$RepoRoot = Split-Path -Parent $PSScriptRoot
$DataDir  = Join-Path $RepoRoot $OutDir
$LogDir   = Join-Path $DataDir "logs"
New-Item -ItemType Directory -Force -Path $DataDir, $LogDir | Out-Null

# Affinity mask covering the first T physical cores (one logical id per core).
function Mask-ForThreads {
    param([int]$T)
    if ($T -gt $PCoreList.Count) {
        throw "T=$T exceeds the $($PCoreList.Count) physical cores in -PCores; add cores or reduce -Threads."
    }
    $m = 0L
    for ($i = 0; $i -lt $T; $i++) { $m = $m -bor (1L -shl $PCoreList[$i]) }
    return $m
}

function Invoke-Native {
    param([string]$Exe, [string[]]$NativeArgs, [string]$LogBase)
    # Quote arguments containing spaces: Start-Process joins -ArgumentList with
    # spaces and quotes NOTHING, so a repo under "C:\Users\Some Name\..." reaches
    # cmake as two broken arguments, with a path in the log that looks right (#438).
    $quoted = $NativeArgs | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }
    $p = Start-Process -FilePath $Exe -ArgumentList $quoted -PassThru -NoNewWindow -Wait `
                       -RedirectStandardOutput "$LogBase.out.log" -RedirectStandardError "$LogBase.err.log"
    return $p.ExitCode
}

function Build-Variant {
    param([string]$Dir, [string[]]$CMakeArgs)
    $full = Join-Path $RepoRoot $Dir
    if (Test-Path $full) { Remove-Item -Recurse -Force $full }
    $base = @("-B", $full, "-DMTL5_BUILD_BENCHMARKS=ON", "-DMTL5_BUILD_TESTS=OFF",
              "-DMTL5_BUILD_EXAMPLES=OFF", "-DCMAKE_BUILD_TYPE=Release")
    if ((Invoke-Native "cmake" ($base + $CMakeArgs) (Join-Path $LogDir "$Dir.configure")) -ne 0) {
        throw "configure failed for $Dir (see $LogDir\$Dir.configure.err.log)"
    }
    if ((Invoke-Native "cmake" @("--build", $full, "--config", "Release", "--target", "bench_all", "-j", "$Jobs") (Join-Path $LogDir "$Dir.build")) -ne 0) {
        throw "build failed for $Dir (see $LogDir\$Dir.build.err.log)"
    }
    return (Join-Path $full "benchmarks\Release\bench_all.exe")
}

$script:Sidecars = @()

# Run the thread sweep for one backend, merging per-T CSVs into one file.
function Run-Scaling {
    param([string]$Exe, [string]$Backend, [string]$ThreadVar)
    $out = Join-Path $DataDir "gemm_scaling_${Backend}.csv"
    if (Test-Path $out) { Remove-Item $out }
    if (Test-Path "$out.sysinfo") { Remove-Item "$out.sysinfo" }
    $first = $true
    foreach ($T in $ThreadList) {
        $mask = Mask-ForThreads $T
        Set-Item "env:$ThreadVar" "$T"
        $env:OMP_NUM_THREADS = "$T"
        $tmp = Join-Path $DataDir "gemm_scaling_${Backend}_t${T}.csv"
        $log = Join-Path $LogDir  "gemm_scaling_${Backend}_t${T}.out"
        Write-Host ("  {0}  T={1}  affinity mask 0x{2:x}" -f $Backend, $T, $mask)

        $psi = New-Object System.Diagnostics.ProcessStartInfo
        $psi.FileName = $Exe
        $psi.Arguments = "--suite gemm --sizes $Sizes --label ${Backend}-t${T} --csv `"$tmp`""
        $psi.UseShellExecute = $false
        $psi.RedirectStandardOutput = $true
        $p = [System.Diagnostics.Process]::Start($psi)
        try { $p.ProcessorAffinity = [IntPtr]$mask; $p.PriorityClass = 'High' } catch { Write-Warning $_ }
        $p.StandardOutput.ReadToEnd() | Out-File "$log.log" -Encoding utf8
        $p.WaitForExit()
        if ($p.ExitCode -ne 0) { throw "bench failed ($Backend T=$T); see $log.log" }

        if ($first) {
            Get-Content $tmp | Set-Content $out
            # bench_all writes <csv>.sysinfo beside the CSV it was given; the
            # per-T runs merge into one file, so keep the first one.
            if (Test-Path "$tmp.sysinfo") { Move-Item "$tmp.sysinfo" "$out.sysinfo" }
            $first = $false
        } else {
            Get-Content $tmp | Select-Object -Skip 1 | Add-Content $out
            if (Test-Path "$tmp.sysinfo") { Remove-Item "$tmp.sysinfo" -Force }
        }
        Remove-Item $tmp -Force
    }
    $script:Sidecars += "$out.sysinfo"
    Write-Host "  -> $out"
}

# --- preflight -------------------------------------------------------------
# Gate the session, and record the state it ran in (#442). Called twice: once
# BEFORE the builds so a dirty tree or competing compile fails in seconds rather
# than after ten minutes of configuring, and once after them, because the builds
# heat the machine and the temperature that matters is the one the MEASUREMENTS
# start at. The second record goes in the sidecars.
#
# The thread budget is validated HERE rather than where the mask is built: the
# old code discovered "T=16 exceeds 8 physical cores" inside the measurement
# loop, after the builds and after the smaller thread counts had already been
# measured and written.
foreach ($T in $ThreadList) {
    if ($T -lt 1) { throw "-Threads: '$T' is not a positive integer" }
    if ($T -gt $PCoreList.Count) {
        throw "T=$T exceeds the $($PCoreList.Count) physical core(s) in -PCores ($PCores); add cores or reduce -Threads."
    }
}
$TMax = ($ThreadList | Measure-Object -Maximum).Maximum
$PreflightScript = Join-Path $PSScriptRoot "preflight.ps1"
function Invoke-Preflight {
    $kv = & $PreflightScript -Threads $TMax -Repo $RepoRoot -AllowDirty:$AllowDirty -IgnorePath $DataDir
    if ($LASTEXITCODE -ne 0) {
        Write-Host "preflight failed -- not measuring. Fix the above, or pass -AllowDirty"
        Write-Host "if you accept a dirty tree (it is then recorded as such)."
        exit 1
    }
    return $kv
}
Invoke-Preflight | Out-Null

# --- build phase -----------------------------------------------------------
# Every variant is built BEFORE any is measured: building and measuring in turn
# timed each backend on a machine still hot from compiling its own binary, a
# per-arm bias in a comparison whose whole purpose is comparing arms.
Write-Host "=== building all variants (nothing is measured yet) ===`n"
$Built = @()
foreach ($v in $Variants) {
    switch ($v) {
        "native-fast" {
            Write-Host "=== native-fast (MTL5_NUM_THREADS) ==="
            $Built += [pscustomobject]@{
                Label = "native-fast"; ThreadVar = "MTL5_NUM_THREADS"
                Exe = (Build-Variant "build-scaling-native-fast" @(
                    "-DMTL5_NATIVE_FAST_GEMM=ON", "-DMTL5_WITH_HIGHWAY=ON", "-DMTL5_NATIVE_ARCH=ON")) }
        }
        "openblas" {
            $obLib = Join-Path $OpenBlasRoot "lib\libopenblas.lib"
            if (-not (Test-Path $obLib)) { Write-Host "=== openblas: SKIPPED (no $obLib) ==="; break }
            Write-Host "=== openblas (OPENBLAS_NUM_THREADS) ==="
            $env:PATH = "$(Join-Path $OpenBlasRoot 'bin');$env:PATH"
            $Built += [pscustomobject]@{
                Label = "openblas"; ThreadVar = "OPENBLAS_NUM_THREADS"
                Exe = (Build-Variant "build-scaling-openblas" @(
                    "-DMTL5_WITH_BLAS=ON", "-DMTL5_WITH_LAPACK=ON",
                    "-DBLAS_LIBRARIES=$obLib", "-DLAPACK_LIBRARIES=$obLib")) }
        }
        "mkl" {
            $mklLib = Join-Path $MklRoot "lib\mkl_rt.lib"
            if (-not (Test-Path $mklLib)) { Write-Host "=== mkl: SKIPPED (no $mklLib) ==="; break }
            Write-Host "=== mkl (MKL_NUM_THREADS) ==="
            $env:PATH = "$(Join-Path $MklRoot 'bin');$env:PATH"
            $Built += [pscustomobject]@{
                Label = "mkl"; ThreadVar = "MKL_NUM_THREADS"
                Exe = (Build-Variant "build-scaling-mkl" @(
                    "-DMTL5_WITH_BLAS=ON", "-DMTL5_WITH_LAPACK=ON",
                    "-DBLAS_LIBRARIES=$mklLib", "-DLAPACK_LIBRARIES=$mklLib")) }
        }
        default { Write-Warning "unknown variant '$v', skipping" }
    }
}

# --- measurement phase -----------------------------------------------------
# NOT interleaved, unlike run_blocking_ab: bench_all truncates its --csv on every
# invocation and analyze_scaling.py keeps the LAST row for a (backend, op, size,
# T) key rather than the best, so rounds would overwrite rather than accumulate.
# Tracked in #442.
$PreflightKv = Invoke-Preflight
Write-Host "`n=== measuring ==="
foreach ($b in $Built) {
    Write-Host "=== $($b.Label) ==="
    Run-Scaling $b.Exe $b.Label $b.ThreadVar
}

# --- record machine state in every sidecar ---------------------------------
# A failed postflight read must not silently drop the key: `unavailable` is a
# fact, an absent key is a reader's guess.
$AfterKv = & $PreflightScript -Phase after
if ($LASTEXITCODE -ne 0) {
    Write-Warning "postflight thermal read failed; recording thermal_after_c=unavailable."
    $AfterKv = "thermal_after_c=unavailable"
}
if (-not ($AfterKv -match 'thermal_after_c=')) { $AfterKv = "thermal_after_c=unavailable" }

foreach ($s in $script:Sidecars) {
    if (Test-Path $s) {
        Add-Content -Path $s -Value $PreflightKv
        Add-Content -Path $s -Value $AfterKv
        Add-Content -Path $s -Value @(
            "harness=run_scaling.ps1",
            "harness_pinned_cores=$PCores",
            "harness_interleaved=0")
    }
}

Write-Host "`nDone. Analyze with:"
Write-Host "  python benchmarks/analyze_scaling.py $DataDir\gemm_scaling_*.csv --plot $DataDir\gemm_scaling.png"
