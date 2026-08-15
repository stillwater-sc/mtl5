<#
.SYNOPSIS
    Windows/MSVC port of run_sweeps.sh: build one bench_all per backend and run
    the single-threaded BLAS/LAPACK sweeps, one tagged CSV per backend.

.DESCRIPTION
    Same "one executable per backend" methodology as the bash script: each
    variant is configured with the flags a dependent application would set for
    the whole program, and the public mtl:: API dispatches accordingly.

    Two things differ from the Linux flow by necessity:
      * Pinning uses a Windows processor-affinity BITMASK, not `taskset -c <id>`.
        The default single-thread mask 0x1 pins to the first logical processor
        of physical core 0 (its SMT sibling is left idle). Discover masks for
        your machine with topology_probe (physical core -> logical mask).
      * BLIS is not built: it has no native MSVC support. The comparison is
        native / native-fast / openblas / mkl (BLIS is a Linux-only curve).

.PARAMETER Sweep
    Sweep spec passed to bench_all (default 65:1025:80, all-odd / non-power-of-2).

.PARAMETER Suites
    Space- or comma-separated suites to run per variant (default "blas").
    native and native-fast have no LAPACK, so use "blas" for the GEMM gate.

.PARAMETER Variants
    Which backends to build and run (default native,native-fast,openblas,mkl).
    Missing dependencies (vcpkg for openblas, oneAPI for mkl) are skipped with a
    warning rather than failing the run.

.PARAMETER PinMask
    Hex/int affinity mask for single-thread runs (default 0x1 = core 0, one thread).

.PARAMETER VcpkgToolchain
    Path to vcpkg.cmake. Defaults to $env:VCPKG_ROOT\scripts\buildsystems\vcpkg.cmake.

.PARAMETER MklSetvars
    Path to oneAPI setvars.bat. Default: C:\Program Files (x86)\Intel\oneAPI\setvars.bat.

.EXAMPLE
    pwsh benchmarks/run_sweeps.ps1 -Suites blas -PinMask 0x1
#>
[CmdletBinding()]
param(
    [string]$Sweep = "65:1025:80",
    [string]$Suites = "blas",
    [string[]]$Variants = @("native", "native-fast", "openblas", "mkl"),
    [long]$PinMask = 0x1,
    [string]$VcpkgToolchain = $(if ($env:VCPKG_ROOT) { Join-Path $env:VCPKG_ROOT "scripts\buildsystems\vcpkg.cmake" } else { "" }),
    # MKL from the pip mkl-devel package: point at its "Library" dir (contains
    # lib\mkl_rt.lib and bin\mkl_rt.*.dll). A full oneAPI install works too --
    # point this at <oneapi>\mkl\latest.
    [string]$MklRoot = "C:\Users\tomtz\dev\mkl-venv\Library",
    # Optimized OpenBLAS: the official prebuilt Windows binary (mingw/DYNAMIC_ARCH,
    # with LAPACK). vcpkg's MSVC OpenBLAS is a generic, untuned kernel (~8x slower)
    # and BLAS-only, so it is not usable as a reference. Point this at the unzipped
    # OpenBLAS-<ver>-x64 dir (contains lib\libopenblas.lib and bin\libopenblas.dll).
    [string]$OpenBlasRoot = "C:\Users\tomtz\dev\openblas-win",
    # REQUIRED, one directory PER MACHINE (e.g. benchmarks\data\ryzen-9-8945hs).
    # There is deliberately no default: the CSVs are named by backend, not by
    # machine, so a shared default silently overwrites another machine's
    # committed results -- which is exactly what happened once (#439).
    [Parameter(Mandatory = $true)][string]$OutDir,
    [int]$Jobs = [Environment]::ProcessorCount
)

# `powershell -File script.ps1 -Variants a,b,c` can arrive as a single-element
# array holding the comma-joined string; normalise to a real list.
if ($Variants.Count -eq 1 -and $Variants[0] -match ',') { $Variants = $Variants[0] -split ',' }

# NB: do NOT set $ErrorActionPreference = 'Stop' here. Native tools (cmake) emit
# progress on stderr, and in Windows PowerShell 5.1 a redirected native stderr is
# wrapped in ErrorRecords that would trip Stop even on exit code 0. We run native
# tools via Start-Process with file redirection and check exit codes explicitly.
$RepoRoot = Split-Path -Parent $PSScriptRoot   # benchmarks/ -> repo root
$DataDir  = Join-Path $RepoRoot $OutDir
$LogDir   = Join-Path $DataDir "logs"
New-Item -ItemType Directory -Force -Path $DataDir, $LogDir | Out-Null

$SuiteList = $Suites -split '[,\s]+' | Where-Object { $_ }

# --- helpers ---------------------------------------------------------------

# Run bench_all pinned to $PinMask with every backend's threading forced to 1,
# so no vendor quietly uses more of the machine than another.
function Invoke-Pinned {
    param([string]$Exe, [string[]]$BenchArgs, [string]$LogPath)

    $env:OMP_NUM_THREADS      = "1"
    $env:OPENBLAS_NUM_THREADS = "1"
    $env:MKL_NUM_THREADS      = "1"
    $env:BLIS_NUM_THREADS     = "1"
    $env:MTL5_NUM_THREADS     = "1"

    # Use System.Diagnostics.Process directly (not Start-Process): it lets us set
    # ProcessorAffinity on the live process AND read ExitCode reliably in PS 5.1.
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $Exe
    $psi.Arguments = ($BenchArgs | ForEach-Object {
        if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }) -join ' '
    $psi.UseShellExecute        = $false
    $psi.RedirectStandardOutput = $true
    $p = [System.Diagnostics.Process]::Start($psi)
    try {
        # Set as early as possible after creation; multi-second kernels mean the
        # brief pre-affinity window never overlaps a timed section.
        $p.ProcessorAffinity = [IntPtr]$PinMask
        $p.PriorityClass     = [System.Diagnostics.ProcessPriorityClass]::High
    } catch {
        Write-Warning "could not set affinity/priority: $_"
    }
    # Read stdout to completion before WaitForExit to avoid a full-pipe deadlock.
    $out = $p.StandardOutput.ReadToEnd()
    $p.WaitForExit()
    $out | Out-File -FilePath $LogPath -Encoding utf8
    if ($p.ExitCode -ne 0) { throw "bench_all failed (exit $($p.ExitCode)); see $LogPath" }
}

# Run a native tool with stdout+stderr redirected to files, returning exit code.
# Using Start-Process (not the call operator with 2>&1) avoids PowerShell 5.1
# wrapping native stderr in ErrorRecords.
function Invoke-Native {
    param([string]$Exe, [string[]]$NativeArgs, [string]$LogBase)
    # Quote arguments containing spaces, exactly as Invoke-Pinned does.
    # Start-Process joins -ArgumentList with spaces and quotes NOTHING, so a repo
    # under "C:\Users\Some Name\..." reaches cmake as two broken arguments and
    # the configure fails with a path that looks correct in the log (#438).
    $quoted = $NativeArgs | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }
    # -Wait is required: without it, Start-Process -PassThru leaves ExitCode
    # empty in Windows PowerShell 5.1 (disposed process handle).
    $p = Start-Process -FilePath $Exe -ArgumentList $quoted -PassThru -NoNewWindow -Wait `
                       -RedirectStandardOutput "$LogBase.out.log" `
                       -RedirectStandardError  "$LogBase.err.log"
    return $p.ExitCode
}

# Clean-configure + build one bench_all variant. Extra cmake args as an array.
function Build-Variant {
    param([string]$Dir, [string[]]$CMakeArgs)
    $full = Join-Path $RepoRoot $Dir
    if (Test-Path $full) { Remove-Item -Recurse -Force $full }
    $base = @(
        "-B", $full,
        "-DMTL5_BUILD_BENCHMARKS=ON",
        "-DMTL5_BUILD_TESTS=OFF",
        "-DMTL5_BUILD_EXAMPLES=OFF",
        "-DCMAKE_BUILD_TYPE=Release"
    )
    $rc = Invoke-Native "cmake" ($base + $CMakeArgs) (Join-Path $LogDir "$Dir.configure")
    if ($rc -ne 0) { throw "configure failed for $Dir (see $LogDir\$Dir.configure.err.log)" }
    $rc = Invoke-Native "cmake" @("--build", $full, "--config", "Release", "--target", "bench_all", "-j", "$Jobs") (Join-Path $LogDir "$Dir.build")
    if ($rc -ne 0) { throw "build failed for $Dir (see $LogDir\$Dir.build.err.log)" }
    return (Join-Path $full "benchmarks\Release\bench_all.exe")
}

$script:Sidecars = @()

function Run-Variant {
    param([string]$Exe, [string]$Label, [switch]$BlasOnly)
    foreach ($s in $SuiteList) {
        if ($BlasOnly -and $s -ne "blas") {
            Write-Host "   ($Label has no LAPACK path; skipping '$s' suite)"; continue
        }
        $csv = Join-Path $DataDir "${s}_sweep_${Label}.csv"
        $log = Join-Path $LogDir  "${s}_sweep_${Label}.out"
        $sweepFlag = if ($s -eq "blas") { "--blas-sweep" } else { "--lapack-sweep" }
        Write-Host ">> ${Label}: $s sweep ($Sweep)"
        Invoke-Pinned -Exe $Exe -LogPath $log -BenchArgs @(
            "--suite", $s, $sweepFlag, $Sweep, "--label", $Label, "--csv", $csv)
        $script:Sidecars += "$csv.sysinfo"
    }
}

# --- preflight -------------------------------------------------------------
# Gate the session, and record the state it ran in (#442). Called twice on
# purpose: once BEFORE the builds so a dirty tree or a competing compile fails in
# seconds rather than after ten minutes of configuring, and once after them,
# because the builds heat the machine and the temperature that matters is the one
# the MEASUREMENTS start at. The second record goes in the sidecars.
$PreflightScript = Join-Path $PSScriptRoot "preflight.ps1"
function Invoke-Preflight {
    $kv = & $PreflightScript -Threads 1 -Repo $RepoRoot
    if ($LASTEXITCODE -ne 0) {
        Write-Host "preflight failed -- not measuring. Fix the above, or set ALLOW_DIRTY=1"
        Write-Host "if you accept a dirty tree (it is then recorded as such)."
        exit 1
    }
    return $kv
}
Invoke-Preflight | Out-Null

# --- build phase -----------------------------------------------------------
# Every variant is built BEFORE any is measured. Building and measuring in turn
# meant each backend was timed on a machine still hot from compiling its own
# binary, while the next compiled during the previous one's cooldown -- a per-arm
# bias in a comparison whose whole purpose is comparing arms.
Write-Host "Pinning single-thread runs to affinity mask 0x$($PinMask.ToString('x'))"
Write-Host "Data -> $DataDir"
Write-Host "=== building all variants (nothing is measured yet) ===`n"

$Built = @()
foreach ($v in $Variants) {
    switch ($v) {
        "native" {
            Write-Host "=== native (generic-only) ==="
            $Built += [pscustomobject]@{ Label = "native"; Exe = (Build-Variant "build-native" @()) }
        }
        "native-fast" {
            Write-Host "=== native-fast (blocked GEMM / SIMD GEMV via Highway) ==="
            $Built += [pscustomobject]@{ Label = "native-fast"; Exe = (Build-Variant "build-native-fast" @(
                "-DMTL5_NATIVE_FAST_GEMM=ON", "-DMTL5_WITH_HIGHWAY=ON", "-DMTL5_NATIVE_ARCH=ON")) }
        }
        "openblas" {
            $obLib = Join-Path $OpenBlasRoot "lib\libopenblas.lib"
            $obBin = Join-Path $OpenBlasRoot "bin"
            if (-not (Test-Path $obLib)) {
                Write-Host "=== openblas: SKIPPED (no $obLib) ==="; break
            }
            Write-Host "=== openblas (official Windows binary, DYNAMIC_ARCH + LAPACK) ==="
            # The official mingw build bundles LAPACK, so libopenblas.lib resolves
            # both BLAS and the Fortran LAPACK symbols; put its DLL on PATH.
            $env:PATH = "$obBin;$env:PATH"
            $Built += [pscustomobject]@{ Label = "openblas"; Exe = (Build-Variant "build-openblas" @(
                "-DMTL5_WITH_BLAS=ON", "-DMTL5_WITH_LAPACK=ON",
                "-DBLAS_LIBRARIES=$obLib", "-DLAPACK_LIBRARIES=$obLib")) }
        }
        "mkl" {
            $mklLib = Join-Path $MklRoot "lib\mkl_rt.lib"
            $mklBin = Join-Path $MklRoot "bin"
            if (-not (Test-Path $mklLib)) {
                Write-Host "=== mkl: SKIPPED (no $mklLib) ==="; break
            }
            Write-Host "=== mkl ==="
            # The pip mkl-devel layout uses _dll-suffixed import libs and a
            # versioned mkl_rt.3.dll, which FindBLAS(Intel10_64lp) won't detect.
            # Link the single-dynamic-library dispatcher mkl_rt.lib directly (it
            # exports the Fortran BLAS/LAPACK symbols MTL5 needs), and put its
            # DLL dir on PATH so the benchmark resolves it at runtime.
            $env:PATH = "$mklBin;$env:PATH"
            $Built += [pscustomobject]@{ Label = "mkl"; Exe = (Build-Variant "build-mkl" @(
                "-DMTL5_WITH_BLAS=ON", "-DMTL5_WITH_LAPACK=ON",
                "-DBLAS_LIBRARIES=$mklLib", "-DLAPACK_LIBRARIES=$mklLib")) }
        }
        default { Write-Warning "unknown variant '$v', skipping" }
    }
}

# --- measurement phase -----------------------------------------------------
# NOT interleaved, unlike run_blocking_ab: bench_all truncates its --csv on every
# invocation, so multiple rounds would overwrite rather than accumulate. Doing it
# properly needs append support in bench_all and best-of-rounds in the analyzers.
# Tracked in #442.
$PreflightKv = Invoke-Preflight
Write-Host "`n=== measuring ==="
foreach ($b in $Built) {
    Write-Host "=== $($b.Label) ==="
    Run-Variant $b.Exe $b.Label
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
            "harness=run_sweeps.ps1",
            "harness_pinned_mask=0x$($PinMask.ToString('x'))",
            "harness_interleaved=0")
    }
}

Write-Host "`nDone. CSVs in $DataDir (each with a .sysinfo tag). Plot with e.g.:"
Write-Host "  python benchmarks/plot_results.py $DataDir\blas_sweep_*.csv --out $DataDir\blas_sweep_gflops.png"
