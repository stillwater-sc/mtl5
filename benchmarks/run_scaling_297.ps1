<#
.SYNOPSIS
    Windows/MSVC port of run_scaling_297.sh: native 1->N scaling of every
    threaded kernel family, one CSV per family (backend column "native-t<T>").

.DESCRIPTION
    Builds the native-fast benchmarks and sweeps MTL5_NUM_THREADS over T, pinning
    each run to the first T physical cores via a Windows affinity bitmask (SMT
    siblings excluded). Families: gemm-rect, lu/qr/cholesky, ewise (bench_all),
    and level-scheduled sparse triangular solves (bench_sparse).

.PARAMETER LapackSizes
    Dense factorization sizes. Default 1024,2048 (the i7 page also ran 4096, but
    the generic native factorizations are ~3x slower under MSVC, so 4096 at T=1
    is impractical here; raise deliberately).

.PARAMETER SparseSizes
    Sparse 2-D grid sides (default 100,160). Cost is dominated by the untimed,
    serial factorization and is paid once per thread count -- see run_sweeps notes.
#>
[CmdletBinding()]
param(
    [string]$PCores = "0,2,4,6,8,10,12,14",
    [string]$Threads = "1,2,4,8",
    [string]$LapackSizes = "1024,2048",
    [string]$SparseSizes = "100,160",
    [string]$OutDir = "benchmarks\data",
    [int]$Jobs = [Environment]::ProcessorCount
)

$RepoRoot = Split-Path -Parent $PSScriptRoot
$DataDir  = Join-Path $RepoRoot $OutDir
$LogDir   = Join-Path $DataDir "logs"
New-Item -ItemType Directory -Force -Path $DataDir, $LogDir | Out-Null
[int[]]$PCoreList  = $PCores  -split ',' | ForEach-Object { [int]$_ }
[int[]]$ThreadList = $Threads -split ',' | ForEach-Object { [int]$_ }

function Mask-ForThreads {
    param([int]$T)
    if ($T -gt $PCoreList.Count) { throw "T=$T exceeds $($PCoreList.Count) physical cores in -PCores." }
    $m = 0L; for ($i = 0; $i -lt $T; $i++) { $m = $m -bor (1L -shl $PCoreList[$i]) }; return $m
}

function Invoke-Native {
    param([string]$Exe, [string[]]$NativeArgs, [string]$LogBase)
    # Quote arguments containing spaces: Start-Process joins -ArgumentList with
    # spaces and quotes NOTHING, so a repo under "C:\Users\Some Name\..." reaches
    # cmake as two broken arguments (#438). The other harnesses already did this.
    $quoted = $NativeArgs | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }
    # NOT Start-Process -Wait: -Wait also waits for every DESCENDANT, and an MSVC
    # build leaves MSBuild's reusable worker nodes alive for ~15 minutes, so each
    # `cmake --build` blocked until they timed out. Wait on the process itself;
    # touching .Handle first keeps ExitCode readable in Windows PowerShell 5.1.
    $p = Start-Process -FilePath $Exe -ArgumentList $quoted -PassThru -NoNewWindow `
                       -RedirectStandardOutput "$LogBase.out.log" -RedirectStandardError "$LogBase.err.log"
    $null = $p.Handle
    $p.WaitForExit()
    return $p.ExitCode
}

# No lingering MSBuild worker nodes after this script's builds (see Invoke-Native).
$env:MSBUILDDISABLENODEREUSE = '1'

# native-fast build (bench_all + bench_sparse).
$build = Join-Path $RepoRoot "build-scaling-297"
Write-Host "=== configure + build native-fast benchmarks (build-scaling-297) ==="
if ((Invoke-Native "cmake" @("-B", $build, "-DMTL5_BUILD_BENCHMARKS=ON", "-DMTL5_BUILD_TESTS=OFF",
    "-DMTL5_BUILD_EXAMPLES=OFF", "-DCMAKE_BUILD_TYPE=Release", "-DMTL5_NATIVE_FAST_GEMM=ON",
    "-DMTL5_WITH_HIGHWAY=ON", "-DMTL5_NATIVE_ARCH=ON") (Join-Path $LogDir "297.configure")) -ne 0) {
    throw "configure failed (see $LogDir\297.configure.err.log)"
}
if ((Invoke-Native "cmake" @("--build", $build, "--config", "Release", "--target", "bench_all", "bench_sparse", "-j", "$Jobs") (Join-Path $LogDir "297.build")) -ne 0) {
    throw "build failed (see $LogDir\297.build.err.log)"
}
$BA = Join-Path $build "benchmarks\Release\bench_all.exe"
$BS = Join-Path $build "benchmarks\Release\bench_sparse.exe"

# Sweep THREADS for one family, pinned, merging per-T CSVs into scaling_<name>.csv.
function Run-Native {
    param([string]$Name, [string]$Exe, [string[]]$BenchArgs)
    $out = Join-Path $DataDir "scaling_${Name}.csv"
    if (Test-Path $out) { Remove-Item $out }
    $first = $true
    foreach ($T in $ThreadList) {
        $mask = Mask-ForThreads $T
        $env:MTL5_NUM_THREADS = "$T"; $env:OMP_NUM_THREADS = "$T"
        $tmp = Join-Path $DataDir "scaling_${Name}_t${T}.csv"
        Write-Host ("  {0}  T={1}  affinity mask 0x{2:x}" -f $Name, $T, $mask)
        $psi = New-Object System.Diagnostics.ProcessStartInfo
        $psi.FileName = $Exe
        $psi.Arguments = (($BenchArgs + @("--label", "native-t${T}", "--csv", $tmp)) |
            ForEach-Object { if ($_ -match '\s') { '"'+$_+'"' } else { $_ } }) -join ' '
        $psi.UseShellExecute = $false; $psi.RedirectStandardOutput = $true
        $p = [System.Diagnostics.Process]::Start($psi)
        try { $p.ProcessorAffinity = [IntPtr]$mask; $p.PriorityClass = 'High' } catch { Write-Warning $_ }
        $p.StandardOutput.ReadToEnd() | Out-Null
        $p.WaitForExit()
        if ($p.ExitCode -ne 0) { throw "bench failed ($Name T=$T)" }
        if ($first) { Get-Content $tmp | Set-Content $out; $first = $false }
        else { Get-Content $tmp | Select-Object -Skip 1 | Add-Content $out }
        Remove-Item $tmp -Force
    }
    Write-Host "  -> $out"
}

Write-Host "=== gemm-rect (multi-loop 2D grid) ==="
Run-Native "gemm_rect" $BA @("--suite", "gemm-rect")
Write-Host "=== dense factorizations (lu / qr / cholesky) ==="
Run-Native "lu"   $BA @("--suite", "lu",       "--lapack-sizes", $LapackSizes)
Run-Native "qr"   $BA @("--suite", "qr",       "--lapack-sizes", $LapackSizes)
Run-Native "chol" $BA @("--suite", "cholesky", "--lapack-sizes", $LapackSizes)
Write-Host "=== element-wise sweeps ==="
Run-Native "ewise" $BA @("--suite", "ewise")
Write-Host "=== sparse triangular solves (level-scheduled) ==="
Run-Native "sparse" $BS @("--sizes", $SparseSizes)

Write-Host "`nDone. Analyze with:"
Write-Host "  python benchmarks/analyze_scaling.py $DataDir\scaling_*.csv --plot docs/img/benchmarks/ryzen/kernel-scaling-297.png"
