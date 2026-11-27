# The column sweep against the row kernel on a Windows machine (#884); the
# PowerShell counterpart of run_column_sweep.sh.
#
# Builds bench_column_sweep with CMake's Visual Studio generator, Release, in
# up to four variants:
#
#   msvc-generic     MSVC, no arch flags;
#   msvc-avx2        MSVC, /arch:AVX2;
#   clangcl-generic  the ClangCL toolset, no arch flags (if installed);
#   clangcl-native   ClangCL, /clang:-march=native.
#
# Each binary runs pinned to one CPU ($env:CPU, default 0) at high priority.
# The active power plan is recorded, not changed.
#
# Output: $env:OUT (default build-platform\column-sweep\<host>-<UTC time>)
# holds <variant>.csv, each with its .csv.sysinfo sidecar, and sysinfo.txt.
#
# Usage: powershell -File benchmarks\run_column_sweep.ps1 [bench args]
# 'Continue', not 'Stop': under Windows PowerShell 5.1 a native command's
# stderr (CMake's) would otherwise end the script; exit codes are checked.
$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $PSScriptRoot
$cpu = if ($env:CPU) { [int]$env:CPU } else { 0 }
# ProcessorAffinity is one pointer-width mask, of the process's processor
# group, and the bit at the sign position doesn't convert to an IntPtr: bits
# 0..62 in a 64-bit PowerShell, 0..30 in a 32-bit one.
$cpuLimit = [Math]::Min([IntPtr]::Size * 8 - 1, [Environment]::ProcessorCount)
if ($cpu -lt 0 -or $cpu -ge $cpuLimit) {
    Write-Host "CPU=$cpu is outside 0..$($cpuLimit - 1)"; exit 2
}
# One core: the public mult would otherwise thread over a pool of every CPU,
# all pinned to the one measured.
$env:MTL5_NUM_THREADS = '1'
$stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
$out = if ($env:OUT) { $env:OUT } else { Join-Path $root "build-platform\column-sweep\$env:COMPUTERNAME-$stamp" }
New-Item -ItemType Directory -Force -Path $out | Out-Null

# The checkout's revision, as the Linux runner records it; a COMMIT file for
# an exported tree with no .git.
$commit = (& git -C $root rev-parse --short HEAD 2>$null)
if ($LASTEXITCODE -ne 0 -or -not $commit) {
    $commit = if (Test-Path (Join-Path $root 'COMMIT')) { (Get-Content (Join-Path $root 'COMMIT')).Trim() } else { 'unknown' }
}
$plan = (powercfg /getactivescheme) -join ' '
# The session's record: preflight's machine state (report only, as before: the
# sweep has never gated on it), and what identifies the run. Each CSV's sidecar
# is this, plus its binary's --sysinfo and its build (#930).
. (Join-Path $PSScriptRoot 'sidecar.ps1')
$preflight = @(& (Join-Path $PSScriptRoot 'preflight.ps1') -Repo $root -ReportOnly -IgnorePath $out 2>$null)
$dirty = if (& git -C $root status --porcelain --untracked-files=no 2>$null) { '1' } else { '0' }
$long = (& git -C $root rev-parse --short=12 HEAD 2>$null)
if ($LASTEXITCODE -ne 0 -or -not $long) { $long = $commit }
@($preflight + @(
    "os=$((Get-CimInstance Win32_OperatingSystem).Caption)"
    "bench_cpus=$cpu"
    "clock_policy=not changed; $plan"
    "git_commit=$long"
    "git_dirty=$dirty"
    "cmake_build_type=Release"
)) | Set-Content (Join-Path $out 'sysinfo.txt')
$SidecarFail = $false

$variants = @(
    @{ name = 'msvc-generic';    toolset = $null;     flags = '' },
    @{ name = 'msvc-avx2';       toolset = $null;     flags = '/arch:AVX2' },
    @{ name = 'clangcl-generic'; toolset = 'ClangCL'; flags = '' },
    @{ name = 'clangcl-native';  toolset = 'ClangCL'; flags = '/clang:-march=native' }
)
foreach ($v in $variants) {
    $dir = Join-Path $out "build-$($v.name)"
    $cfg = @('-S', $root, '-B', $dir, '-DMTL5_BUILD_BENCHMARKS=ON', '-DMTL5_BUILD_TESTS=OFF',
             '-DMTL5_BUILD_EXAMPLES=OFF')
    if ($v.toolset) { $cfg += @('-T', $v.toolset) }
    if ($v.flags) { $cfg += "-DCMAKE_CXX_FLAGS=/EHsc $($v.flags)" }
    Write-Host "configuring $($v.name)"
    & cmake @cfg *> (Join-Path $out "$($v.name).configure.log")
    if ($LASTEXITCODE -ne 0) { Write-Host "  $($v.name): configure failed, skipped"; continue }
    & cmake --build $dir --config Release --target bench_column_sweep *> (Join-Path $out "$($v.name).build.log")
    if ($LASTEXITCODE -ne 0) { Write-Host "  $($v.name): build failed, skipped"; continue }
    $exe = Get-ChildItem -Path $dir -Recurse -Filter 'bench_column_sweep.exe' | Select-Object -First 1
    $compiler = (Select-String -Path (Join-Path $out "$($v.name).configure.log") -Pattern 'The CXX compiler identification is (.*)' |
                 Select-Object -First 1).Matches.Groups[1].Value
    Add-Content (Join-Path $out 'sysinfo.txt') "compiler_$($v.name)=$compiler; flags=$($v.flags)"
    Write-Host "running $($v.name) on cpu $cpu"
    $csv = Join-Path $out "$($v.name).csv"
    # Start-Process rejects an empty -ArgumentList, so pass it only with arguments.
    $sp = @{ FilePath = $exe.FullName; NoNewWindow = $true; PassThru = $true; RedirectStandardOutput = $csv }
    if ($args.Count -gt 0) { $sp.ArgumentList = $args }
    $p = Start-Process @sp
    # Read the handle now: without it, ExitCode is $null once the process ends.
    $null = $p.Handle
    # A 64-bit 1: an Int32's shift count wraps at 32, so 1 -shl 32 was CPU 0.
    # A run that could not be pinned is stopped and kept as failed: its
    # sidecar would claim bench_cpu for an unpinned run.
    $pinned = $true
    try {
        $p.ProcessorAffinity = [IntPtr]([long]1 -shl $cpu)
        $p.PriorityClass = 'High'
    } catch {
        $pinned = $false
        Write-Host "  $($v.name): could not pin to cpu ${cpu}: $($_.Exception.Message)"
        Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
    }
    $p.WaitForExit()
    if (-not $pinned -or $p.ExitCode -ne 0) {
        # A partial CSV is kept under another name, with no sidecar, so
        # column_sweep_table.py never reads it as a run.
        Write-Host "  $($v.name): failed (pinned $pinned, exit $($p.ExitCode)); its CSV is $($v.name).failed.csv"
        Move-Item -Force $csv (Join-Path $out "$($v.name).failed.csv")
        Remove-Item -Recurse -Force $dir
        continue
    }
    # The CSV's sidecar (benchmarks/check_sidecars.sh): the binary's keys, the
    # session's record, and this build's flags.
    $keys = @(& $exe.FullName --sysinfo)
    @(@("label=$($v.name)") + $keys + (Get-Content (Join-Path $out 'sysinfo.txt')) +
      @("cxx_flags=$($v.flags)", "compiler_identification=$compiler")) | Set-Content "$csv.sysinfo"
    $class = if ($args -contains '--quick') { 'quick' } else { 'full' }
    if (-not (Complete-Sidecar "$csv.sysinfo" @{ benchmark = 'bench_column_sweep'; class = $class })) {
        $SidecarFail = $true
    }
    Remove-Item -Recurse -Force $dir
}
Write-Host "results: $out"
if ($SidecarFail) {
    Write-Host "run_column_sweep.ps1: a sidecar lacks required keys (above); do not commit it as is"
    exit 1
}
