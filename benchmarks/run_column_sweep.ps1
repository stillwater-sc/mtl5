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
# holds <variant>.csv and sysinfo.txt.
#
# Usage: powershell -File benchmarks\run_column_sweep.ps1 [bench args]
# 'Continue', not 'Stop': under Windows PowerShell 5.1 a native command's
# stderr (CMake's) would otherwise end the script; exit codes are checked.
$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $PSScriptRoot
$cpu = if ($env:CPU) { [int]$env:CPU } else { 0 }
# One core: the public mult would otherwise thread over a pool of every CPU,
# all pinned to the one measured.
$env:MTL5_NUM_THREADS = '1'
$stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
$out = if ($env:OUT) { $env:OUT } else { Join-Path $root "build-platform\column-sweep\$env:COMPUTERNAME-$stamp" }
New-Item -ItemType Directory -Force -Path $out | Out-Null

$commit = if (Test-Path (Join-Path $root 'COMMIT')) { (Get-Content (Join-Path $root 'COMMIT')).Trim() } else { 'unknown' }
$plan = (powercfg /getactivescheme) -join ' '
@(
    "host=$env:COMPUTERNAME"
    "cpu=$((Get-CimInstance Win32_Processor).Name.Trim())"
    "os=$((Get-CimInstance Win32_OperatingSystem).Caption)"
    "bench_cpu=$cpu"
    "clock_policy=not changed; $plan"
    "commit=$commit"
) | Set-Content (Join-Path $out 'sysinfo.txt')

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
    $p.ProcessorAffinity = [IntPtr](1 -shl $cpu)
    $p.PriorityClass = 'High'
    $p.WaitForExit()
    Remove-Item -Recurse -Force $dir
}
Write-Host "results: $out"
