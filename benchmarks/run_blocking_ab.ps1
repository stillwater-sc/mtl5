<#
.SYNOPSIS
    Windows/MSVC port of run_blocking_ab.sh: A/B the detected cache blocking
    against the compile-time defaults (#426, #430, #432).

.DESCRIPTION
    Both arms are the same source. bench_blocking_ab_detected is built with
    MTL5_ENABLE_CACHE_DETECTION; bench_blocking_ab_default is what MTL5 ships.
    Nothing else differs, so a difference in throughput is a difference in kc/mc.
    Both targets live in ONE build tree, so this configures and builds once.

    Protocol matches the bash driver and the rest of docs/benchmarks/:
      * PINNED -- a Windows affinity BITMASK (OR of one bit per physical core),
        since there is no taskset. SMT siblings excluded.
      * INTERLEAVED with the order alternating per round, so warm-up and thermal
        drift do not accrue to one arm.
      * MIN of N -- the binary reports the minimum of -Reps runs per point.
      * SHAPES DERIVED ONCE from the detected arm and passed to both, so the arms
        are never compared on different shapes.

    TWO WINDOWS-SPECIFIC CAVEATS, both worth reading before trusting a result:

    1. ISA SELECTION. MTL5_NATIVE_ARCH is `if(MTL5_NATIVE_ARCH AND NOT MSVC)` in
       CMakeLists.txt -- a NO-OP under MSVC. MSVC x64 defaults to SSE2, so
       without an explicit /arch: flag Highway selects a 128-bit target and the
       run measures blocking for 2 doubles. On a Zen 4 or Golden Cove part, whose
       interest is AVX-512 / AVX2, that answers the wrong question while looking
       perfectly clean. -Arch therefore defaults to /arch:AVX512 and is echoed
       into the log. Set -Arch "" only if you intend to measure the baseline.

    2. DETECTION AND AFFINITY. On Linux, cache detection reads sysfs restricted
       to the process's affinity mask, so it describes the pinned cores. Windows
       has no such path and falls back to CPUID, which describes whichever core
       the thread is on -- and .NET can only set ProcessorAffinity AFTER the
       process starts, so detection may already have run on another core. That is
       harmless on a HOMOGENEOUS part (every core reports the same), which covers
       ryzen-9-8945hs. On a hybrid Windows machine the detected arm's figures
       would not be reproducible; do not use this driver there without fixing
       #432's Windows half first.

.PARAMETER PCores
    First-logical-processor id of each physical core, in the order to fill. The
    default 0,2,4,... matches an SMT machine whose sibling threads are adjacent
    (verify with topology_probe: physical core -> logical mask).

.PARAMETER Threads
    Thread counts to sweep (default "1,8"). Each must be <= the number of -PCores.

.PARAMETER Arch
    Compiler ISA flag. Default /arch:AVX512 -- see caveat 1.

.EXAMPLE
    pwsh benchmarks/run_blocking_ab.ps1 -OutDir "benchmarks\data\ryzen-9-8945hs"
#>
[CmdletBinding()]
param(
    # Comma strings, not [int[]]: passing "1,8" to an [int[]] param via
    # `powershell -File` coerces it to the single integer 18 (commas read as
    # digit-group separators). Parse the strings into int lists ourselves.
    [string]$PCores  = "0,2,4,6,8,10,12,14",
    [string]$Threads = "1,8",
    [int]$Reps       = 5,
    [int]$Rounds     = 5,
    [string]$DType   = "double",
    [string]$Shapes  = "",          # empty: derive from the machine (preferred)
    [string]$Arch    = "/arch:AVX512",
    [string]$OutDir  = "",          # REQUIRED -- one directory per machine

    [string]$BuildDir = "build-blocking-ab",
    [int]$Jobs       = [Environment]::ProcessorCount,

    # Permit a dirty working tree (still recorded as tree_git_dirty=1). A switch,
    # not only the ALLOW_DIRTY environment variable, because the obvious way to
    # set that variable does not work here: `set ALLOW_DIRTY=1` is cmd.exe syntax,
    # and in PowerShell `set` aliases Set-Variable, so the run fails again with
    # the same message and no clue why.
    [switch]$AllowDirty
)

[int[]]$PCoreList  = $PCores  -split ',' | ForEach-Object { [int]$_ }
[int[]]$ThreadList = $Threads -split ',' | ForEach-Object { [int]$_ }

# Validate -PCores BEFORE -Threads, because -Threads is checked against its
# count. "0,0" would otherwise pass a two-thread run whose mask is 0x1: two
# nominal threads sharing one logical processor, producing scaling data that is
# wrong rather than merely noisy. A 64-bit process affinity mask addresses ids
# 0-63, so anything outside that cannot be pinned at all.
$seen = @{}
foreach ($c in $PCoreList) {
    if ($c -lt 0 -or $c -gt 63) {
        throw "-PCores: id $c is outside the 0-63 range a process affinity mask can address."
    }
    if ($seen.ContainsKey($c)) {
        throw "-PCores: id $c appears more than once. Give one logical id per PHYSICAL core; duplicates silently shrink the affinity mask."
    }
    $seen[$c] = $true
}

foreach ($T in $ThreadList) {
    if ($T -lt 1) { throw "-Threads: '$T' is not a positive integer." }
    if ($T -gt $PCoreList.Count) {
        throw "-Threads $T exceeds the $($PCoreList.Count) physical core(s) in -PCores ($PCores). Set -PCores to one logical id per physical core on this machine."
    }
}
# -OutDir is REQUIRED and must name a per-machine directory. The CSVs are named
# by arm, not by machine, and this script deletes them before writing -- so a
# shared default silently destroys another machine's committed results. That is
# not hypothetical: a Zen 4 run overwrote the i7's data exactly this way.
if ($OutDir -eq "") {
    $existing = (Get-ChildItem -Directory (Join-Path (Split-Path -Parent $PSScriptRoot) "benchmarks\data") -ErrorAction SilentlyContinue | ForEach-Object { "  benchmarks\data\$($_.Name)" }) -join "`n"
    throw "-OutDir is required: give this machine its own directory, e.g.`n  -OutDir `"benchmarks\data\ryzen-9-8945hs`"`nExisting machine directories:`n$existing"
}
if ($Reps -lt 1)   { throw "-Reps must be > 0." }
if ($Rounds -lt 1) { throw "-Rounds must be > 0." }
if ($DType -ne "double" -and $DType -ne "float") { throw "-DType must be double or float." }

# See run_sweeps.ps1 for why $ErrorActionPreference is left at Continue.
$RepoRoot = Split-Path -Parent $PSScriptRoot
$DataDir  = Join-Path $RepoRoot $OutDir
$LogDir   = Join-Path $DataDir "logs"
New-Item -ItemType Directory -Force -Path $DataDir, $LogDir | Out-Null

# Affinity mask covering the first T physical cores (one logical id per core).
function Mask-ForThreads {
    param([int]$T)
    $m = 0L
    for ($i = 0; $i -lt $T; $i++) { $m = $m -bor (1L -shl $PCoreList[$i]) }
    return $m
}

# -ArgumentList joins an array with spaces, so an element that CONTAINS a space
# would be split into two arguments. Windows repo paths routinely sit under
# "C:\Users\First Last\...", which would otherwise break `-B <builddir>` in a way
# that surfaces as a confusing CMake error rather than a quoting error.
function Quote-Arg {
    param([string]$a)
    if ($a -match '\s') { return '"' + $a + '"' } else { return $a }
}

function Invoke-Native {
    param([string]$Exe, [string[]]$NativeArgs, [string]$LogBase)
    $quoted = @($NativeArgs | ForEach-Object { Quote-Arg $_ })
    $p = Start-Process -FilePath $Exe -ArgumentList $quoted -PassThru -NoNewWindow -Wait `
                       -RedirectStandardOutput "$LogBase.out.log" -RedirectStandardError "$LogBase.err.log"
    return $p.ExitCode
}

# --- preflight, before the build ---------------------------------------------
# Gate FIRST, not after the build: a dirty tree or a competing compile should
# cost seconds, not a full configure-and-build. It is also the only order that
# works -- this harness builds into the repo and writes its cmake logs beside the
# CSVs, so gating afterwards meant the run tripped over ITS OWN output and failed
# dirty every time (the logs are now gitignored as well).
$TMax = ($ThreadList | Measure-Object -Maximum).Maximum
$PreflightScript = Join-Path $PSScriptRoot "preflight.ps1"
function Invoke-Preflight {
    $kv = & $PreflightScript -Threads $TMax -Repo $RepoRoot -AllowDirty:$AllowDirty -IgnorePath $DataDir
    if ($LASTEXITCODE -ne 0) {
        Write-Host "preflight failed -- not measuring. Fix the above, or pass -AllowDirty"
        Write-Host "if you accept a dirty tree (it is then recorded as such in the sidecar)."
        exit 1
    }
    return $kv
}
Invoke-Preflight | Out-Null

# --- build both arms from one tree ------------------------------------------
$Full = Join-Path $RepoRoot $BuildDir
$cfg  = @("-B", $Full,
          "-DMTL5_BUILD_BENCHMARKS=ON", "-DMTL5_BUILD_TESTS=OFF", "-DMTL5_BUILD_EXAMPLES=OFF",
          "-DCMAKE_BUILD_TYPE=Release", "-DMTL5_WITH_HIGHWAY=ON",
          # ALWAYS set it, empty included. Omitting it for -Arch "" would leave a
          # previous /arch:AVX512 sitting in this build tree's CMake cache while
          # the log below announced an SSE2 baseline -- the build would be AVX-512
          # and the record would say otherwise. That is precisely the failure this
          # script exists to prevent, so it must not be reintroduced by the escape
          # hatch.
          "-DCMAKE_CXX_FLAGS=$Arch")

Write-Host "=== configure ($BuildDir)"
Write-Host ("    ISA flag: {0}" -f $(if ($Arch -ne "") { $Arch } else { "<none> -- measuring the MSVC default (SSE2)" }))
if ((Invoke-Native "cmake" $cfg (Join-Path $LogDir "blocking_ab.configure")) -ne 0) {
    throw "configure failed (see $LogDir\blocking_ab.configure.err.log)"
}
Write-Host "=== build"
$targets = @("--build", $Full, "--config", "Release",
             "--target", "bench_blocking_ab_detected", "bench_blocking_ab_default", "-j", "$Jobs")
if ((Invoke-Native "cmake" $targets (Join-Path $LogDir "blocking_ab.build")) -ne 0) {
    throw "build failed (see $LogDir\blocking_ab.build.err.log)"
}

$Det = Join-Path $Full "benchmarks\Release\bench_blocking_ab_detected.exe"
$Def = Join-Path $Full "benchmarks\Release\bench_blocking_ab_default.exe"
foreach ($exe in @($Det, $Def)) {
    if (-not (Test-Path $exe)) { throw "missing $exe after build" }
}

# --- run one arm, pinned -----------------------------------------------------
# Affinity is applied immediately after Start(); see caveat 2 in the header for
# what that means for detection on a hybrid part.
function Run-Arm {
    param([string]$Exe, [string]$Label, [string]$Csv, [long]$Mask, [int]$T, [string]$ShapeSpec)
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName  = $Exe
    $psi.Arguments = "--label $Label --dtype $DType --threads $T --reps $Reps --shapes `"$ShapeSpec`" --csv `"$Csv`""
    $psi.UseShellExecute = $false          # stderr inherits the console: the
    $psi.RedirectStandardOutput = $true    # per-arm blocking params stay visible
    $env:MTL5_NUM_THREADS = "$T"
    $p = [System.Diagnostics.Process]::Start($psi)
    # Pinning is FATAL, not advisory. Warning and continuing would append samples
    # to a CSV whose whole claim is that they were taken on N pinned physical
    # cores -- unpinned numbers filed as pinned ones are worse than no numbers.
    try { $p.ProcessorAffinity = [IntPtr]$Mask }
    catch {
        try { $p.Kill() } catch { }
        throw "could not pin $Label to mask 0x$('{0:x}' -f $Mask): $_"
    }
    # Priority is a nice-to-have and stays advisory.
    try { $p.PriorityClass = 'High' } catch { Write-Warning "priority class: $_" }
    $p.StandardOutput.ReadToEnd() | Out-Null
    $p.WaitForExit()
    if ($p.ExitCode -ne 0) { throw "$Label arm failed (T=$T, exit $($p.ExitCode))" }
}

# --- preflight, again, now the build is done ---------------------------------
# The builds heat the machine, so the temperature that belongs in the sidecar is
# the one the MEASUREMENTS start at, not the one the session started at. This
# record is what travels with the data.
$PreflightKv = Invoke-Preflight

# --- shapes: derived ONCE, from the detected arm, at the largest thread count -
if ($Shapes -eq "") {
    $maskMax = Mask-ForThreads $TMax
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName  = $Det
    $psi.Arguments = "--suggest-shapes --threads $TMax --dtype $DType"
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $p = [System.Diagnostics.Process]::Start($psi)
    try { $p.ProcessorAffinity = [IntPtr]$maskMax }
    catch {
        try { $p.Kill() } catch { }
        throw "could not pin shape discovery to mask 0x$('{0:x}' -f $maskMax): $_"
    }
    $Shapes = ($p.StandardOutput.ReadToEnd()).Trim()
    $p.WaitForExit()
    # Exit code first: a failing --suggest-shapes that still wrote something to
    # stdout would otherwise feed a garbage shape list to BOTH arms, and the run
    # would complete and compare them happily.
    if ($p.ExitCode -ne 0) { throw "shape discovery failed (exit $($p.ExitCode)) from $Det" }
    if ($Shapes -eq "") { throw "could not derive shapes from $Det" }
}
Write-Host "shapes: $Shapes"

$CsvDet = Join-Path $DataDir "blocking_ab_detected.csv"
$CsvDef = Join-Path $DataDir "blocking_ab_default.csv"
foreach ($f in @($CsvDet, $CsvDef, "$CsvDet.sysinfo", "$CsvDef.sysinfo")) {
    if (Test-Path $f) { Remove-Item $f -Force }
}

for ($round = 1; $round -le $Rounds; $round++) {
    foreach ($T in $ThreadList) {
        $mask = Mask-ForThreads $T
        Write-Host ("== round {0}, T={1}, affinity mask 0x{2:x}" -f $round, $T, $mask)
        if ($round % 2 -eq 1) {
            Run-Arm $Det "detected" $CsvDet $mask $T $Shapes
            Run-Arm $Def "default"  $CsvDef $mask $T $Shapes
        } else {
            Run-Arm $Def "default"  $CsvDef $mask $T $Shapes
            Run-Arm $Det "detected" $CsvDet $mask $T $Shapes
        }
    }
}

# Thermal AFTER the session, next to the before reading: a run that ended hot and
# a configuration that is simply slow give the same GFLOP/s, and only the pair of
# temperatures separates them. Appended once at the end because each benchmark
# invocation truncates its own sidecar.
# If the postflight read fails the MEASUREMENTS are still good -- they are
# already on disk, and discarding a completed session over a failed thermometer
# read would destroy data to punish a missing probe. What must not happen is the
# key going missing silently, so a failure is recorded as `unavailable` and said
# out loud. `unavailable` is a fact; an absent key is a reader's guess. This
# matters more on Windows than anywhere else, where the sensor is usually absent.
$AfterKv = & $PreflightScript -Phase after
if ($LASTEXITCODE -ne 0) {
    Write-Warning "postflight thermal read failed; recording thermal_after_c=unavailable. The measurements themselves are unaffected."
    $AfterKv = "thermal_after_c=unavailable"
}
if (-not ($AfterKv -match 'thermal_after_c=')) { $AfterKv = "thermal_after_c=unavailable" }

# Did we measure the code the tree is on? The binary records the commit it was
# BUILT from (mtl/build_info.hpp); preflight records where the tree is NOW. They
# differ whenever someone edits, forgets to rebuild, and measures -- an easy
# error that hides completely in the numbers.
$TreeCommit  = ($PreflightKv | Select-String -Pattern '^tree_git_commit=(.+)$').Matches.Groups[1].Value
$BuildCommit = ""
if (Test-Path "$CsvDet.sysinfo") {
    $m = Get-Content "$CsvDet.sysinfo" | Select-String -Pattern '^git_commit=(.+)$' | Select-Object -First 1
    if ($m) { $BuildCommit = $m.Matches.Groups[1].Value }
}
$Stale = "unknown"
if ($TreeCommit -and $BuildCommit -and $TreeCommit -ne "unknown" -and $BuildCommit -ne "unknown") {
    if ($TreeCommit -eq $BuildCommit) {
        $Stale = "0"
    } else {
        $Stale = "1"
        Write-Warning "binary was built from $BuildCommit but the tree is on $TreeCommit."
        Write-Warning "These numbers describe the BUILT code; rebuild if that is not what you meant."
    }
}

foreach ($s in @("$CsvDet.sysinfo", "$CsvDef.sysinfo")) {
    if (Test-Path $s) {
        Add-Content -Path $s -Value $PreflightKv
        Add-Content -Path $s -Value $AfterKv
        Add-Content -Path $s -Value @(
            "binary_stale=$Stale",
            "harness=run_blocking_ab.ps1",
            "harness_rounds=$Rounds",
            "harness_reps=$Reps")
    }
}

Write-Host ""
Write-Host "wrote $CsvDet and $CsvDef (+ .sysinfo sidecars)"
Write-Host "compare with: python benchmarks/analyze_blocking_ab.py `"$CsvDet`" `"$CsvDef`""
