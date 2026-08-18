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
    # Arms to run, in the order they lead the first round. All are the same
    # source and differ only in which detected cache levels feed the blocking:
    #   default  none   -- the compile-time model MTL5 ships
    #   detected L1+L2  -- kc from L1, mc from L2
    #   kconly   L1     -- kc detected, mc from the default model
    #   mconly   L2     -- mc detected, kc from the default model
    #   ccap     L2     -- mconly PLUS the runtime C-strip bound on mc (#453)
    #   ccap2    L2     -- as ccap, but the bound charges the C strip ALONE
    # kconly/mconly exist because #430 implicated kc and exonerated mc without
    # ever varying them separately. Run all four in ONE session to settle it:
    #   -Arms "default,detected,kconly,mconly"
    [string]$Arms    = "detected,default",
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
[string[]]$ArmList = $Arms -split '[,\s]+' | Where-Object { $_ }
if ($ArmList.Count -lt 2) { throw "-Arms needs at least two arms to compare, got '$Arms'" }

# The rotation only cancels position effects when every arm leads an equal number
# of rounds, which needs -Rounds to be a multiple of the arm count. At 5 rounds
# over 4 arms the first arm leads twice and one never leads -- a residual bias of
# exactly the size the rotation exists to remove.
if ($Rounds % $ArmList.Count -ne 0) {
    $asked = $Rounds
    $Rounds = [int][Math]::Ceiling($Rounds / $ArmList.Count) * $ArmList.Count
    Write-Host "NOTE: -Rounds $asked over $($ArmList.Count) arms cannot balance the rotation;"
    Write-Host "      running $Rounds rounds so each arm leads $($Rounds / $ArmList.Count)."
}
$known = @("default", "detected", "kconly", "mconly", "ccap", "ccap2")
foreach ($a in $ArmList) {
    if ($known -notcontains $a) { throw "unknown arm '$a'; known arms: $($known -join ', ')" }
}
$targets = @("--build", $Full, "--config", "Release", "--target") +
           ($ArmList | ForEach-Object { "bench_blocking_ab_$_" }) + @("-j", "$Jobs")
if ((Invoke-Native "cmake" $targets (Join-Path $LogDir "blocking_ab.build")) -ne 0) {
    throw "build failed (see $LogDir\blocking_ab.build.err.log)"
}

function Get-ArmExe($arm) { Join-Path $Full "benchmarks\Release\bench_blocking_ab_$arm.exe" }
function Get-ArmCsv($arm) { Join-Path $DataDir "blocking_ab_$arm.csv" }
foreach ($a in $ArmList) {
    if (-not (Test-Path (Get-ArmExe $a))) { throw "missing $(Get-ArmExe $a) after build" }
}
# Shapes come from the FIRST arm and go to every arm, so the arms are never
# compared on different shapes.
$FirstArm = $ArmList[0]
$Det = Get-ArmExe $FirstArm

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

foreach ($a in $ArmList) {
    foreach ($f in @((Get-ArmCsv $a), "$(Get-ArmCsv $a).sysinfo")) {
        if (Test-Path $f) { Remove-Item $f -Force }
    }
}

# Rotate the arm order by round so every arm leads an equal share. Running one
# arm first every round would fold warm-up, frequency ramp and thermal drift into
# the ratio in a fixed direction; this generalises the two-arm alternation to any
# number of arms.
for ($round = 1; $round -le $Rounds; $round++) {
    $shiftBy = ($round - 1) % $ArmList.Count
    $order = @($ArmList[$shiftBy..($ArmList.Count - 1)])
    if ($shiftBy -gt 0) { $order += @($ArmList[0..($shiftBy - 1)]) }
    foreach ($T in $ThreadList) {
        $mask = Mask-ForThreads $T
        Write-Host ("== round {0}, T={1}, affinity mask 0x{2:x}, order: {3}" -f $round, $T, $mask, ($order -join ' '))
        foreach ($a in $order) { Run-Arm (Get-ArmExe $a) $a (Get-ArmCsv $a) $mask $T $Shapes }
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
if (Test-Path "$(Get-ArmCsv $FirstArm).sysinfo") {
    $m = Get-Content "$(Get-ArmCsv $FirstArm).sysinfo" | Select-String -Pattern '^git_commit=(.+)$' | Select-Object -First 1
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

foreach ($s in ($ArmList | ForEach-Object { "$(Get-ArmCsv $_).sysinfo" })) {
    if (Test-Path $s) {
        Add-Content -Path $s -Value $PreflightKv
        Add-Content -Path $s -Value $AfterKv
        # The INVOCATION, not just the harness name. Which cpus were pinned,
        # which arms ran, which ISA flag and which machine profile chose them are
        # what make a committed CSV re-runnable.
        Add-Content -Path $s -Value @(
            "binary_stale=$Stale",
            "harness=run_blocking_ab.ps1",
            "harness_profile=$(if ($env:BENCH_PROFILE) { $env:BENCH_PROFILE } else { 'none' })",
            "harness_rounds=$Rounds",
            "harness_reps=$Reps",
            "harness_arms=$($ArmList -join ',')",
            "harness_pcores=$PCores",
            "harness_threads=$Threads",
            "harness_arch=$(if ($Arch -ne '') { $Arch } else { 'none' })",
            "harness_dtype=$DType")
    }
}

Write-Host ""
Write-Host "wrote (+ .sysinfo sidecars):"
foreach ($a in $ArmList) { Write-Host "  $(Get-ArmCsv $a)" }
# Every arm against the baseline. `default` is the baseline when present -- it is
# what MTL5 ships, so every other arm is a proposed change to it.
$Base = if ($ArmList -contains "default") { "default" } else { $FirstArm }
# Arms that derived the SAME kc/mc measure the same thing, so comparing them
# checks the SESSION rather than the code -- and the analyzer fails it when they
# disagree systematically (#430).
# By HEADER NAME, not column number. The CSV has grown columns twice (`pool`,
# then the mc_used group), so kc sits at field 9 in the pre-`pool` files still
# committed under benchmarks/data and at field 10 in current ones -- a positional
# read silently returns (mc,nc) for the older layout.
function Get-ArmBlocking($arm) {
    $csv = Get-ArmCsv $arm
    if (-not (Test-Path $csv)) { return "" }
    $lines = Get-Content $csv | Select-Object -First 2
    if ($lines.Count -lt 2) { return "" }
    $head = $lines[0] -split ','
    $row  = $lines[1] -split ','
    $kc = [array]::IndexOf($head, 'kc'); $mc = [array]::IndexOf($head, 'mc')
    if ($kc -lt 0 -or $mc -lt 0) { return "" }
    return "$($row[$kc]),$($row[$mc])"
}
$checks = @()
for ($i = 0; $i -lt $ArmList.Count; $i++) {
    for ($j = $i + 1; $j -lt $ArmList.Count; $j++) {
        if ((Get-ArmBlocking $ArmList[$i]) -eq (Get-ArmBlocking $ArmList[$j]) -and (Get-ArmBlocking $ArmList[$i]) -ne "") {
            $checks += "  python benchmarks/analyze_blocking_ab.py `"$(Get-ArmCsv $ArmList[$i])`" `"$(Get-ArmCsv $ArmList[$j])`""
        }
    }
}
if ($checks.Count -gt 0) {
    Write-Host ""
    Write-Host "consistency checks -- these arms derived IDENTICAL kc/mc, so they must agree;"
    Write-Host "the analyzer fails them if the machine drifted more than the effect under test:"
    $checks | ForEach-Object { Write-Host $_ }
}

Write-Host "compare with:"
foreach ($a in $ArmList) {
    if ($a -eq $Base) { continue }
    Write-Host "  python benchmarks/analyze_blocking_ab.py `"$(Get-ArmCsv $a)`" `"$(Get-ArmCsv $Base)`""
}
