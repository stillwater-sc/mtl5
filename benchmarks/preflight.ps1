<#
.SYNOPSIS
Preflight: gate a measurement on machine state, and record that state (#442).

.DESCRIPTION
The Windows half of benchmarks/preflight.sh; the contract is identical and the
key names it emits are the same, so a Zen 4 sidecar and an i7 sidecar can be read
side by side.

Two jobs:
  1. FAIL on conditions that make a measurement meaningless (dirty tree,
     competing build, more threads than cpus). Not warn -- a warning scrolls past
     and the CSV gets committed anyway.
  2. EMIT key=value lines for the sidecar, so machine state is IN the data.

WHAT WINDOWS CANNOT TELL US. Two of the probes are usually blank here, and both
are recorded as `unavailable` rather than guessed:

  * Temperature. MSAcpi_ThermalZoneTemperature is an ACPI optional feature that
    most desktop firmware does not implement; on the Zen 4 machine it returns
    nothing. The thermal gate therefore only FAILS where a sensor and a limit are
    both readable -- refusing to run would make the machine unbenchmarkable, and
    passing silently would hide the gap.
  * Turbo state. Reachable only by parsing powercfg, and not on every SKU.

.PARAMETER Phase
'before' (default) runs every gate and emits the full record. 'after' emits
thermal_after_c only; call it once the run finishes, since a session that ends
hot and one that ran slow look identical in GFLOP/s alone.

.PARAMETER Threads
The largest thread count the run will ask for.

.PARAMETER Repo
Repository to check. Defaults to the parent of this script.

.PARAMETER ReportOnly
Probe and emit, never fail. For CI and for inspecting a machine; NOT for taking
measurements.

.NOTES
Environment: ALLOW_DIRTY=1 permits a dirty tree (still recorded),
MIN_THERMAL_HEADROOM_C (default 15) sets the required margin.
Exit: 0 all gates pass, 1 a gate failed, 2 bad usage.
#>
param(
    [ValidateSet('before', 'after')][string]$Phase = 'before',
    [int]$Threads = 0,
    [string]$Repo = '',
    [switch]$ReportOnly
)

$ErrorActionPreference = 'Continue'

if (-not $Repo) { $Repo = Split-Path -Parent $PSScriptRoot }
$margin = if ($env:MIN_THERMAL_HEADROOM_C) { [double]$env:MIN_THERMAL_HEADROOM_C } else { 15.0 }

$script:failures = 0
function Emit($k, $v) { Write-Output "$k=$v" }
function GateFailed($msg) {
    Write-Error -Message "preflight: FAIL: $msg" -ErrorAction Continue
    $script:failures++
}
function Warn($msg) { Write-Warning "preflight: $msg" }

# -- thermal ----------------------------------------------------------------
# Tenths of a Kelvin, per the ACPI spec. Absent on most desktop firmware.
function Read-Thermal {
    $r = [pscustomobject]@{ Temp = 'unavailable'; Limit = 'unavailable'; Sensor = 'unavailable' }
    try {
        $z = Get-CimInstance -Namespace 'root/wmi' -ClassName 'MSAcpi_ThermalZoneTemperature' -ErrorAction Stop |
             Select-Object -First 1
        if ($z) {
            $r.Temp = '{0:F1}' -f (($z.CurrentTemperature / 10.0) - 273.15)
            $r.Sensor = "acpi:$($z.InstanceName)"
            # CriticalTripPoint is where the machine shuts down, not where it
            # throttles. Gate on it anyway when it is all we have -- it is still
            # a real limit -- but the headroom it yields is generous.
            if ($z.CriticalTripPoint -gt 0) {
                $r.Limit = '{0:F1}' -f (($z.CriticalTripPoint / 10.0) - 273.15)
            }
        }
    } catch {
        # Not implemented by this firmware; leave everything 'unavailable'.
    }
    return $r
}

$thermal = Read-Thermal

if ($Phase -eq 'after') {
    Emit 'thermal_after_c' $thermal.Temp
    exit 0
}

Emit 'preflight_version' 1
Emit 'preflight_host'    $env:COMPUTERNAME
Emit 'preflight_kernel'  ([System.Environment]::OSVersion.VersionString -replace ' ', '_')

# -- working tree -----------------------------------------------------------
# The BINARY records the commit it was built from (mtl/build_info.hpp); this
# records the commit the tree is on NOW. If they disagree the binary is stale
# relative to the checkout.
$treeCommit = 'unknown'
$treeDirty = 'unknown'
if (Get-Command git -ErrorAction SilentlyContinue) {
    $sha = & git -C $Repo rev-parse --short=12 HEAD 2>$null
    if ($LASTEXITCODE -eq 0 -and $sha) {
        $treeCommit = $sha.Trim()
        $status = & git -C $Repo status --porcelain --untracked-files=normal 2>$null
        if ($LASTEXITCODE -eq 0) {
            if ([string]::IsNullOrWhiteSpace($status -join '')) {
                $treeDirty = '0'
            } else {
                $treeDirty = '1'
                if ($env:ALLOW_DIRTY -eq '1') {
                    Warn "working tree is dirty; ALLOW_DIRTY=1, recording tree_git_dirty=1"
                } else {
                    GateFailed "working tree is dirty -- this result could not be reproduced. Commit, stash, or set ALLOW_DIRTY=1 to record it as dirty."
                }
            }
        }
    }
}
Emit 'tree_git_commit' $treeCommit
Emit 'tree_git_dirty'  $treeDirty

# -- competing load ---------------------------------------------------------
# A competing compile does not make the run noisy, it makes it wrong: the arms
# are interleaved, so a build finishing mid-session penalises whichever arm was
# running then, and the difference is reported as a result.
#
# Matched on process NAME. Matching command lines caught the shell that had
# merely typed the benchmark's path on the POSIX side, which failed every clean
# run.
$busyNames = @('make', 'ninja', 'cmake', 'MSBuild', 'cl', 'link', 'devenv', 'cc1plus')
$busy = @()
foreach ($n in $busyNames) {
    Get-Process -Name $n -ErrorAction SilentlyContinue | ForEach-Object { $busy += "$($_.ProcessName)($($_.Id))" }
}
Get-Process -Name 'bench_*' -ErrorAction SilentlyContinue |
    Where-Object { $_.Id -ne $PID } |
    ForEach-Object { $busy += "$($_.ProcessName)($($_.Id))" }

if ($busy.Count -gt 0) {
    Emit 'competing_load' ($busy -join ',')
    GateFailed "a build or benchmark is already running: $($busy -join ' ')"
} else {
    Emit 'competing_load' 'none'
}

# -- cores and thread budget ------------------------------------------------
# ProcessorCount reflects the affinity this process actually has, which is the
# number a thread budget must fit into; NumberOfLogicalProcessors is the machine.
$affinity = [System.Environment]::ProcessorCount
$online = $affinity
try {
    $cs = Get-CimInstance -ClassName Win32_ComputerSystem -ErrorAction Stop
    if ($cs.NumberOfLogicalProcessors) { $online = $cs.NumberOfLogicalProcessors }
} catch { }
Emit 'cpu_online'   $online
Emit 'cpu_affinity' $affinity
if ($Threads -gt 0 -and $Threads -gt $affinity) {
    GateFailed "$Threads threads requested but only $affinity cpu(s) available."
}

# -- frequency policy -------------------------------------------------------
# Recorded and warned about, not failed: every machine in systems.md runs a
# non-performance policy today, and failing here would block all of them. What
# makes the data defensible is that the policy is IN the sidecar, so a balanced
# run is never silently compared with a pinned one.
$governor = 'unavailable'
try {
    $scheme = & powercfg /getactivescheme 2>$null
    if ($LASTEXITCODE -eq 0 -and $scheme) {
        if ($scheme -match '\(([^)]+)\)') { $governor = $Matches[1] -replace ' ', '_' }
    }
} catch { }
Emit 'governor' $governor
if ($governor -notmatch '[Pp]erformance' -and $governor -ne 'unavailable') {
    Warn "power scheme is '$governor', not a performance plan -- clocks are not pinned"
}

$turbo = 'unavailable'
try {
    $boost = & powercfg /q SCHEME_CURRENT SUB_PROCESSOR PERFBOOSTMODE 2>$null
    if ($LASTEXITCODE -eq 0 -and $boost) {
        $ac = ($boost | Select-String -Pattern 'Current AC Power Setting Index:\s*(0x[0-9a-fA-F]+)')
        if ($ac) { $turbo = if ([int]$ac.Matches[0].Groups[1].Value -eq 0) { 'disabled' } else { 'enabled' } }
    }
} catch { }
Emit 'turbo' $turbo

Emit 'power_mode' 'n/a'   # nvpmodel is a Jetson concept

# -- thermal gate -----------------------------------------------------------
Emit 'thermal_sensor'   $thermal.Sensor
Emit 'thermal_before_c' $thermal.Temp
Emit 'thermal_limit_c'  $thermal.Limit
if ($thermal.Temp -ne 'unavailable' -and $thermal.Limit -ne 'unavailable') {
    $headroom = [double]$thermal.Limit - [double]$thermal.Temp
    Emit 'thermal_headroom_c' ('{0:F1}' -f $headroom)
    if ($headroom -lt $margin) {
        GateFailed ("only {0:F1}C below the {1}C limit (need {2}C). Let the machine cool -- throttled data is indistinguishable from a slow configuration." -f $headroom, $thermal.Limit, $margin)
    }
} else {
    # Not a failure. See the note in the header.
    Emit 'thermal_headroom_c' 'unavailable'
}

Emit 'preflight_gates' $(if ($ReportOnly) { 'report-only' } else { 'enforced' })

if ($ReportOnly) {
    if ($script:failures -gt 0) { Warn "$($script:failures) gate(s) would have failed; -ReportOnly, continuing" }
    exit 0
}
if ($script:failures -gt 0) { exit 1 }
exit 0
