#!/usr/bin/env bash
# Preflight: gate a measurement on machine state, and record that state (#442).
#
# Every guard in this repo's benchmark harnesses was added AFTER the matching
# error had already produced a wrong result -- a Zen 4 run destroyed the i7's
# committed CSVs, the analyzer called a 4.8% "win" between byte-identical arms,
# the Jetson asked for 8 threads on a 6-core part. This script is the first one
# written before the error rather than after it, and it exists because the
# remaining gaps all have the same shape: the number lands in the CSV and the
# reason it is wrong lands nowhere.
#
# Two jobs, deliberately in one place:
#
#   1. FAIL on conditions that make a measurement meaningless. Not warn -- a
#      warning scrolls past and the CSV gets committed anyway.
#   2. EMIT key=value lines for the sidecar, so the machine state is IN the data
#      rather than in someone's memory of the session. The Jetson numbers on
#      docs/benchmarks/systems.md are annotated "15 W, schedutil, clocks not
#      pinned" only because someone thought to ask.
#
# Usage:
#   preflight.sh [--phase before|after] [--threads N] [--repo DIR] [--report-only]
#
#   --phase before   (default) run every gate, emit the full record
#   --phase after    emit thermal_after_c only -- call once the run finishes, so
#                    a throttled session is visible in the data. A run that ends
#                    hot and one that ran slow look identical in GFLOP/s alone.
#   --threads N      the largest thread count the run will ask for
#   --repo DIR       repository to check (default: the parent of this script)
#   --report-only    probe and emit, never fail. For CI and for inspecting a
#                    machine; NOT for taking measurements.
#
# Environment:
#   ALLOW_DIRTY=1              permit a dirty working tree (still recorded)
#   MIN_THERMAL_HEADROOM_C=15  required margin below the throttle limit
#
# Exit: 0 all gates pass, 1 a gate failed, 2 bad usage.
#
# Thermal policy: this FAILS only where a sensor and its limit are both
# readable. On a machine that exposes neither (typically Windows, and macOS
# without extra tooling) it records `thermal_before_c=unavailable` and proceeds
# -- refusing to run there would make those machines unbenchmarkable, and
# silently passing would hide the gap. Unavailable is recorded, never assumed
# cool.
set -uo pipefail

PHASE=before
THREADS_REQ=0
REPORT_ONLY=0
REPO=""
MARGIN=${MIN_THERMAL_HEADROOM_C:-15}

# A gate that cannot be reached is worse than no gate: a trailing `--threads`
# with no value used to leave $1 unchanged (shift 2 fails on one argument), so
# the loop spun forever, and a non-numeric value silently skipped the thread
# budget check entirely. Demand the value up front.
need_value() {            # flag, remaining-count
    if [ "$2" -lt 2 ]; then
        echo "preflight: $1 requires a value" >&2
        exit 2
    fi
}
while [ $# -gt 0 ]; do
    case "$1" in
        --phase)       need_value "$1" $#; PHASE=$2; shift 2 ;;
        --threads)     need_value "$1" $#; THREADS_REQ=$2; shift 2 ;;
        --repo)        need_value "$1" $#; REPO=$2; shift 2 ;;
        --report-only) REPORT_ONLY=1; shift ;;
        -h|--help)     sed -n '2,45p' "$0"; exit 0 ;;
        *) echo "preflight: unknown argument '$1'" >&2; exit 2 ;;
    esac
done
case "$PHASE" in
    before|after) ;;
    *) echo "preflight: --phase must be 'before' or 'after'" >&2; exit 2 ;;
esac
case "$THREADS_REQ" in
    ''|*[!0-9]*)
        echo "preflight: --threads must be a non-negative integer, got '$THREADS_REQ'" >&2
        exit 2 ;;
esac
if [ -z "$REPO" ]; then
    REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
fi

FAILURES=0
emit() { printf '%s=%s\n' "$1" "$2"; }
gate_failed() {           # message...
    echo "preflight: FAIL: $*" >&2
    FAILURES=$((FAILURES + 1))
}
warn() { echo "preflight: warn: $*" >&2; }

# ── thermal ─────────────────────────────────────────────────────────────────
# Sets THERMAL_C, THERMAL_LIMIT_C, THERMAL_SENSOR; each may be "unavailable".
#
# Two sources, because the machines in docs/benchmarks/systems.md split between
# them: x86 parts expose hwmon (coretemp/k10temp) with the throttle point in
# temp*_max, while the Jetson exposes thermal zones named CPU-therm/soc*-therm.
# Prefer hwmon: it names the package and carries a limit, where a zone's trip
# points are often unpopulated (this Xeon reports -274000, i.e. absolute zero, on
# both of its trip points).
milli_to_c() { awk -v m="$1" 'BEGIN { printf "%.1f", m / 1000.0 }'; }

read_thermal() {
    THERMAL_C=unavailable
    THERMAL_LIMIT_C=unavailable
    THERMAL_SENSOR=unavailable

    for h in /sys/class/hwmon/hwmon*; do
        [ -r "$h/name" ] || continue
        local name; name=$(cat "$h/name" 2>/dev/null) || continue
        case "$name" in
            coretemp|k10temp|zenpower|cpu_thermal|soc_thermal) ;;
            *) continue ;;
        esac
        [ -r "$h/temp1_input" ] || continue
        local label=$name
        [ -r "$h/temp1_label" ] && label="$name:$(cat "$h/temp1_label")"
        THERMAL_C=$(milli_to_c "$(cat "$h/temp1_input")")
        THERMAL_SENSOR=$label
        # temp*_max is the point the part throttles at; temp*_crit is where it
        # shuts down. Gate on the first if it is present -- data taken while
        # throttling is already invalid, long before anything is at risk.
        for lim in "$h/temp1_max" "$h/temp1_crit"; do
            if [ -r "$lim" ]; then
                local v; v=$(cat "$lim")
                if [ "$v" -gt 0 ] 2>/dev/null; then
                    THERMAL_LIMIT_C=$(milli_to_c "$v")
                    break
                fi
            fi
        done
        return
    done

    for z in /sys/class/thermal/thermal_zone*; do
        [ -r "$z/type" ] && [ -r "$z/temp" ] || continue
        local type; type=$(cat "$z/type" 2>/dev/null) || continue
        case "$type" in
            *cpu*|*CPU*|x86_pkg_temp|*soc*|*SOC*) ;;
            *) continue ;;
        esac
        THERMAL_C=$(milli_to_c "$(cat "$z/temp")")
        THERMAL_SENSOR="thermal_zone:$type"
        local best=""
        for t in "$z"/trip_point_*_temp; do
            [ -r "$t" ] || continue
            local v; v=$(cat "$t")
            # Unpopulated trip points read as -274000. Ignore them rather than
            # computing headroom against a temperature below absolute zero.
            [ "$v" -gt 0 ] 2>/dev/null || continue
            if [ -z "$best" ] || [ "$v" -lt "$best" ]; then best=$v; fi
        done
        [ -n "$best" ] && THERMAL_LIMIT_C=$(milli_to_c "$best")
        return
    done
}

read_thermal

if [ "$PHASE" = after ]; then
    emit thermal_after_c "$THERMAL_C"
    exit 0
fi

emit preflight_version 1
emit preflight_host "$(uname -n)"
emit preflight_kernel "$(uname -sr)"

# ── working tree ────────────────────────────────────────────────────────────
# The BINARY records the commit it was built from (mtl/build_info.hpp). This
# records the commit the tree is on NOW. They are different facts: if they
# disagree, the binary is stale relative to the checkout, and the run measures
# code that is no longer what the tree says it is.
TREE_COMMIT=unknown
TREE_DIRTY=unknown
if command -v git >/dev/null 2>&1 && git -C "$REPO" rev-parse HEAD >/dev/null 2>&1; then
    TREE_COMMIT=$(git -C "$REPO" rev-parse --short=12 HEAD)
    if [ -z "$(git -C "$REPO" status --porcelain --untracked-files=normal)" ]; then
        TREE_DIRTY=0
    else
        TREE_DIRTY=1
        if [ "${ALLOW_DIRTY:-0}" = 1 ]; then
            warn "working tree is dirty; ALLOW_DIRTY=1, recording tree_git_dirty=1"
        else
            gate_failed "working tree is dirty -- this result could not be reproduced." \
                        "Commit, stash, or set ALLOW_DIRTY=1 to record it as dirty."
        fi
    fi
fi
emit tree_git_commit "$TREE_COMMIT"
emit tree_git_dirty  "$TREE_DIRTY"

# ── competing load ──────────────────────────────────────────────────────────
# "Check pgrep -a make before building" has lived in CLAUDE.md and in habit. A
# competing compile does not make the run noisy, it makes it wrong: the arms are
# interleaved, so a build that finishes mid-session penalises whichever arm was
# running at the time and the difference is reported as a result.
BUSY=""
if command -v pgrep >/dev/null 2>&1; then
    # Record NAMES, not bare pids: a pid in a sidecar read weeks later says
    # nothing, while "make(12345)" says the session raced a build.
    while read -r pid name _rest; do
        [ -n "${pid:-}" ] && BUSY="$BUSY $name($pid)"
    done < <(pgrep -xl "make|ninja|cmake|cc1|cc1plus|lto1|ld|conftest" 2>/dev/null)
    # Match the process NAME, never the command line: -f matched the shell that
    # had merely typed the benchmark's path, so preflight reported the session
    # it was running inside as competing load and failed every clean run.
    while read -r pid name _rest; do
        [ -n "${pid:-}" ] && [ "$pid" != "$$" ] && BUSY="$BUSY $name($pid)"
    done < <(pgrep -xl "bench_[a-z0-9_]+" 2>/dev/null)
fi
BUSY=$(echo "$BUSY" | tr -s ' ' | sed 's/^ *//;s/ *$//')
if [ -n "$BUSY" ]; then
    emit competing_load "$(echo "$BUSY" | tr ' ' ',')"
    gate_failed "a build or benchmark is already running: $BUSY"
else
    emit competing_load none
fi

LOAD1=unavailable
[ -r /proc/loadavg ] && LOAD1=$(cut -d' ' -f1 /proc/loadavg)
emit loadavg_1m "$LOAD1"

# ── cores and thread budget ─────────────────────────────────────────────────
# nproc honours the affinity mask, so under taskset this is the count the run can
# actually use -- which is the number the thread budget must fit into.
#
# macOS ships no nproc (it is GNU coreutils), which made this 0 there and failed
# EVERY run with "0 cpus available" -- a gate that blocks the machine outright is
# not a stricter gate, it is a broken one. sysctl is the platform's own answer.
NPROC_AFFINITY=$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 0)
NPROC_ONLINE=$(nproc --all 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo "$NPROC_AFFINITY")
emit cpu_online "$NPROC_ONLINE"
emit cpu_affinity "$NPROC_AFFINITY"
if [ "$THREADS_REQ" -gt 0 ] 2>/dev/null && [ "$THREADS_REQ" -gt "$NPROC_AFFINITY" ]; then
    gate_failed "$THREADS_REQ threads requested but only $NPROC_AFFINITY cpu(s) available."
fi

# ── frequency policy ────────────────────────────────────────────────────────
# Recorded, and warned about, not failed: every machine in systems.md today runs
# a non-performance governor (powersave on the i7, ondemand on the Xeon,
# schedutil on the Jetson), and failing here would block all of them. What makes
# the data defensible is that the governor is IN the sidecar, so a run taken
# under schedutil is never silently compared with one taken pinned.
GOVERNORS=$(cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null | sort -u | tr '\n' ',' | sed 's/,$//')
[ -z "$GOVERNORS" ] && GOVERNORS=unavailable
emit governor "$GOVERNORS"
case "$GOVERNORS" in
    performance) ;;
    unavailable) ;;
    *) warn "governor is '$GOVERNORS', not 'performance' -- clocks are not pinned" ;;
esac

TURBO=unavailable
if [ -r /sys/devices/system/cpu/intel_pstate/no_turbo ]; then
    if [ "$(cat /sys/devices/system/cpu/intel_pstate/no_turbo)" = 0 ]; then TURBO=enabled; else TURBO=disabled; fi
elif [ -r /sys/devices/system/cpu/cpufreq/boost ]; then
    if [ "$(cat /sys/devices/system/cpu/cpufreq/boost)" = 1 ]; then TURBO=enabled; else TURBO=disabled; fi
fi
emit turbo "$TURBO"

# Jetson: the power mode caps clocks AND decides how many cores are online, and
# it differs per module and per JetPack image. A Jetson CSV without it cannot be
# compared with another Jetson CSV.
POWER_MODE=n/a
if command -v nvpmodel >/dev/null 2>&1; then
    POWER_MODE=$(nvpmodel -q 2>/dev/null | tr '\n' ' ' | sed 's/  */ /g;s/ *$//')
    [ -z "$POWER_MODE" ] && POWER_MODE=unknown
fi
emit power_mode "$POWER_MODE"

# ── thermal gate ────────────────────────────────────────────────────────────
emit thermal_sensor   "$THERMAL_SENSOR"
emit thermal_before_c "$THERMAL_C"
emit thermal_limit_c  "$THERMAL_LIMIT_C"
if [ "$THERMAL_C" != unavailable ] && [ "$THERMAL_LIMIT_C" != unavailable ]; then
    HEADROOM=$(awk -v t="$THERMAL_C" -v l="$THERMAL_LIMIT_C" 'BEGIN { printf "%.1f", l - t }')
    emit thermal_headroom_c "$HEADROOM"
    if awk -v h="$HEADROOM" -v m="$MARGIN" 'BEGIN { exit !(h < m) }'; then
        gate_failed "only ${HEADROOM}C below the ${THERMAL_LIMIT_C}C limit (need ${MARGIN}C)." \
                    "Let the machine cool -- throttled data is indistinguishable from a slow configuration."
    fi
else
    # Not a failure. See the thermal policy note at the top.
    emit thermal_headroom_c unavailable
fi

emit preflight_gates "$([ "$REPORT_ONLY" = 1 ] && echo report-only || echo enforced)"

if [ "$REPORT_ONLY" = 1 ]; then
    [ "$FAILURES" -gt 0 ] && warn "$FAILURES gate(s) would have failed; --report-only, continuing"
    exit 0
fi
[ "$FAILURES" -gt 0 ] && exit 1
exit 0
