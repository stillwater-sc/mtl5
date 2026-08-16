#!/usr/bin/env bash
# Cache-blocking A/B on the Jetson Orin Nano (Tegra234, 6x Cortex-A78AE).
#
# This file IS the invocation, and on a Jetson it has to carry more than the pin
# list, because the numbers are only meaningful relative to a POWER MODE. The
# mode selects clock caps (and on other modules, which cores are online), it
# differs per module and per JetPack image, and an unpinned, unconfigured run
# measures the cooling solution rather than the code.
#
# So the output directory FOLLOWS the mode: a 15 W run and a MAXN run land in
# different directories and can never be silently averaged together. The
# committed 15 W data was taken under `schedutil` with clocks unpinned, which is
# why it is labelled that way in docs/benchmarks/systems.md.
#
# Topology: six A78AE cores, no SMT, all online -- six is the module's PHYSICAL
# core count, not a power-mode reduction (nvpmodel.conf defines CPU_A78_0..5 and
# nothing else). Two asymmetric clusters (4-core + 2-core), 64 KB L1d and 256 KB
# L2 per core, 2 MB L3 per cluster, 4 MB system cache shared.
#
# Usage:  benchmarks/machines/jetson-orin-nano.sh
#         sudo nvpmodel -m 0 && sudo jetson_clocks   # MAXN, DVFS pinned
#         benchmarks/machines/jetson-orin-nano.sh    # -> data/jetson-orin-nano-MAXN
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$ROOT"

# Refuse to write Jetson numbers on something that is not one (see the i7
# profile for why this guard exists).
if [ -r /proc/device-tree/model ]; then
    MODEL=$(tr -d '\0' < /proc/device-tree/model)
else
    MODEL=unknown
fi
case "$MODEL" in
    *Orin*) ;;
    *) echo "This profile is for a Jetson Orin; this machine reports: $MODEL" >&2
       echo "Refusing to write Orin data. Set FORCE=1 if this really is one." >&2
       [ "${FORCE:-0}" = 1 ] || exit 2 ;;
esac

# The power mode decides the clocks, so it decides what the numbers mean. Read it
# rather than asking the operator to remember: `nvpmodel -q` prints
#   NV Power Mode: 15W
#   0
MODE="unknown"
if command -v nvpmodel >/dev/null 2>&1; then
    MODE=$(nvpmodel -q 2>/dev/null | sed -n 's/^NV Power Mode:[[:space:]]*//p' | head -1)
    [ -n "$MODE" ] || MODE="unknown"
fi
if [ "$MODE" = unknown ]; then
    echo "WARNING: could not read the power mode (nvpmodel missing, or it needs root)." >&2
    echo "         The mode decides the clock caps, so a run without it cannot be" >&2
    echo "         compared with one that has it. Set JETSON_MODE=<name> to record" >&2
    echo "         it by hand, or run: sudo nvpmodel -q" >&2
    MODE=${JETSON_MODE:-unknown}
fi
MODE=$(printf '%s' "$MODE" | tr -d ' /')

# Clocks: `jetson_clocks` pins DVFS. Without it the early reps run at a lower
# clock than the late ones, which reads as a warm-up effect and is not one.
#
# Asked of sysfs rather than of jetson_clocks --show, which needs root on some
# images: scaling_min_freq == scaling_max_freq is exactly what pinning produces,
# and it is world-readable.
CLOCKS=unpinned
CPUFREQ=/sys/devices/system/cpu/cpu0/cpufreq
if [ -r "$CPUFREQ/scaling_min_freq" ] && [ -r "$CPUFREQ/scaling_max_freq" ]; then
    if [ "$(cat "$CPUFREQ/scaling_min_freq")" = "$(cat "$CPUFREQ/scaling_max_freq")" ]; then
        CLOCKS="pinned at $(( $(cat "$CPUFREQ/scaling_max_freq") / 1000 )) MHz"
    else
        CLOCKS="unpinned ($(( $(cat "$CPUFREQ/scaling_min_freq") / 1000 ))-$(( $(cat "$CPUFREQ/scaling_max_freq") / 1000 )) MHz)"
    fi
fi
if [ "${CLOCKS#unpinned}" != "$CLOCKS" ]; then
    echo "NOTE: CPU clocks are $CLOCKS. Run 'sudo jetson_clocks' first if you want" >&2
    echo "      this session comparable with a pinned one -- DVFS ramping otherwise" >&2
    echo "      makes the first reps of every arm slower than the last." >&2
fi

export BENCH_PROFILE="jetson-orin-nano-${MODE}"
export OUTDIR=${OUTDIR:-benchmarks/data/jetson-orin-nano-${MODE}}
export BUILD_DIR=${BUILD_DIR:-build-release}
export BENCH_PCPUS=${BENCH_PCPUS:-0,1,2,3,4,5}   # six physical cores, no SMT
export THREADS=${THREADS:-"1 6"}                 # 6, NOT the 8 that was asked once
export ROUNDS=${ROUNDS:-5}
export REPS=${REPS:-5}
export DTYPE=${DTYPE:-double}
export ARMS=${ARMS:-"default detected kconly mconly"}

echo "profile: $BENCH_PROFILE"
echo "  model:      $MODEL"
echo "  power mode: $MODE   (clocks: $CLOCKS)"
echo "  pinning:    $BENCH_PCPUS"
echo "  arms:       $ARMS"
echo "  outdir:     $OUTDIR"
echo
echo "  thermal now: $(cat /sys/devices/virtual/thermal/thermal_zone*/temp 2>/dev/null | \
                       awk '{printf "%.1fC ", $1/1000}')"
echo

if [ ! -x "$BUILD_DIR/benchmarks/bench_blocking_ab_default" ]; then
    echo "== configuring and building $BUILD_DIR"
    cmake --preset release -DMTL5_WITH_HIGHWAY=ON -DMTL5_NATIVE_ARCH=ON >/dev/null
    targets=""
    for a in $ARMS; do targets="$targets bench_blocking_ab_$a"; done
    # shellcheck disable=SC2086
    cmake --build "$BUILD_DIR" --target $targets -j4 >/dev/null
fi

exec "$ROOT/benchmarks/run_blocking_ab.sh"
