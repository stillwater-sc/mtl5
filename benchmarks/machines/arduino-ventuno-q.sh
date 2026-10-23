#!/usr/bin/env bash
# Cache-blocking A/B on the Arduino Ventuno Q (Qualcomm "Monaco", 4x Cortex-A78C
# + 4x Cortex-A55).
#
# This file IS the invocation, and like the Jetson profile it has to carry more
# than the pin list. Two things decide what the numbers mean here:
#
#   * WHICH CORES. The part is big.LITTLE: four Cortex-A78C (2.36 GHz max) and
#     four Cortex-A55 (1.96 GHz max), with different cache hierarchies. A run
#     that lands on an A55, or a team that mixes the two, measures the little
#     core -- the slowest tid sets the region's time -- and the `detected` arm
#     detects whichever hierarchy the thread happened to start on (#432). So the
#     pin list is the A78C cores ONLY, and it is READ from /proc/cpuinfo rather
#     than hardcoded: lscpu does not say which logical ids are which cluster,
#     and guessing "the big cores are 4-7" is the kind of confident wrong label
#     the sidecars exist to remove.
#
#   * THE GOVERNOR. There is no nvpmodel here; the clocks follow the cpufreq
#     governor of the A78C cluster's policy (the board ships `schedutil` on it,
#     `ondemand` on the other). Under a ramping governor the early reps of an
#     arm run slower than the late ones. So, as the Jetson's output directory
#     follows its power mode, this one FOLLOWS THE GOVERNOR: a schedutil run and
#     a performance run land in different directories and can never be silently
#     averaged together.
#
# Topology: 8 cores, no SMT, one cluster of each class. Record the cache sizes
# with `lscpu -C` alongside the first committed run; they are not assumed here.
#
# Usage:  benchmarks/machines/arduino-ventuno-q.sh       # -> data/arduino-ventuno-q-schedutil
#         # pin the clocks first for a comparable session (needs root):
#         echo performance | sudo tee /sys/devices/system/cpu/cpu<N>/cpufreq/scaling_governor
#         benchmarks/machines/arduino-ventuno-q.sh       # -> data/arduino-ventuno-q-performance
#         THREADS="1 2 4" ARMS="detected default" benchmarks/machines/arduino-ventuno-q.sh
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$ROOT"

# The device tree names the SoC, not the board ("Qualcomm Technologies, Inc.
# Monaco Monza addons"), so that is what is matched.
. "$SCRIPT_DIR/_identify.sh"
require_machine "Monaco" "Arduino Ventuno Q (Qualcomm Monaco)"

# The A78C cores, by MIDR part number (0xd4b; the A55 is 0xd05). /proc/cpuinfo
# on arm64 gives one block per logical cpu, "processor : N" then "CPU part : 0x...".
BIG_PART=0xd4b
DETECTED_PCPUS=$(awk -F: -v part="$BIG_PART" '
    /^processor/ { cpu = $2 + 0 }
    /^CPU part/  { gsub(/[[:space:]]/, "", $2); if ($2 == part) ids = ids (ids ? "," : "") cpu }
    END { print ids }' /proc/cpuinfo)
NBIG=$(printf '%s' "$DETECTED_PCPUS" | tr ',' '\n' | grep -c . || true)
if [ "$NBIG" -ne 4 ] && [ -z "${BENCH_PCPUS:-}" ]; then
    echo "Expected 4 Cortex-A78C cores (CPU part $BIG_PART) in /proc/cpuinfo, found $NBIG:" >&2
    echo "  '${DETECTED_PCPUS}'" >&2
    echo "Some may be offline. Set BENCH_PCPUS to the A78C logical ids by hand" >&2
    echo "(see 'grep -E \"processor|CPU part\" /proc/cpuinfo')." >&2
    exit 2
fi
DEFAULT_PCPUS=$DETECTED_PCPUS
export BENCH_PCPUS=${BENCH_PCPUS:-$DEFAULT_PCPUS}

# The governor and clock state of the cluster the run is pinned to, read from
# the first pinned core's cpufreq policy (world-readable; no root needed).
FIRST=$(printf '%s' "$BENCH_PCPUS" | cut -d, -f1)
CPUFREQ=/sys/devices/system/cpu/cpu$FIRST/cpufreq
GOV=unknown
[ -r "$CPUFREQ/scaling_governor" ] && GOV=$(cat "$CPUFREQ/scaling_governor")
CLOCKS=unknown
if [ -r "$CPUFREQ/scaling_min_freq" ] && [ -r "$CPUFREQ/scaling_max_freq" ]; then
    LO=$(( $(cat "$CPUFREQ/scaling_min_freq") / 1000 ))
    HI=$(( $(cat "$CPUFREQ/scaling_max_freq") / 1000 ))
    if [ "$GOV" = performance ] || [ "$LO" = "$HI" ]; then
        CLOCKS="pinned at $HI MHz"
    else
        CLOCKS="unpinned ($LO-$HI MHz)"
    fi
fi
if [ "${CLOCKS#unpinned}" != "$CLOCKS" ]; then
    echo "NOTE: cpu$FIRST's cluster runs the '$GOV' governor, clocks $CLOCKS." >&2
    echo "      For a session comparable with a pinned one, first run:" >&2
    echo "        echo performance | sudo tee $CPUFREQ/scaling_governor" >&2
    echo "      and keep the board cooled: DVFS ramping otherwise makes the first" >&2
    echo "      reps of every arm slower than the last." >&2
fi

export BENCH_PROFILE="arduino-ventuno-q-${GOV}"
export OUTDIR=${OUTDIR:-benchmarks/data/arduino-ventuno-q-${GOV}}
export BUILD_DIR=${BUILD_DIR:-build-release}
export THREADS=${THREADS:-"1 4"}                 # 1 and all four A78C cores
export ROUNDS=${ROUNDS:-5}
export REPS=${REPS:-5}
export DTYPE=${DTYPE:-double}
export ARMS=${ARMS:-"default detected kconly mconly"}

echo "profile: $BENCH_PROFILE"
echo "  machine:  $MTL5_MACHINE_ID"
echo "  governor: $GOV   (clocks: $CLOCKS)"
if [ "$BENCH_PCPUS" = "$DEFAULT_PCPUS" ]; then
    echo "  pinning:  $BENCH_PCPUS (Cortex-A78C only; the A55 cores are excluded)"
else
    echo "  pinning:  $BENCH_PCPUS (OVERRIDDEN; profile default is ${DEFAULT_PCPUS:-none detected})"
fi
echo "  arms:     $ARMS"
echo "  outdir:   $OUTDIR"
echo
echo "  thermal now: $(cat /sys/devices/virtual/thermal/thermal_zone*/temp 2>/dev/null | \
                       awk '{printf "%.1fC ", $1/1000}')"
echo

# -march=native, as on the other hosts: without it Highway compiles for the
# baseline and the blocking under test is for the wrong vector length. Both core
# classes report the same ISA flags, so the binary is the same whichever cluster
# the compiler runs on.
if [ ! -x "$BUILD_DIR/benchmarks/bench_blocking_ab_default" ]; then
    echo "== configuring and building $BUILD_DIR"
    cmake --preset release -DMTL5_WITH_HIGHWAY=ON -DMTL5_NATIVE_ARCH=ON >/dev/null
    targets=""
    for a in $ARMS; do targets="$targets bench_blocking_ab_$a"; done
    # shellcheck disable=SC2086
    cmake --build "$BUILD_DIR" --target $targets -j4 >/dev/null
fi

exec "$ROOT/benchmarks/run_blocking_ab.sh"
