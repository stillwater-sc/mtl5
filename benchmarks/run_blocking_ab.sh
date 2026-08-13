#!/usr/bin/env bash
# Cache-blocking A/B (#426, #432): does GEMM go faster with kc/mc derived from
# the DETECTED cache hierarchy than with the hardcoded Haswell-class defaults?
#
# Both arms are the same source. `bench_blocking_ab_detected` is built with
# MTL5_ENABLE_CACHE_DETECTION; `bench_blocking_ab_default` is what MTL5 ships.
# Nothing else differs, so a difference in throughput is a difference in kc/mc.
#
# Detection is opt-in precisely because this harness found it losing by up to 45%
# on an i7-12700K (see simd/blocking.hpp). Re-run this on new hardware before
# concluding anything for that machine.
#
# Protocol, matching the discipline the rest of docs/benchmarks/ uses:
#   * PINNED    -- one logical id per physical core. On a hybrid CPU this also
#                  decides which cache hierarchy is detected at all (#432), so an
#                  unpinned run here is not merely noisy, it is ambiguous.
#   * INTERLEAVED -- arms alternate within one session, and the ORDER within a
#                  round alternates too, so warm-up and thermal drift do not
#                  accrue to one arm. A ratio is only defensible when both sides
#                  come from the same session.
#   * MIN of N  -- the binary reports the minimum of --reps runs per point.
#   * SHAPES DERIVED ONCE -- the shape list comes from the detected arm and is
#                  passed verbatim to both, so the arms are never compared on
#                  different shapes (their own mc/nc differ, so each would
#                  otherwise pick its own).
#
# Environment:
#   BUILD_DIR    build tree containing the benchmarks (default: build-release)
#   BENCH_PCPUS  comma list of logical ids, one per physical core, in the order
#                to use. MUST match your topology -- see `lscpu -e=CPU,CORE`.
#                  i7-12700K P-cores : 0,2,4,6,8,10,12,14   (E-cores excluded)
#                  Xeon E5-2420 v2   : 0,1,2,3,4,5
#                Default: 0,2,4,6,8,10,12,14
#   THREADS      thread counts to sweep (default "1 8")
#   REPS         repetitions per point (default 5)
#   ROUNDS       interleaved A/B rounds (default 3)
#   DTYPE        double | float (default double)
#   SHAPES       override the derived shape list, "m,n,k;m,n,k". Use for a quick
#                smoke run; prefer the derived list for real measurements, since
#                it is what puts the jc loop in play on THIS machine.
#   OUTDIR       where CSVs land (default benchmarks/data)
set -euo pipefail

BUILD_DIR=${BUILD_DIR:-build-release}
BENCH_PCPUS=${BENCH_PCPUS:-0,2,4,6,8,10,12,14}
THREADS=${THREADS:-"1 8"}
REPS=${REPS:-5}
ROUNDS=${ROUNDS:-3}
DTYPE=${DTYPE:-double}
OUTDIR=${OUTDIR:-benchmarks/data}

DET="$BUILD_DIR/benchmarks/bench_blocking_ab_detected"
DEF="$BUILD_DIR/benchmarks/bench_blocking_ab_default"
for b in "$DET" "$DEF"; do
    if [ ! -x "$b" ]; then
        echo "missing $b" >&2
        echo "build with:  cmake --preset release && cmake --build $BUILD_DIR --target bench_blocking_ab_detected bench_blocking_ab_default -j4" >&2
        exit 1
    fi
done
mkdir -p "$OUTDIR"

# Pin to the first T ids of BENCH_PCPUS. On a hybrid part this is also what fixes
# WHICH cache hierarchy gets detected, so it is applied to every run including
# the single-threaded ones.
pin_for() {
    local t=$1
    echo "$BENCH_PCPUS" | cut -d, -f1-"$t"
}

# Every requested thread count must be positive and must fit BENCH_PCPUS. `cut`
# silently returns the whole (shorter) list when asked for more fields than it
# has, so an oversized THREADS would run more workers than pinned cores while the
# CSV still recorded the larger count -- a wrong number that looks like a result.
NPCPUS=$(echo "$BENCH_PCPUS" | tr ',' '\n' | grep -c .)
TMAX=0
for t in $THREADS; do
    if ! [ "$t" -ge 1 ] 2>/dev/null; then
        echo "THREADS: '$t' is not a positive integer" >&2; exit 2
    fi
    if [ "$t" -gt "$NPCPUS" ]; then
        echo "THREADS=$t exceeds the $NPCPUS cpu id(s) in BENCH_PCPUS ($BENCH_PCPUS)." >&2
        echo "Set BENCH_PCPUS to one logical id per physical core on this machine." >&2
        exit 2
    fi
    [ "$t" -gt "$TMAX" ] && TMAX=$t
done

# Shapes: derived ONCE, from the detected arm, at the largest thread count (the
# wide/short shapes depend on the thread budget). Both arms then get this list.
SHAPES=${SHAPES:-$(taskset -c "$(pin_for "$TMAX")" "$DET" --suggest-shapes --threads "$TMAX" --dtype "$DTYPE")}
echo "shapes: $SHAPES"

CSV_DET="$OUTDIR/blocking_ab_detected.csv"
CSV_DEF="$OUTDIR/blocking_ab_default.csv"
rm -f "$CSV_DET" "$CSV_DEF" "$CSV_DET.sysinfo" "$CSV_DEF.sysinfo"

run_arm() {   # bin label csv cpus threads
    MTL5_NUM_THREADS=$5 taskset -c "$4" "$1" \
        --label "$2" --dtype "$DTYPE" --threads "$5" --reps "$REPS" \
        --shapes "$SHAPES" --csv "$3"
}

for round in $(seq 1 "$ROUNDS"); do
    for t in $THREADS; do
        cpus=$(pin_for "$t")
        echo "== round $round, T=$t, cpus=$cpus"
        # Interleaved AND order-alternated. Running one arm first every round
        # would fold warm-up, frequency ramp and thermal drift into the ratio in
        # a fixed direction; alternating cancels it to first order.
        if [ $((round % 2)) -eq 1 ]; then
            run_arm "$DET" detected "$CSV_DET" "$cpus" "$t"
            run_arm "$DEF" default  "$CSV_DEF" "$cpus" "$t"
        else
            run_arm "$DEF" default  "$CSV_DEF" "$cpus" "$t"
            run_arm "$DET" detected "$CSV_DET" "$cpus" "$t"
        fi
    done
done

echo
echo "wrote $CSV_DET and $CSV_DEF (+ .sysinfo sidecars)"
echo "compare with: benchmarks/analyze_blocking_ab.py $CSV_DET $CSV_DEF"
