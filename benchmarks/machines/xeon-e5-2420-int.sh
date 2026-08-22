#!/usr/bin/env bash
# Integer arms on the Xeon E5-2420 v2 (Ivy Bridge EP, 6x, SSE4) -- #451.
#
# THIS MACHINE IS THE CONTROL, and it is the only one of the four that is a
# control for the GEMM question rather than for the dot question.
#
# It has no VNNI, no AVX2, no dot-product instruction of any kind: BOTH int8
# arms are decomposed here. That is exactly what makes it useful. The quad
# micro-kernel (#451 phase 5) changes two things at once relative to
# widen-on-load -- the four-products-per-lane KERNEL SHAPE, and, on hardware that
# has it, the INSTRUCTION. A machine where the instruction is absent measures the
# shape alone:
#
#     gemm_i8_i32_quad / gemm_i8_i32   ~1.25x, from the shape only
#
# and that number is the baseline any VNNI part has to beat before its silicon
# can be credited with anything. Committed run: data/xeon-e5-2420/int_arms.csv,
# 1.19x at n=128 rising to 1.27x at n=1024 -- it grows with n, because the small
# sizes never leave cache and the traffic reduction has nothing to pay for yet.
#
# WHY --allow-decomposed IS BAKED IN HERE. run_int_bench.sh refuses to time a
# decomposed build, because nothing in a timing distinguishes `vpdpbusd` from a
# `vpmaddwd` decomposition and a mistake is otherwise invisible. On this part the
# decomposition is not a mistake, it is the measurement: there is no VNNI silicon
# to reach and no compiler flag that would change it. So the profile asserts that
# on the operator's behalf, and the CSV still records `label=native-int-decomposed`
# so it can never be confused with a native run.
#
# TOPOLOGY, and it differs from every other profile here. Six physical cores with
# SMT, and the siblings are (0,6) (1,7) ... (5,11) -- BLOCKED, not interleaved.
# The i7 and Ryzen profiles pin 0,2,4,6,... because their siblings are adjacent
# pairs; doing that here would put two threads on three physical cores and leave
# three idle. Verified on the machine:
#     /sys/devices/system/cpu/cpu0/topology/thread_siblings_list -> 0,6
#
# Usage:  bash benchmarks/machines/xeon-e5-2420-int.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Refuse to write this machine's numbers on a different machine: the CSVs are
# named by suite, not by host, so a profile run on the wrong box replaces
# committed evidence with data from somewhere else (#439).
EXPECT="E5-2420"
ACTUAL="$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2- | sed 's/^ *//' || echo unknown)"
if [[ "$ACTUAL" != *"$EXPECT"* ]]; then
    echo "This profile is for a Xeon $EXPECT; this machine reports:"
    echo "  $ACTUAL"
    if [ "${FORCE:-0}" != "1" ]; then
        echo "Refusing to write $EXPECT data. Set FORCE=1 if this really is one."
        exit 2
    fi
fi

# One logical id per physical core -- see the topology note above.
PIN="${PIN:-0,1,2,3,4,5}"

echo "profile: xeon-e5-2420-int"
echo "  cpu:     $ACTUAL"
echo "  arch:    -march=native   (SSE4; no VNNI silicon exists on this part)"
echo "  pinning: $PIN            (siblings are 0/6..5/11, so ids 0-5 are distinct cores)"
echo "  expect:  int8 quad dot: decomposed  -- both int8 arms, which is the point"
echo ""

exec "$REPO_ROOT/benchmarks/run_int_bench.sh" \
    --arch "-march=native" \
    --outdir "benchmarks/data/xeon-e5-2420" \
    --pin "$PIN" \
    --allow-decomposed \
    "$@"
