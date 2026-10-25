#!/usr/bin/env bash
# Dispatch and join latency of thread_pool and spin_pool (#636).
#
# Builds bench_pool_latency once, then runs every arm (thread, spin, sleep) in
# a process of its own, at each gap, for ROUNDS rounds, so the CSV shows the
# spread between processes as well as the percentiles within one.
#
# Usage:
#   OUTDIR=benchmarks/data/<machine>-pool-latency BENCH_PCPUS=0,2,4,6,8,10,12,14 \
#       benchmarks/run_pool_latency.sh
# Environment:
#   OUTDIR       required; one directory per machine
#   BENCH_PCPUS  cores to pin to (one logical cpu per physical core); the
#                largest of THREADS must not exceed their count
#   THREADS      default 2,4,8
#   GAPS         idle microseconds between regions, default "0 1000": back to
#                back, and a 1 kHz loop (past the spin, so workers sleep)
#   REGIONS      default 20000 at gap 0, 2000 at a nonzero gap
#   ROUNDS       default 3
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

if [ -z "${OUTDIR:-}" ]; then
    echo "OUTDIR is required: give this machine its own directory, e.g." >&2
    echo "  OUTDIR=benchmarks/data/<machine>-pool-latency $0" >&2
    exit 2
fi

THREADS=${THREADS:-2,4,8}
GAPS=${GAPS:-"0 1000"}
ROUNDS=${ROUNDS:-3}
MAXT=$(printf '%s' "$THREADS" | tr ',' '\n' | sort -n | tail -1)

PIN=()
if [ -n "${BENCH_PCPUS:-}" ]; then
    PIN=(taskset -c "$BENCH_PCPUS")
    NPIN=$(printf '%s' "$BENCH_PCPUS" | tr ',' '\n' | grep -c .)
    if [ "$MAXT" -gt "$NPIN" ]; then
        echo "THREADS goes to $MAXT but BENCH_PCPUS has $NPIN cpus" >&2
        exit 2
    fi
else
    echo "WARNING: BENCH_PCPUS not set -- no pinning; on a hybrid CPU the" >&2
    echo "         workers land on E-cores too." >&2
fi

PREFLIGHT=$("$SCRIPT_DIR/preflight.sh" --threads "$MAXT" --repo "$ROOT" \
                --ignore-path "$OUTDIR") || {
    echo "preflight failed -- not measuring (ALLOW_DIRTY=1 accepts a dirty tree)" >&2
    exit 1
}

dir=build-pool-latency
echo "== building"
mkdir -p "$dir"
cmake -B "$dir" -DCMAKE_BUILD_TYPE=Release -DMTL5_BUILD_BENCHMARKS=ON \
      -DMTL5_BUILD_TESTS=OFF -DMTL5_BUILD_EXAMPLES=OFF >"$dir/pool-latency.cmake.log" 2>&1
cmake --build "$dir" --target bench_pool_latency -j4 >/dev/null
# Only now: the build records whether the tree was dirty, and an output
# directory made before it would count as untracked.
mkdir -p "$OUTDIR"

csv=$OUTDIR/pool_latency.csv
rm -f "$csv" "$csv.sysinfo"
for round in $(seq 1 "$ROUNDS"); do
    for gap in $GAPS; do
        regions=${REGIONS:-$([ "$gap" = 0 ] && echo 20000 || echo 2000)}
        for arm in thread spin sleep; do
            echo "== round $round, gap ${gap}us, $arm"
            ${PIN[@]+"${PIN[@]}"} "$dir/benchmarks/bench_pool_latency" --arm "$arm" \
                --threads "$THREADS" --gap-us "$gap" --regions "$regions" \
                --round "$round" --csv "$csv"
        done
    done
done
{ printf '%s\n' "$PREFLIGHT"
  "$SCRIPT_DIR/preflight.sh" --phase after --repo "$ROOT" --report-only; } >>"$csv.sysinfo"
