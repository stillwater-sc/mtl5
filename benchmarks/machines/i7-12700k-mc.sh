#!/usr/bin/env bash
# #458 before/after on the i7-12700K (Alder Lake, hybrid): the threaded ic step
# chosen by the busiest thread's micro-panels (`default`) against the pre-#458
# block-count rule (`legacymc`). Same tree, same session, interleaved rounds.
#
# The control machine. Its shipped plans at T = 8 are already panel-optimal
# (mc = 64 against mr = 6 at kc = 256) and measured best of mc in [30, 72], so
# the change leaves them alone; it moves plans at T = 2..6, predicted x1.01-1.06.
# T = 1 is a null control: the serial path is untouched.
#
# Topology (lscpu -e=CPU,CORE): 8 P-cores with SMT on CPUs 0..15, one id per core
# is 0,2,...,14; the E-cores 16..19 are EXCLUDED. Mixing core classes measures
# the slower one, and an unpinned run lets the scheduler choose per round.
#
# Usage:  benchmarks/machines/i7-12700k-mc.sh
#         ROUNDS=6 benchmarks/machines/i7-12700k-mc.sh     # anything is overridable
#
# The arms are rebuilt EVERY run (incrementally). Skipping the build when a
# binary exists would compare a stale `default` -- one built before the change
# under test -- against a fresh `legacymc`, which is an A/B of two old trees.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$ROOT"

. "$SCRIPT_DIR/_identify.sh"
require_machine "i7-12700K" "i7-12700K"

export BENCH_PROFILE=i7-12700k-mc
# Its own directory: the runner deletes the arm CSVs it is about to write, and
# benchmarks/data/i7-12700k holds committed `default` results from #430.
export OUTDIR=${OUTDIR:-benchmarks/data/i7-12700k-mc458}
export BUILD_DIR=${BUILD_DIR:-build-release}
DEFAULT_PCPUS=0,2,4,6,8,10,12,14
export BENCH_PCPUS=${BENCH_PCPUS:-$DEFAULT_PCPUS}
export THREADS=${THREADS:-"1 2 4 6 8"}
export ROUNDS=${ROUNDS:-4}
export REPS=${REPS:-5}
export DTYPE=${DTYPE:-double}
export ARMS=${ARMS:-"default legacymc"}
# #458's shapes: the issue's 2048^3, the squares either side, two that are not a
# multiple of the block size, and a wide one where n > nc puts the jc loop in play.
export SHAPES=${SHAPES:-"1000,1000,1000;1024,1024,1024;2048,2048,2048;3000,3000,1000;4096,4096,1024;512,8192,1024"}

echo "profile: $BENCH_PROFILE"
echo "  cpu:     $MTL5_MACHINE_ID"
echo "  build:   cmake --preset release -DMTL5_WITH_HIGHWAY=ON -DMTL5_NATIVE_ARCH=ON"
if [ "$BENCH_PCPUS" = "$DEFAULT_PCPUS" ]; then
    echo "  pinning: $BENCH_PCPUS (P-cores only; E-cores 16-19 excluded)"
else
    echo "  pinning: $BENCH_PCPUS (OVERRIDDEN; profile default is $DEFAULT_PCPUS)"
fi
echo "  arms:    $ARMS"
echo "  threads: $THREADS"
echo "  outdir:  $OUTDIR"
echo

echo "== configuring and building $BUILD_DIR"
cmake --preset release -DMTL5_WITH_HIGHWAY=ON -DMTL5_NATIVE_ARCH=ON >/dev/null
targets=""
for a in $ARMS; do targets="$targets bench_blocking_ab_$a"; done
# shellcheck disable=SC2086
cmake --build "$BUILD_DIR" --target $targets -j4 >/dev/null

exec "$ROOT/benchmarks/run_blocking_ab.sh"
