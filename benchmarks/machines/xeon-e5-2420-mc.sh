#!/usr/bin/env bash
# #458 before/after on the Xeon E5-2420 v2 (Ivy Bridge-EP): the threaded ic step
# chosen by the busiest thread's micro-panels (`default`) against the pre-#458
# block-count rule (`legacymc`). Same tree, same session, interleaved rounds.
#
# This is the machine #458's W curve was measured on, and the one where the
# change is predicted to matter most: its shipped threaded mc = 32 against
# mr = 6 puts a ragged panel in every ic block. Predicted from the panel count,
# x1.04-1.16 at T = 2..6 on the shapes below; T = 1 is a null control, since the
# serial path is untouched and both arms must agree there.
#
# Topology: six physical cores, SMT siblings 0/6 .. 5/11, so one id per core is
# 0..5. Hyperthreads are excluded -- two threads on one core would measure the
# sharing, not the partition.
#
# Usage:  benchmarks/machines/xeon-e5-2420-mc.sh
#         ROUNDS=6 benchmarks/machines/xeon-e5-2420-mc.sh     # anything is overridable
#
# The arms are rebuilt EVERY run (incrementally). Skipping the build when a
# binary exists would compare a stale `default` -- one built before the change
# under test -- against a fresh `legacymc`, which is an A/B of two old trees.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$ROOT"

. "$SCRIPT_DIR/_identify.sh"
require_machine "E5-2420" "Xeon E5-2420 v2"

export BENCH_PROFILE=xeon-e5-2420-mc
# Its own directory: the runner deletes the arm CSVs it is about to write, and
# benchmarks/data/xeon-e5-2420 holds committed `default` results from #430.
export OUTDIR=${OUTDIR:-benchmarks/data/xeon-e5-2420-mc458}
export BUILD_DIR=${BUILD_DIR:-build-release}
DEFAULT_PCPUS=0,1,2,3,4,5
export BENCH_PCPUS=${BENCH_PCPUS:-$DEFAULT_PCPUS}
export THREADS=${THREADS:-"1 2 4 6"}
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
    echo "  pinning: $BENCH_PCPUS (one id per physical core)"
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
