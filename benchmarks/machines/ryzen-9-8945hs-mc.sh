#!/usr/bin/env bash
# #458 before/after on the Ryzen 9 8945HS (Zen 4): the threaded ic step chosen
# by the busiest thread's micro-panels (`default`) against the pre-#458
# block-count rule (`legacymc`). Same tree, same session, interleaved rounds.
#
# The third machine of #458's acceptance criterion, and the AVX-512 one. T = 1
# is a null control: the serial path is untouched, so both arms must agree there.
#
# Run it under WSL with gcc >= 12 (MTL5_NATIVE_ARCH gives -march=native, which
# is znver4 there; MSVC cannot build this, see ryzen-9-8945hs.ps1):
#
#     bash benchmarks/machines/ryzen-9-8945hs-mc.sh
#
# WSL2 is a VM, so the pinning below is to virtual CPUs and Windows places
# them. The arms are interleaved in one session, so the A/B stays fair; the
# absolute GF/s carry more noise than a bare-metal run. Run on AC power, in the
# "Best performance" power mode, with the machine otherwise idle -- WSL cannot
# see or record either setting.
#
# Topology: eight homogeneous cores, SMT siblings odd, so one id per core is
# 0,2,..,14. Hyperthreads are excluded -- two threads on one core would measure
# the sharing, not the partition.
#
# The arms are rebuilt EVERY run (incrementally). Skipping the build when a
# binary exists would compare a stale `default` against a fresh `legacymc`.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$ROOT"

. "$SCRIPT_DIR/_identify.sh"
require_machine "8945HS" "Ryzen 9 8945HS"

export BENCH_PROFILE=ryzen-9-8945hs-mc
# Its own directory: the runner deletes the arm CSVs it is about to write.
export OUTDIR=${OUTDIR:-benchmarks/data/ryzen-9-8945hs-mc458}
export BUILD_DIR=${BUILD_DIR:-build-release}
DEFAULT_PCPUS=0,2,4,6,8,10,12,14
export BENCH_PCPUS=${BENCH_PCPUS:-$DEFAULT_PCPUS}
export THREADS=${THREADS:-"1 2 4 6 8"}
export ROUNDS=${ROUNDS:-4}
export REPS=${REPS:-5}
export DTYPE=${DTYPE:-double}
export ARMS=${ARMS:-"default legacymc"}
# The same shapes as the i7 and Xeon runs, so the three machines compare.
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
