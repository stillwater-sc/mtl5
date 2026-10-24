#!/usr/bin/env bash
# #643 register-tile sweep on the Ryzen 9 8945HS (Zen 4, AVX-512).
#
# The derivation assumes AVX2's 16 vector registers on every ISA, so an AVX-512
# build gets 6x16 (fp64) and 6x32 (fp32): 12 accumulator vectors of the 32 zmm.
# This sweeps mr at fixed nr, kc, mc and nc:
#
#     default   mr = 6     the shipped tile, and where the shapes come from
#     tile8     mr = 8     16 accumulators
#     tile10    mr = 10    20
#     tile12    mr = 12    24 = 3/4 of the file, what isaregs derives on AVX-512
#     tile14    mr = 14    28, plus 2 B vectors and 1 A broadcast = 31 of 32
#
# BLIS's hand-written Zen 4 / Skylake-X dgemm kernel is 8x24 (24 accumulators),
# so the 3/4 rule has a precedent here; NEON is the open question, this is the
# check that the rule carries to the other 32-register ISA.
#
# Run it under WSL with gcc >= 12 (MTL5_NATIVE_ARCH gives -march=native, which
# is znver4 there and defines __AVX512F__/__AVX512VL__; MSVC cannot build this,
# see ryzen-9-8945hs.ps1):
#
#     bash benchmarks/machines/ryzen-9-8945hs-tile.sh
#     DTYPES=double bash benchmarks/machines/ryzen-9-8945hs-tile.sh
#
# WSL2 is a VM, so the pinning below is to virtual CPUs and Windows places
# them. The arms are interleaved in one session, so the A/B stays fair; the
# absolute GF/s carry more noise than a bare-metal run. Run on AC power, in the
# "Best performance" power mode, with the machine otherwise idle.
#
# Topology: eight homogeneous cores, SMT siblings odd, so one id per core is
# 0,2,..,14. Hyperthreads are excluded.
#
# One run per element type, each into its own directory, because the runner
# deletes the CSVs of the arms it is about to write. The arms are rebuilt once,
# before the first type, so `default` is never stale against the tile arms.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$ROOT"

. "$SCRIPT_DIR/_identify.sh"
require_machine "8945HS" "Ryzen 9 8945HS"

export BUILD_DIR=${BUILD_DIR:-build-release}
DEFAULT_PCPUS=0,2,4,6,8,10,12,14
export BENCH_PCPUS=${BENCH_PCPUS:-$DEFAULT_PCPUS}
export THREADS=${THREADS:-"1 8"}
export ROUNDS=${ROUNDS:-5}
export REPS=${REPS:-5}
export ARMS=${ARMS:-"default tile8 tile10 tile12 tile14"}
DTYPES=${DTYPES:-"double float"}

echo "profile: ryzen-9-8945hs-tile"
echo "  cpu:     $MTL5_MACHINE_ID"
echo "  build:   cmake --preset release -B $BUILD_DIR -DMTL5_WITH_HIGHWAY=ON -DMTL5_NATIVE_ARCH=ON"
if [ "$BENCH_PCPUS" = "$DEFAULT_PCPUS" ]; then
    echo "  pinning: $BENCH_PCPUS (one id per physical core)"
else
    echo "  pinning: $BENCH_PCPUS (OVERRIDDEN; profile default is $DEFAULT_PCPUS)"
fi
echo "  arms:    $ARMS"
echo "  threads: $THREADS"
echo "  dtypes:  $DTYPES"
echo

echo "== configuring and building $BUILD_DIR"
cmake --preset release -B "$BUILD_DIR" -DMTL5_WITH_HIGHWAY=ON -DMTL5_NATIVE_ARCH=ON >/dev/null
targets=""
for a in $ARMS; do targets="$targets bench_blocking_ab_$a"; done
# shellcheck disable=SC2086
cmake --build "$BUILD_DIR" --target $targets -j4 >/dev/null

for dt in $DTYPES; do
    # Each type in its own directory, and a subshell per type so OUTDIR and
    # BENCH_PROFILE do not leak from one run into the next.
    ( export DTYPE=$dt
      export BENCH_PROFILE="ryzen-9-8945hs-tile-$dt"
      export OUTDIR=${OUTDIR:-benchmarks/data/ryzen-9-8945hs-tile-$dt}
      echo "== $DTYPE -> $OUTDIR"
      "$ROOT/benchmarks/run_blocking_ab.sh" )
done
