#!/usr/bin/env bash
# #643 register-tile sweep on the Jetson Orin Nano (6x Cortex-A78AE).
#
# The derivation assumes AVX2's 16 vector registers on every ISA, so a NEON
# build gets 6x4 (fp64) and 6x8 (fp32): 12 accumulator vectors of the 32 that
# aarch64 has. This sweeps mr at fixed nr, kc, mc and nc to find out whether
# a larger tile wins here before the default moves:
#
#     default   mr = 6     the shipped tile, and where the shapes come from
#     tile8     mr = 8     16 accumulators
#     tile10    mr = 10    20
#     tile12    mr = 12    24 = 3/4 of the file, what isaregs derives on NEON
#     tile14    mr = 14    28, plus 2 B vectors and 1 A broadcast = 31 of 32
#
# Not established in advance: 12 accumulators already cover the A78's FMA
# latency floor, so a bigger tile buys fewer loads per FMA, which only helps if
# the core is load-bound at 6x4. #351 measured the peak at ~3/4 of the file on
# AVX2 and a loss once the tile filled it; whether NEON agrees is the question.
#
# One run per element type, each into its own directory (`...-tile-double`,
# `...-tile-float`), at T=1 and on all six A78AE cores. Everything else -- pinning,
# clock/power-mode checks, the output directory's machine state -- is the base
# profile's, which this calls once per type. The first call rebuilds every arm
# so `default` is never stale against the tile arms.
#
# Usage:  benchmarks/machines/jetson-orin-nano-tile.sh
#         DTYPES=double benchmarks/machines/jetson-orin-nano-tile.sh
#         ARMS="default tile12" benchmarks/machines/jetson-orin-nano-tile.sh
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export ARMS=${ARMS:-"default tile8 tile10 tile12 tile14"}
DTYPES=${DTYPES:-"double float"}

force=1
for dt in $DTYPES; do
    FORCE_BUILD=$force DTYPE=$dt BENCH_VARIANT="tile-$dt" "$SCRIPT_DIR/jetson-orin-nano.sh"
    force=0
done
