#!/usr/bin/env bash
# #643 register-tile sweep on the Arduino Ventuno Q (4x Cortex-A78C).
#
# aarch64 builds now derive mr = 8 (8x4 fp64, 8x8 fp32), from the fp64 sweep on
# the Ventuno Q; before #643 they derived AVX2's 6. This sweeps mr at fixed nr,
# kc, mc and nc around it:
#
#     default   mr = 8     the shipped tile, and where the shapes come from
#     tile6     mr = 6     the pre-#643 tile, 12 accumulators of 32
#     tile10    mr = 10    20
#     tile12    mr = 12    24 = 3/4 of the file (#351's AVX2 rule, `isaregs`)
#     tile14    mr = 14    28, plus 2 B vectors and 1 A broadcast = 31 of 32
#
# Compare every arm against tile6 for the gain over the old tile, and against
# default for whether 8 is still the peak.
#
# The fp64 Ventuno Q run put the peak at 8 (x1.17 at T=1, x1.10 at T=4) and
# found 12 losing at T=4. What it did not cover: fp32, and every other aarch64
# core. docs/benchmarks/systems.md has the table.
#
# One run per element type, each into its own directory (`...-tile-double`,
# `...-tile-float`), at T=1 and on all four A78C cores. Everything else -- pinning,
# clock/power-mode checks, the output directory's machine state -- is the base
# profile's, which this calls once per type. The first call rebuilds every arm
# so `default` is never stale against the tile arms.
#
# Usage:  benchmarks/machines/arduino-ventuno-q-tile.sh
#         DTYPES=double benchmarks/machines/arduino-ventuno-q-tile.sh
#         ARMS="default tile12" benchmarks/machines/arduino-ventuno-q-tile.sh
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export ARMS=${ARMS:-"default tile6 tile10 tile12 tile14"}
DTYPES=${DTYPES:-"double float"}

force=1
for dt in $DTYPES; do
    FORCE_BUILD=$force DTYPE=$dt BENCH_VARIANT="tile-$dt" "$SCRIPT_DIR/arduino-ventuno-q.sh"
    force=0
done
