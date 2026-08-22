#!/usr/bin/env bash
# Integer arms on the i7-12700K (Alder Lake, 8P + 4E) -- #451.
#
# NOT to be confused with i7-12700k.sh in this directory, which runs the
# cache-blocking A/B (`run_blocking_ab.sh`). Same machine, different experiment,
# different CSV; this one is the int suite.
#
# THIS IS THE MACHINE FROM §7: it HAS VNNI silicon and MTL5 cannot use it.
# Alder Lake implements AVX-VNNI, the VEX-encoded 256-bit form. Highway
# implements the quad multiply-accumulate only in its AVX3_DL target, gated on
# the full seven-macro AVX-512 conjunction, and Alder Lake's AVX-512 is fused
# off. There is no compiler flag that fixes this and no Highway AVX-VNNI target
# to select -- so this part measures the decomposition, and `build_isa` records
# `AVXVNNI(unused)` precisely so a reader is not invited to rediscover it.
#
# WHY THAT MAKES IT VALUABLE RATHER THAN A CONSOLATION PRIZE. It is a MODERN
# memory system with the instruction absent, which is the one combination that
# separates traffic from arithmetic. §6 of the assessment turns on this: the
# instruction's share of the dot speedup looked like 1.42x from the Zen 4 run
# alone and came out at ~1.2x once this machine could run both arms decomposed,
# because the two arms also take different DECOMPOSITION SHAPES and Zen 4 can
# never show that -- there the mixed form is never decomposed.
#
# The same control applies to the phase-5 GEMM arms: `gemm_i8_i32_quad` here is
# the kernel shape with no instruction, on hardware twelve years newer than the
# Xeon, so the pair of them says whether ~1.25x is an artefact of a 2013 memory
# system or a property of the kernel.
#
# WHY --allow-decomposed IS BAKED IN. run_int_bench.sh refuses to time a
# decomposed build so that a misconfigured run costs a message instead of a
# plausible CSV. Here the decomposition is not misconfiguration, it is the
# capability fact under test. If this flag ever becomes unnecessary -- the guard
# reporting NATIVE on this part -- then §7 has been fixed upstream and both the
# assessment and the hardware plan need revisiting.
#
# PINNING: P-cores only. An unpinned run lets short kernels land on E-cores,
# which have different cache and different clocks; mixing them measures the
# scheduler. P-core SMT siblings are adjacent pairs, so 0,2,4,... is one id per
# physical P-core, and E-cores 16-19 are excluded entirely.
#
# Usage:  bash benchmarks/machines/i7-12700k-int.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Refuse to write this machine's numbers on a different machine (#439).
EXPECT="12700K"
ACTUAL="$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2- | sed 's/^ *//' || echo unknown)"
if [[ "$ACTUAL" != *"$EXPECT"* ]]; then
    echo "This profile is for an i7-$EXPECT; this machine reports:"
    echo "  $ACTUAL"
    if [ "${FORCE:-0}" != "1" ]; then
        echo "Refusing to write $EXPECT data. Set FORCE=1 if this really is one."
        exit 2
    fi
fi

# One logical id per physical P-core; E-cores (16-19) excluded.
PIN="${PIN:-0,2,4,6,8,10,12,14}"

# -march=alderlake, not -march=native: native would be identical here, but the
# committed sidecar records the flag, and a named target is reproducible on a
# machine whose microcode or compiler has since changed.
echo "profile: i7-12700k-int"
echo "  cpu:     $ACTUAL"
echo "  arch:    -march=alderlake   (AVX2; AVXVNNI present but unreachable, §7)"
echo "  pinning: $PIN               (P-cores, one id per core; E-cores excluded)"
echo "  expect:  int8 quad dot: decomposed  -- if it says NATIVE, §7 was fixed"
echo ""

exec "$REPO_ROOT/benchmarks/run_int_bench.sh" \
    --arch "-march=alderlake" \
    --outdir "benchmarks/data/i7-12700k" \
    --pin "$PIN" \
    --allow-decomposed \
    "$@"
