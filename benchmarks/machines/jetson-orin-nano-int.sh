#!/usr/bin/env bash
# Integer arms on the Jetson Orin Nano (Tegra234, 6x Cortex-A78AE) -- #451.
#
# NOT to be confused with jetson-orin-nano.sh, which runs the cache-blocking A/B.
# Same machine, different experiment; this one is the int suite.
#
# THIS IS THE SECOND NATIVE MACHINE, AND THE FIRST THAT IS NOT x86.
#
# Cortex-A78AE is Armv8.2-A with FEAT_DotProd, so `SumOfMulQuadAccumulate`
# compiles to `SDOT`/`UDOT` -- a real quad multiply-accumulate, not a
# decomposition. Until this profile existed the programme had exactly ONE native
# datapoint (Zen 4), on hardware whose AVX-512 is double-pumped through a
# 256-bit datapath, and the hardware plan's ranked question was whether §6 is
# about algorithms or about x86. This machine answers part of that for free.
#
# THE ASYMMETRY IS MIRRORED, AND THAT IS THE FIND. Read this before interpreting
# anything. Both ISAs implement one 8-bit pairing natively and emulate the other
# -- but they pick OPPOSITE ones (verified in Highway's own sources, x86_128-inl.h
# and arm_neon-inl.h, not inferred):
#
#                       x86 AVX3_DL (Zen 4)        ARM NEON+DotProd (this part)
#   u8 x i8    native  `vpdpbusd`                  EMULATED -- needs I8MM
#                                                  (2x UDOT + shift + subtract)
#   i8 x i8    EMULATED -- needs AVX10.2           native  `SDOT`
#              (2x dpbusd + shift + subtract)
#   u8 x u8    native                              native  `UDOT`
#
# So `gemm_u8i8_i32_quad`, the arm that is VNNI's *native* shape and the fastest
# on every x86 measured, is the EMULATED one here, and `gemm_i8_i32_quad`, the
# emulated one on x86, is native. A cross-machine comparison that pairs arms by
# NAME rather than by whether they were native will therefore get the sign of the
# effect wrong. This is the single most misreadable thing in the int suite.
#
# A78AE has FEAT_DotProd but NOT FEAT_I8MM (Armv8.6); the Orin's Cortex-A78AE
# cores are Armv8.2. If a future ARM part here reports I8MM, the mixed arm
# becomes native too and the table above changes.
#
# WHAT THE GUARD WILL SAY, AND WHY IT IS COARSER THAN THE TRUTH.
# `mtl::simd::has_native_quad_dot` is ONE boolean, keyed on `__ARM_FEATURE_DOTPROD`
# alone, so the banner will read
#
#     SIMD backend:    NEON   int8 quad dot: NATIVE (vpdpbusd / SDOT)
#
# and run_int_bench.sh will proceed WITHOUT --allow-decomposed -- correctly, for
# the symmetric arm. It is not correct for `gemm_u8i8_i32_quad` / `dot_u8i8_i32`,
# which are emulated on this part despite that banner. The flag cannot currently
# express "native for some pairings", so the CSV must be read with this comment
# beside it. Recorded as a known limitation rather than worked around here,
# because widening the flag changes what it means on x86 too.
#
# POWER MODE DECIDES WHAT THE NUMBERS MEAN, so the output directory follows it: a
# 15 W run and a MAXN run land in different directories and can never be silently
# averaged. Same rule as the A/B profile.
#
# Usage:  bash benchmarks/machines/jetson-orin-nano-int.sh
#         sudo nvpmodel -m 0 && sudo jetson_clocks   # MAXN, DVFS pinned
#         bash benchmarks/machines/jetson-orin-nano-int.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Refuse to write Jetson numbers on something that is not one (#439).
if [ -r /proc/device-tree/model ]; then
    MODEL=$(tr -d '\0' < /proc/device-tree/model)
else
    MODEL=unknown
fi
case "$MODEL" in
    *Orin*) ;;
    *) echo "This profile is for a Jetson Orin; this machine reports: $MODEL" >&2
       echo "Refusing to write Orin data. Set FORCE=1 if this really is one." >&2
       [ "${FORCE:-0}" = 1 ] || exit 2 ;;
esac

# The power mode selects the clock caps, so it selects what the run means. Read
# it rather than asking the operator to remember; `nvpmodel -q` prints
#   NV Power Mode: 15W
#   0
MODE="unknown"
if command -v nvpmodel >/dev/null 2>&1; then
    MODE=$(nvpmodel -q 2>/dev/null | sed -n 's/^NV Power Mode:[[:space:]]*//p' | head -1)
    [ -n "$MODE" ] || MODE="unknown"
fi
if [ "$MODE" = unknown ]; then
    echo "WARNING: could not read the power mode (nvpmodel missing, or it needs root)." >&2
    echo "         The mode decides the clock caps, so a run without it cannot be" >&2
    echo "         compared with one that has it. Set JETSON_MODE=<name> to record" >&2
    echo "         it by hand, or run: sudo nvpmodel -q" >&2
    MODE=${JETSON_MODE:-unknown}
fi
MODE=$(printf '%s' "$MODE" | tr -d ' /')

# Clocks: without `jetson_clocks` the early reps run at a lower clock than the
# late ones, which reads as a warm-up effect and is not one. Asked of sysfs
# rather than of `jetson_clocks --show`, which needs root on some images.
CLOCKS=unpinned
CPUFREQ=/sys/devices/system/cpu/cpu0/cpufreq
if [ -r "$CPUFREQ/scaling_min_freq" ] && [ -r "$CPUFREQ/scaling_max_freq" ]; then
    if [ "$(cat "$CPUFREQ/scaling_min_freq")" = "$(cat "$CPUFREQ/scaling_max_freq")" ]; then
        CLOCKS="pinned at $(( $(cat "$CPUFREQ/scaling_max_freq") / 1000 )) MHz"
    else
        CLOCKS="unpinned ($(( $(cat "$CPUFREQ/scaling_min_freq") / 1000 ))-$(( $(cat "$CPUFREQ/scaling_max_freq") / 1000 ))) MHz"
    fi
fi
if [ "${CLOCKS#unpinned}" != "$CLOCKS" ]; then
    echo "NOTE: CPU clocks are $CLOCKS. Run 'sudo jetson_clocks' first if you want" >&2
    echo "      this session comparable with a pinned one." >&2
fi

# Six A78AE cores, no SMT, all online.
PIN="${PIN:-0,1,2,3,4,5}"

# -mcpu=native picks up +dotprod on this part. If a toolchain rejects it, pass
# --arch '-mcpu=cortex-a78ae' through; the dot-product extension is part of that
# core's baseline, so the guard should still report NATIVE.
ARCH="${JETSON_ARCH:--mcpu=native}"

echo "profile: jetson-orin-nano-int"
echo "  model:      $MODEL"
echo "  power mode: $MODE   (clocks: $CLOCKS)"
echo "  arch:       $ARCH   (Armv8.2 + FEAT_DotProd -> native SDOT/UDOT)"
echo "  pinning:    $PIN    (six physical cores, no SMT)"
echo "  outdir:     benchmarks/data/jetson-orin-nano-${MODE}"
echo ""
echo "  READ THE HEADER OF THIS FILE BEFORE COMPARING WITH x86:"
echo "    i8 x i8   is NATIVE here and emulated on x86"
echo "    u8 x i8   is EMULATED here (no I8MM) and native on x86"
echo "  The guard's 'NATIVE' banner refers to the symmetric pairing only."
echo ""
echo "  thermal now: $(cat /sys/devices/virtual/thermal/thermal_zone*/temp 2>/dev/null | \
                       awk '{printf "%.1fC ", $1/1000}')"
echo ""

# No --allow-decomposed: the symmetric arm really is native here, so the guard
# should pass on its own. If it does NOT, the build did not get FEAT_DotProd and
# that is worth stopping for rather than measuring around.
exec "$REPO_ROOT/benchmarks/run_int_bench.sh" \
    --arch "$ARCH" \
    --outdir "benchmarks/data/jetson-orin-nano-${MODE}" \
    --pin "$PIN" \
    "$@"
