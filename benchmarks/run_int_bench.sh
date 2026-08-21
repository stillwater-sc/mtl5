#!/usr/bin/env bash
# Integer-arm benchmark run (#451 phase 4).
#
# THE POINT OF THIS SCRIPT IS THE GUARD, not the invocation.
#
# The int8 kernel compiles either to the hardware's quad multiply-accumulate
# (`vpdpbusd` on x86, `SDOT` on NEON) or to a decomposition into `vpmaddwd`
# that is correct and several times longer. Nothing in a timing tells you which
# you got. Worse, having the *hardware* is not sufficient: Highway selects its
# VNNI target only on the full AVX3_DL conjunction -- VNNI plus VBMI, VBMI2,
# BITALG, VPOPCNTDQ, VAES and VPCLMULQDQ -- so an AVX-512 machine with VNNI can
# still compile to the slow path, and MSVC's /arch:AVX512 does not define those
# macros at all.
#
# This is not hypothetical for this repository: the first Zen 4 A/B was an AVX2
# run when AVX-512 was intended, and no CSV could say so until `build_isa` was
# added (see benchmarks/machines/ryzen-9-8945hs.ps1). So this script REFUSES to
# measure a decomposed build unless told to on purpose, and records what it got
# either way.
#
# Usage:
#   benchmarks/run_int_bench.sh --outdir benchmarks/data/<machine> [options]
#
#   --arch <flag>     compiler arch flag (default: -march=native)
#   --outdir <dir>    REQUIRED, one per machine
#   --pin <cpus>      taskset CPU list, e.g. 0,2,4,6
#   --allow-decomposed   measure anyway, and say so in the label
#   --allow-dirty     permit a dirty tree (still recorded)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCH="-march=native"
OUTDIR=""
PIN=""
ALLOW_DECOMPOSED=0
ALLOW_DIRTY=0

while [ $# -gt 0 ]; do
    case "$1" in
        --arch)   ARCH="$2"; shift 2 ;;
        --outdir) OUTDIR="$2"; shift 2 ;;
        --pin)    PIN="$2"; shift 2 ;;
        --allow-decomposed) ALLOW_DECOMPOSED=1; shift ;;
        --allow-dirty)      ALLOW_DIRTY=1; shift ;;
        -h|--help) sed -n '1,32p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

if [ -z "$OUTDIR" ]; then
    echo "error: --outdir is required (one directory per machine)." >&2
    echo "       The CSVs are named by suite, not by machine, so a shared" >&2
    echo "       default silently overwrites another machine's committed data." >&2
    exit 2
fi

cd "$REPO_ROOT"

# ---- run contract: preflight before building, so build logs cannot dirty it --
PREFLIGHT_ARGS=(--ignore-path "$OUTDIR")
[ "$ALLOW_DIRTY" = "1" ] && PREFLIGHT_ARGS+=(--allow-dirty)
if [ -x benchmarks/preflight.sh ]; then
    benchmarks/preflight.sh "${PREFLIGHT_ARGS[@]}"
fi

BUILD_DIR="build-int-bench"
echo "== configuring ($ARCH) =="
cmake -S . -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DMTL5_BUILD_BENCHMARKS=ON -DMTL5_BUILD_TESTS=OFF -DMTL5_BUILD_EXAMPLES=OFF \
    -DMTL5_WITH_HIGHWAY=ON -DMTL5_NATIVE_FAST_GEMM=ON \
    -DCMAKE_CXX_FLAGS="$ARCH" > /dev/null
cmake --build "$BUILD_DIR" --target bench_all --parallel "$(nproc)" > /dev/null

BENCH="$BUILD_DIR/benchmarks/bench_all"

# ---- the guard --------------------------------------------------------------
# bench_all prints the compiled SIMD target and whether the quad dot is native.
BANNER="$("$BENCH" --suite int-dot --int-sizes 8 2>/dev/null | grep -E '^SIMD backend:' || true)"
echo "== $BANNER"
if ! grep -q "NATIVE" <<< "$BANNER"; then
    echo ""
    echo "This build does NOT have the native int8 quad multiply-accumulate."
    echo "Timing it would measure the decomposition, not the instruction --"
    echo "and the CSV would look exactly the same either way."
    echo ""
    echo "On x86 Highway needs the whole AVX3_DL set, not just AVX512VNNI:"
    echo "  __AVX512VNNI__ __VAES__ __VPCLMULQDQ__ __AVX512VBMI__"
    echo "  __AVX512VBMI2__ __AVX512VPOPCNTDQ__ __AVX512BITALG__"
    echo "Try --arch '-march=znver4' (Zen 4), '-march=icelake-server',"
    echo "or '-march=sapphirerapids'. MSVC cannot define these; use clang-cl,"
    echo "gcc or clang."
    echo ""
    echo "Pass --allow-decomposed to measure it anyway (recorded as such)."
    [ "$ALLOW_DECOMPOSED" = "1" ] || exit 3
    LABEL="native-int-decomposed"
else
    LABEL="native-int"
fi

mkdir -p "$OUTDIR"
CSV="$OUTDIR/int_arms.csv"
RUN=("$BENCH" --suite int --label "$LABEL" --csv "$CSV")
[ -n "$PIN" ] && RUN=(taskset -c "$PIN" "${RUN[@]}")

echo "== running: ${RUN[*]}"
"${RUN[@]}"

# ---- provenance -------------------------------------------------------------
# The sidecar is what makes the number checkable later. build_isa now carries
# AVX512VNNI / AVX3_DL / DOTPROD, so it records the decision the guard made.
{
    echo "suite=int"
    echo "label=$LABEL"
    echo "arch_flag=$ARCH"
    echo "pin=${PIN:-none}"
    echo "$BANNER"
} >> "$CSV.sysinfo" 2>/dev/null || true

echo ""
echo "Done. $CSV (with .sysinfo)."
echo "Reminder: a dot is bandwidth-bound at large n -- int8 moves 1 byte per"
echo "element against fp64's 8, so a large-n speedup is traffic, not VNNI."
echo "The instruction shows at the L1-resident sizes. Read the curve."
