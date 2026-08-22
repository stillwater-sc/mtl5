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
# CAPTURE the preflight key-values, do not just run them. This script ran
# preflight and threw its output away, so the machine state -- governor, thermal
# headroom, competing load, the dirty-tree verdict -- reached the operator's
# terminal and never the sidecar. run_blocking_ab.sh has appended it since #442;
# the integer suite did not, which is why no committed int_arms sidecar records
# what the machine was doing while it was measured.
PREFLIGHT_KV=""
PREFLIGHT_ARGS=(--ignore-path "$OUTDIR")
[ "$ALLOW_DIRTY" = "1" ] && PREFLIGHT_ARGS+=(--allow-dirty)
if [ -x benchmarks/preflight.sh ]; then
    if ! PREFLIGHT_KV=$(benchmarks/preflight.sh "${PREFLIGHT_ARGS[@]}"); then
        printf '%s\n' "$PREFLIGHT_KV" >&2
        echo "preflight failed -- not measuring." >&2
        exit 1
    fi
    printf '%s\n' "$PREFLIGHT_KV"
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
# bench_all prints the compiled SIMD target and, PER OPERAND PAIRING, whether the
# quad dot is native. Two lines now, because support is per pairing and the ISAs
# disagree about which one they implement: `u8 x i8` is native on x86 and
# emulated on a Cortex-A78, where the symmetric pairings are the native ones.
BANNER="$("$BENCH" --suite int-dot --int-sizes 8 2>/dev/null | grep -E '^SIMD backend:|^ +u8\*i8' || true)"
echo "== $BANNER"

# PARTIAL is the COMMON case, not an edge case: every machine measured so far
# that has the instruction at all has it for some pairings and not others.
# AVX3_DL gets `u8 x i8` and emulates the symmetric pair; NEON+DotProd gets the
# symmetric pair and emulates `u8 x i8`. Only AVX10.2 and NEON+I8MM are `all`.
# So a partial build is a legitimate measurement -- it just has to be LABELLED,
# because half its arms are decomposed and nothing in a timing says which half.
if grep -q "int8 quad dot: PARTIAL" <<< "$BANNER"; then
    echo ""
    echo "This build is PARTIALLY native: some pairings use the hardware op and"
    echo "some are emulated at roughly three times the work. See the per-pairing"
    echo "line above -- an arm on the emulated side is not comparable with the"
    echo "same-named arm on a machine where it was native."
    echo ""
    LABEL="native-int-partial"
elif ! grep -q "int8 quad dot: NATIVE" <<< "$BANNER"; then
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
    echo "Reaching AVX3_DL gets PARTIAL, not full native: it provides the mixed"
    echo "u8 x i8 form and emulates both symmetric ones, which need AVX10.2."
    echo "On ARM it is the other way round -- FEAT_DotProd gives the symmetric"
    echo "pairings and the mixed one needs I8MM."
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
    echo "harness=run_int_bench.sh"
    [ -n "$PREFLIGHT_KV" ] && printf '%s\n' "$PREFLIGHT_KV"
    echo "$BANNER"
} >> "$CSV.sysinfo" 2>/dev/null || true

echo ""
echo "Done. $CSV (with .sysinfo)."
echo "Reminder: a dot is bandwidth-bound at large n -- int8 moves 1 byte per"
echo "element against fp64's 8, so a large-n speedup is traffic, not VNNI."
echo "The instruction shows at the L1-resident sizes. Read the curve."
