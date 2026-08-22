#!/usr/bin/env bash
# Integer arms on the Ryzen 9 8945HS (Zen 4 / Hawk Point) -- #451 phase 4.
#
# WHY THIS IS A SHELL SCRIPT FOR A WINDOWS MACHINE.
#
# The sibling profile, ryzen-9-8945hs.ps1, builds with MSVC. MSVC cannot produce
# a VNNI build of this kernel. Highway selects its AVX3_DL target -- the only x86
# target with `vpdpbusd` -- on the conjunction of SEVEN predefined macros:
#
#     __AVX512VNNI__ __VAES__ __VPCLMULQDQ__ __AVX512VBMI__
#     __AVX512VBMI2__ __AVX512VPOPCNTDQ__ __AVX512BITALG__
#
# MSVC's /arch:AVX512 covers AVX-512 F/CD/BW/DQ/VL and defines none of them, and
# MSVC has no /arch for VNNI. Highway's own source carries the comment "not yet
# known whether these will be set by MSVC". Verified here: `-march=znver4` under
# gcc defines all seven and emits `vpdpbusd`; MSVC does not.
#
# So the VNNI measurement on this machine has to come from gcc or clang -- in
# WSL, or with clang-cl. Run this under WSL:
#
#     bash benchmarks/machines/ryzen-9-8945hs-int.sh
#
# For native Windows with clang-cl, pass the equivalent through instead:
#
#     -DCMAKE_CXX_COMPILER=clang-cl -DCMAKE_CXX_FLAGS="/clang:-march=znver4"
#
# Either way the run REFUSES to proceed on a decomposed build (run_int_bench.sh
# checks the compiled target before timing anything), so a mistake here costs a
# message rather than a plausible-looking CSV.
#
# EXPECT `PARTIAL`, NOT `NATIVE`. AVX3_DL implements the mixed `u8 x i8` form
# (`vpdpbusd`) and EMULATES both symmetric ones -- `vpdpbssd`/`vpdpbuud` arrive
# only with AVX10.2. So `dot_u8i8_i32` and `gemm_u8i8_i32_quad` are the native
# arms here and `dot_i8_i32` / `gemm_i8_i32_quad` are not, and the run is
# labelled `native-int-partial`. Committed CSVs taken before the flag became
# per-pairing say `native-int`, which over-claimed for the symmetric arms. That guard exists because this
# machine has already produced one silently-wrong run: the first Zen 4 A/B was
# an AVX2 build when AVX-512 was intended, and nothing in the data could say so.
#
# WHAT TO EXPECT, so a result is not over-read. A dot product is a streaming
# reduction: at large n it is bandwidth-bound, and int8 moves one byte per
# element where fp64 moves eight. Most of a large-n speedup is therefore traffic,
# on any machine, VNNI or not. The instruction shows at the L1-resident sizes.
# The Xeon E5-2420 v2 (SSE4, no VNNI) is the control: it shows ~1.0x at n=1024
# and ~7x at n=4M. A Zen 4 advantage at the SMALL end that the Xeon lacks is the
# instruction; the large end is bytes on both.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Refuse to write this machine's numbers on a different machine: the CSVs are
# named by suite, not by host, so a profile run on the wrong box replaces
# committed evidence with data from somewhere else (#439).
EXPECT="8945HS"
ACTUAL="$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2- | sed 's/^ *//' || echo unknown)"
if [[ "$ACTUAL" != *"$EXPECT"* ]]; then
    echo "This profile is for a Ryzen 9 $EXPECT; this machine reports:"
    echo "  $ACTUAL"
    if [ "${FORCE:-0}" != "1" ]; then
        echo "Refusing to write $EXPECT data. Set FORCE=1 if this really is one."
        exit 2
    fi
fi

# One logical id per physical core. Eight homogeneous Zen 4 cores; SMT siblings
# are the odd ids, and pinning to one-per-core is what the other profiles do.
PIN="${PIN:-0,2,4,6,8,10,12,14}"

echo "profile: ryzen-9-8945hs-int"
echo "  cpu:     $ACTUAL"
echo "  arch:    -march=znver4   (defines all seven AVX3_DL macros; MSVC does not)"
echo "  pinning: $PIN"
echo ""

exec "$REPO_ROOT/benchmarks/run_int_bench.sh" \
    --arch "-march=znver4" \
    --outdir "benchmarks/data/ryzen-9-8945hs" \
    --pin "$PIN" \
    "$@"
