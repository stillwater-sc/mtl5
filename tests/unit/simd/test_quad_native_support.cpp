// Which 8-bit pairings this build has the quad multiply-accumulate for --
// #451 phase 5.
//
// WHY THIS IS PER PAIRING AND NOT ONE BOOLEAN. Support is per pairing in the
// hardware, and the two ISAs that have it implement OPPOSITE pairings first:
//
//                   x86 AVX3_DL   x86 AVX10.2   NEON+DotProd   NEON+I8MM
//   u8 x i8           NATIVE        native       emulated       NATIVE
//   i8 x i8          emulated       native        NATIVE        native
//   u8 x u8          emulated       native        NATIVE        native
//
// So `u8 x i8` -- the fastest arm on every x86 measured, and the shape VNNI
// exists for -- is the EMULATED one on a Cortex-A78, where the symmetric
// pairings are native. A single native/decomposed flag mislabels at least one
// arm on every machine the programme has measured, and a benchmark comparison
// that pairs arms by NAME across those machines gets the sign of the effect
// wrong. That is what this replaces.
//
// WHAT THIS FILE CAN AND CANNOT CHECK, stated because the distinction decides
// how much the green tick is worth.
//
//   It CAN check the invariants -- that the summary agrees with the parts, that
//   a pairing the op does not accept reads false. Those hold on every target and
//   are what a refactor would break.
//
//   It CANNOT independently confirm the per-target gate, because the only other
//   source of truth is Highway's own condition and asserting our copy against
//   our copy proves nothing. (Highway's `HWY_NATIVE_*_SUMOFMULQUADACCUMULATE`
//   toggles are NOT usable for this: measured on an SSE4 build, where all three
//   pairings are emulated, both macros are still DEFINED -- generic_ops-inl.h
//   defines them when it supplies the fallback.)
//
// The gate is instead verified OUT OF TREE, by compiling assertions against
// expectations read from Highway's source, per target:
//
//   x86, verified with gcc 13.3 and a negative control per arm:
//     x86-64-v2, x86-64-v3, skylake-avx512  -> none native
//     icelake-server, sapphirerapids, znver4 -> u8 x i8 only
//
//   `skylake-avx512` reading "none" is the section 7 lesson in one line:
//   AVX-512F does not imply the AVX3_DL set, so an AVX-512 machine can still
//   be emulating.
//
//   ARM was NOT compile-verified locally (no aarch64 sysroot on the dev host);
//   its clauses are transcribed from arm_neon-inl.h. The ARM64 CI jobs compile
//   this file, which exercises the branch but at baseline NEON, where the
//   expected answer is "none native". A run on the Jetson prints the per-pairing
//   line and is the real check.
#include <catch2/catch_test_macros.hpp>

#include <mtl/simd/batch.hpp>

#include <cstdint>
#include <string>

namespace {
using u8 = std::uint8_t;
using i8 = std::int8_t;
using mtl::simd::has_native_quad_dot_v;
} // namespace

TEST_CASE("the summary agrees with the per-pairing flags", "[simd][quad][native]") {
    using mtl::simd::quad_dot_native_support;
    using mtl::simd::quad_dot_support;

    constexpr bool a = has_native_quad_dot_v<u8, i8>;
    constexpr bool b = has_native_quad_dot_v<i8, i8>;
    constexpr bool c = has_native_quad_dot_v<u8, u8>;

    STATIC_REQUIRE((quad_dot_native_support == quad_dot_support::all) == (a && b && c));
    STATIC_REQUIRE((quad_dot_native_support == quad_dot_support::none) == (!a && !b && !c));
    STATIC_REQUIRE((quad_dot_native_support == quad_dot_support::partial) ==
                   ((a || b || c) && !(a && b && c)));
}

TEST_CASE("has_native_quad_dot keeps its original ANY meaning", "[simd][quad][native]") {
    // Deliberately NOT "all pairings". "All" would read false on Zen 4 and on a
    // Cortex-A78 alike -- both are partial -- and would erase the very
    // distinction the sidecars exist to record, while silently changing what
    // every committed CSV's label meant. It answers "did this build get the
    // instruction at all", which is what its callers ask.
    STATIC_REQUIRE(mtl::simd::has_native_quad_dot ==
                   (has_native_quad_dot_v<u8, i8> ||
                    has_native_quad_dot_v<i8, i8> ||
                    has_native_quad_dot_v<u8, u8>));
}

TEST_CASE("a pairing the hardware op does not accept is not native",
          "[simd][quad][native]") {
    // (i8, u8) is absent from `quad_accumulator` on purpose -- a GEMM is not
    // symmetric in its operands, so there is no swap to suggest. It must read
    // false rather than fail to compile, because callers branch on it.
    STATIC_REQUIRE_FALSE(has_native_quad_dot_v<i8, u8>);
    // Types that are not 8-bit operands at all.
    STATIC_REQUIRE_FALSE(has_native_quad_dot_v<std::int16_t, std::int16_t>);
    STATIC_REQUIRE_FALSE(has_native_quad_dot_v<std::int32_t, std::int32_t>);
    STATIC_REQUIRE_FALSE(has_native_quad_dot_v<float, float>);
}

TEST_CASE("the scalar fallback claims nothing", "[simd][quad][native]") {
    // Without the Highway backend there is no hardware op to have, so every
    // pairing must read emulated -- a scalar build that advertised NATIVE would
    // put the word into a sidecar for a loop of four scalar multiplies.
#if !MTL5_SIMD_USE_HIGHWAY
    STATIC_REQUIRE_FALSE(has_native_quad_dot_v<u8, i8>);
    STATIC_REQUIRE_FALSE(has_native_quad_dot_v<i8, i8>);
    STATIC_REQUIRE_FALSE(has_native_quad_dot_v<u8, u8>);
    STATIC_REQUIRE_FALSE(mtl::simd::has_native_quad_dot);
    CHECK(std::string(mtl::simd::quad_dot_support_name()) == "DECOMPOSED");
#else
    SUCCEED("Highway backend active; see the per-target notes in the file header");
#endif
}

TEST_CASE("the reported token matches the support level", "[simd][quad][native]") {
    // This string lands in every benchmark sidecar, so it is the thing a reader
    // years later actually has. It must not disagree with the flags above.
    using mtl::simd::quad_dot_native_support;
    using mtl::simd::quad_dot_support;
    const std::string tok = mtl::simd::quad_dot_support_name();
    switch (quad_dot_native_support) {
        case quad_dot_support::all:     CHECK(tok == "NATIVE");     break;
        case quad_dot_support::partial: CHECK(tok == "PARTIAL");    break;
        case quad_dot_support::none:    CHECK(tok == "DECOMPOSED"); break;
    }
    INFO("backend " << mtl::simd::backend_name() << " reports " << tok);
    CHECK_FALSE(tok.empty());
}

TEST_CASE("x86 without the AVX3_DL set is never native", "[simd][quad][native]") {
    // The section 7 trap, pinned where it is cheap: AVX-512F does NOT imply
    // the quad multiply-accumulate. An Alder Lake part HAS AVX-VNNI silicon and
    // still reads emulated, because Highway gates the op on its AVX3_DL target.
    // Skylake-X has AVX-512 and no VNNI at all. Either way, a build that reached
    // only AVX2 or plain AVX-512 must claim nothing.
    // `#if`, not `if constexpr`: outside a template both branches of an
    // `if constexpr` are still compiled, so a `static_assert` in the discarded
    // one fires anyway.
#if MTL5_SIMD_USE_HIGHWAY && HWY_ARCH_X86
  #if HWY_TARGET > HWY_AVX3_DL
    STATIC_REQUIRE_FALSE(has_native_quad_dot_v<u8, i8>);
    STATIC_REQUIRE_FALSE(has_native_quad_dot_v<i8, i8>);
    STATIC_REQUIRE_FALSE(has_native_quad_dot_v<u8, u8>);
  #else
    // AVX3_DL or better: the mixed form is the one that arrives first.
    STATIC_REQUIRE(has_native_quad_dot_v<u8, i8>);
  #endif
    SUCCEED("x86 gate checked against the compiled target");
#else
    SUCCEED("not an x86 Highway build");
#endif
}
