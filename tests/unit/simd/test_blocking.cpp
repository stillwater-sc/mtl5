// Tests for hw_traits + constexpr GEMM blocking-parameter derivation (#85).
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include <mtl/simd/blocking.hpp>

#include <cstddef>

using mtl::simd::hw_traits;
using mtl::simd::blocking_params;
using mtl::simd::derive_blocking;
using mtl::simd::default_hw_traits;

TEST_CASE("constexpr integer helpers", "[simd][blocking]") {
    using namespace mtl::simd::detail;
    STATIC_REQUIRE(ceil_div(10, 4) == 3);
    STATIC_REQUIRE(round_up(5, 4) == 8);
    STATIC_REQUIRE(round_down(10, 4) == 8);
    STATIC_REQUIRE(isqrt_ceil(32) == 6);   // 6*6=36 >= 32, 5*5=25 < 32
    STATIC_REQUIRE(isqrt_ceil(64) == 8);
    STATIC_REQUIRE(isqrt_ceil(0) == 0);
}

TEST_CASE("AVX2 double derivation matches published Haswell-class values", "[simd][blocking]") {
    // Nvec=4 (AVX2 double), default Haswell-class traits.
    //
    // 6x8, not the 4x8 this pinned before #351. The tile is now sized against
    // the REGISTER FILE (12 accumulator vectors of 16 ymm) rather than at the
    // dependent-FMA latency floor, which is a lower bound the old derivation
    // used as a target. 6x8 is also BLIS's hand-written Haswell dgemm tile, and
    // it measured +17% single-core over 4x8 on an i7-12700K.
    constexpr blocking_params bp = derive_blocking<double>(4);
    STATIC_REQUIRE(bp.mr == 6);
    STATIC_REQUIRE(bp.nr == 8);     // NRVEC=2 vectors of Nvec=4; matches BLIS Haswell 6x8
    STATIC_REQUIRE(bp.kc == 256);   // (32KB/2)/(8*8)
    // (256KB/2)/(256*8) = 64, NOT rounded to a multiple of mr (#408). It was 60
    // here until that rounding was removed: coupling the L2 block size to the
    // register tile meant #382's mr 4 -> 6 silently re-blocked the m dimension,
    // which cost -7.4% on 8 threads at m = 1024. mc is a cache quantity and now
    // derives from the cache alone; detail::balanced_mc lowers it against the
    // thread count at runtime.
    STATIC_REQUIRE(bp.mc == 64);
    STATIC_REQUIRE(bp.nc == 4096);  // 8MB/(256*8), multiple of nr

    // The accumulators fit the file with room for the B panel and the A
    // broadcast -- the constraint the old derivation did not model at all.
    constexpr std::size_t acc_vectors = bp.mr * (bp.nr / 4);
    STATIC_REQUIRE(acc_vectors == 12);
    STATIC_REQUIRE(acc_vectors + 2 + 1 <= default_hw_traits.vec_registers);

    // ...and still clear the latency floor it used to sit exactly on.
    STATIC_REQUIRE(bp.mr * bp.nr
                   > 4 * default_hw_traits.fma_latency * default_hw_traits.fma_units);
}

TEST_CASE("register-file budget binds above the latency floor (#351)", "[simd][blocking]") {
    // The floor is a LOWER bound, not a target. On a register-poor target it
    // binds and the tile falls back; on a normal one the file budget governs.
    hw_traits poor = default_hw_traits;
    poor.vec_registers = 8;
    const blocking_params bp_poor = derive_blocking<double>(4, poor);
    CHECK(bp_poor.mr * bp_poor.nr >= 4 * poor.fma_latency * poor.fma_units);

    // A 32-register file (AVX-512, NEON, SVE) buys a wider tile, at the same
    // ~3/4 occupancy that measured best on 16.
    hw_traits wide = default_hw_traits;
    wide.vec_registers = 32;
    const blocking_params bp_wide = derive_blocking<double>(8, wide);
    CHECK(bp_wide.mr * (bp_wide.nr / 8) == 24);
    CHECK(bp_wide.mr * (bp_wide.nr / 8) <= wide.vec_registers);
}

namespace {
// Structural + cache-residency invariants the model must always satisfy.
void check_valid(const blocking_params& bp, std::size_t nvec, std::size_t sdata,
                 const hw_traits& hw) {
    INFO("nvec=" << nvec << " sdata=" << sdata
         << " mr=" << bp.mr << " nr=" << bp.nr << " kc=" << bp.kc
         << " mc=" << bp.mc << " nc=" << bp.nc);
    CHECK(bp.mr >= 1);
    CHECK(bp.nr >= 1);
    CHECK(bp.mr * bp.nr >= nvec * hw.fma_latency * hw.fma_units);  // enough accumulators (Eq.1)
    CHECK(bp.nr % nvec == 0);                                      // vector dimension
    CHECK(bp.kc >= 1);
    // mc is deliberately NOT required to be a multiple of mr (#408). It is a
    // CACHE quantity; tying it to the register tile meant #382's mr 4 -> 6 also
    // re-blocked the m dimension (mc 64 -> 60), which broke the threaded
    // partition's divisibility and cost -7.4% on 8 threads at m = 1024. pack_A
    // already handles a ragged final panel -- it must, since m is not generally
    // a multiple of mr -- so the constraint bought nothing and cost that.
    //
    // nc keeps its multiple-of-nr requirement: the jc partition is over whole
    // nr-column panels, and nc is L3-sized so it is nowhere near as sensitive.
    CHECK(bp.mc >= 1);
    CHECK(bp.nc % bp.nr == 0);
    CHECK(bp.kc * bp.nr * sdata <= hw.l1_bytes);   // B micro-panel resident in L1
    CHECK(bp.mc * bp.kc * sdata <= hw.l2_bytes);   // packed A block resident in L2
    CHECK(bp.kc * bp.nc * sdata <= hw.l3_bytes);   // packed B panel resident in L3
}
}

TEMPLATE_TEST_CASE("derivation satisfies the blocking invariants across SIMD widths", "[simd][blocking]", float, double) {
    const std::size_t sdata = sizeof(TestType);
    const std::size_t nvecs[] = {1, 2, 4, 8, 16};
    for (std::size_t nvec : nvecs) {
        check_valid(derive_blocking<TestType>(nvec), nvec, sdata, default_hw_traits);
    }
}

TEST_CASE("derivation adapts to a different hardware profile (AVX-512 / bigger caches)", "[simd][blocking]") {
    constexpr hw_traits avx512{
        /*fma_latency*/ 4, /*fma_units*/ 2,
        /*l1_bytes*/ 32u * 1024, /*l1_assoc*/ 8, /*line_bytes*/ 64,
        /*l2_bytes*/ 1024u * 1024,           // 1 MB L2 (Skylake-X)
        /*l3_bytes*/ 16u * 1024 * 1024,
        /*page_bytes*/ 4096,
        /*vec_registers*/ 32,          // AVX-512: 32 zmm
    };
    constexpr blocking_params bp = derive_blocking<double>(8, avx512);   // 8 doubles = AVX-512
    STATIC_REQUIRE(bp.nr % 8 == 0);
    check_valid(bp, 8, sizeof(double), avx512);
    // bigger L2/L3 than the default => larger mc/nc than the AVX2 default run
    CHECK(bp.nc >= 4096);
}

TEMPLATE_TEST_CASE("default_blocking compiles and is valid for the build's SIMD width", "[simd][blocking]", float, double) {
    constexpr blocking_params bp = mtl::simd::default_blocking<TestType>;
    check_valid(bp, mtl::simd::width<TestType>, sizeof(TestType), default_hw_traits);
}
