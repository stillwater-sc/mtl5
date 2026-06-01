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
    constexpr blocking_params bp = derive_blocking<double>(4);
    STATIC_REQUIRE(bp.mr == 4);
    STATIC_REQUIRE(bp.nr == 8);     // multiple of Nvec=4; matches OpenBLAS Haswell 4x8
    STATIC_REQUIRE(bp.kc == 256);   // (32KB/2)/(8*8)
    STATIC_REQUIRE(bp.mc == 64);    // (256KB/2)/(256*8), multiple of mr
    STATIC_REQUIRE(bp.nc == 4096);  // 8MB/(256*8), multiple of nr
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
    CHECK(bp.mc % bp.mr == 0);
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
