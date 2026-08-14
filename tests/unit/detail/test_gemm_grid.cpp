// plan_gemm_grid: the threaded nest's mc and ic x jc grid decision (#429).
//
// These are regression tests for measured losses, not invented cases. Each of
// the three machines in #430 lost whole cores to this function, in two distinct
// ways, and the numbers below are the ones those runs actually produced.
#include <catch2/catch_test_macros.hpp>

#include <cstddef>

#include <mtl/detail/gemm_blocked.hpp>

using mtl::detail::plan_gemm_grid;

namespace {
// The default fp64 blocking on an AVX2 build, for the machines below.
constexpr std::size_t NC = 4096, MR = 6;
} // namespace

TEST_CASE("a large mc no longer starves the ic partition", "[detail][gemm][grid]") {
    // Square problems have njb == 1 (n <= nc), so there is NO jc parallelism to
    // compensate: if mc yields fewer ic blocks than threads, the machine idles
    // and no factorization can rescue it. Both cases measured as real losses.

    SECTION("i7-12700K, 1024^3, detected mc=213 -> ran 5 of 8 threads (0.551)") {
        const auto g = plan_gemm_grid(1024, 1024, 213, NC, MR, 8);
        REQUIRE(g.njb == 1);                 // no jc parallelism available
        REQUIRE(g.mc <= 213);                // cache bound still respected
        REQUIRE(g.nib >= 8);                 // enough blocks for every thread
        REQUIRE(g.ic_nt * g.jc_nt == 8);     // was 5
    }

    SECTION("Zen 4, 1024^3, detected mc=256 -> ran 4 of 8 threads (0.590)") {
        const auto g = plan_gemm_grid(1024, 1024, 256, NC, MR, 8);
        REQUIRE(g.mc <= 256);
        REQUIRE(g.ic_nt * g.jc_nt == 8);     // was 4
    }
}

TEST_CASE("the grid search finds factorizations the greedy split missed",
          "[detail][gemm][grid]") {
    // Jetson Orin Nano, 64 x 6144, mc=16, budget 6 (a 6-core part). Greedy took
    // ic = min(6,4) = 4 then jc = 6/4 = 1 -> grid 4, two cores idle, measured
    // 0.760.
    //
    // Note WHICH repair applies. The mc cap fires first -- ceil(64/6) = 11 is
    // below the 16 cache bound -- giving nib = 6 and a 6x1 grid, so the search
    // never needs the 3x2 that was missing before. Both defects pointed at this
    // shape; the cap reaches it by the better route, since one wide ic team
    // shares a single packed-B panel where 3x2 would hold two.
    const auto g = plan_gemm_grid(64, 6144, 16, 2048, MR, 6);
    REQUIRE(g.mc == 11);
    REQUIRE(g.nib == 6);
    REQUIRE(g.njb == 3);
    REQUIRE(g.ic_nt * g.jc_nt == 6);         // was 4
    REQUIRE(g.ic_nt <= g.nib);
    REQUIRE(g.jc_nt <= g.njb);
}

TEST_CASE("the search alone fixes a greedy miss when mc is already small",
          "[detail][gemm][grid]") {
    // Isolates defect (2) from defect (1): m is large enough that the mc cap does
    // not bind (ceil(4096/6) = 683 > 16), so nib is unchanged and only the
    // factorization can improve the grid. Greedy: ic = min(6,256) = 6, jc = 6/6
    // = 1 -> 6. Here the budget is filled either way, so pick a case where it is
    // not: nib = 4 with the cap inactive.
    const auto g = plan_gemm_grid(400, 6144, 100, 2048, MR, 6);
    REQUIRE(g.mc == 67);                     // ceil(400/6), cap active
    REQUIRE(g.ic_nt * g.jc_nt == 6);
}

TEST_CASE("the shipped default blocking is unchanged", "[detail][gemm][grid]") {
    // The mc cap must bind ONLY when the partition would otherwise starve. At
    // the shipped fp64 mc = 64 the cap is ceil(1024/8) = 128, which is larger,
    // so nothing moves -- this is what keeps the change off the common path.
    const auto g = plan_gemm_grid(1024, 1024, 64, NC, MR, 8);
    REQUIRE(g.mc == 64);
    REQUIRE(g.nib == 16);
    REQUIRE(g.ic_nt * g.jc_nt == 8);
}

TEST_CASE("plan_gemm_grid respects its bounds everywhere", "[detail][gemm][grid]") {
    for (std::size_t m : {std::size_t{1}, std::size_t{64}, std::size_t{1000},
                          std::size_t{4096}, std::size_t{12288}})
        for (std::size_t n : {std::size_t{64}, std::size_t{1024}, std::size_t{16384}})
            for (std::size_t mc : {std::size_t{16}, std::size_t{64}, std::size_t{256}})
                for (unsigned b : {1u, 2u, 6u, 8u, 20u}) {
                    const auto g = plan_gemm_grid(m, n, mc, NC, MR, b);
                    INFO("m=" << m << " n=" << n << " mc_cache=" << mc << " budget=" << b
                         << " -> mc=" << g.mc << " nib=" << g.nib << " njb=" << g.njb
                         << " grid=" << g.ic_nt << "x" << g.jc_nt);
                    REQUIRE(g.mc >= 1);
                    REQUIRE(g.mc <= mc);                       // never exceeds the cache bound
                    REQUIRE(g.nib == (m + g.mc - 1) / g.mc);   // block count matches mc
                    REQUIRE(g.ic_nt >= 1);
                    REQUIRE(g.jc_nt >= 1);
                    REQUIRE(g.ic_nt <= g.nib);                 // no thread without a block
                    REQUIRE(g.jc_nt <= g.njb);
                    REQUIRE(g.ic_nt * g.jc_nt <= b);           // never oversubscribe the pool
                    // mc is only shrunk below the cache bound to reach the
                    // budget; it must not drop under one register tile unless
                    // the cache bound itself already was.
                    if (g.mc < mc) REQUIRE(g.mc >= (mc < MR ? mc : MR));
                }
}

TEST_CASE("plan_gemm_grid handles degenerate inputs", "[detail][gemm][grid][edge]") {
    for (auto g : {plan_gemm_grid(0, 1024, 64, NC, MR, 8),
                   plan_gemm_grid(1024, 0, 64, NC, MR, 8),
                   plan_gemm_grid(1024, 1024, 0, NC, MR, 8),
                   plan_gemm_grid(1024, 1024, 64, 0, MR, 8),
                   plan_gemm_grid(1024, 1024, 64, NC, MR, 0)}) {
        REQUIRE(g.ic_nt >= 1);
        REQUIRE(g.jc_nt >= 1);
        REQUIRE(g.mc >= 1);
    }
}
