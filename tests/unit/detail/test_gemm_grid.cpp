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

/// The factorization plan_gemm_grid REPLACED: fill ic, then hand jc the integer
/// quotient. Kept here as a reference so a test can assert the planner is at
/// least as good, and strictly better where it matters. Without this the search
/// is easy to test vacuously -- most shapes are solved by the mc cap alone, and
/// a greedy split reaches the same grid.
unsigned greedy_grid(std::size_t nib, std::size_t njb, unsigned budget) {
    unsigned ic = static_cast<unsigned>(nib < budget ? nib : budget);
    if (ic < 1) ic = 1;
    unsigned jc = budget / ic;
    if (jc > njb) jc = static_cast<unsigned>(njb);
    if (jc < 1) jc = 1;
    ic = budget / jc;
    if (ic > nib) ic = static_cast<unsigned>(nib);
    if (ic < 1) ic = 1;
    return ic * jc;
}
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
    // Note WHICH repair applies: the mc cap, NOT the search. ceil(64/6) = 11 is
    // below the 16 cache bound, giving nib = 6 and a 6x1 grid that a greedy split
    // would also have found. Both defects pointed at this shape; the cap reaches
    // it by the better route, since one wide ic team shares a single packed-B
    // panel where 3x2 would hold two. The search is exercised separately below,
    // because this case does not exercise it at all.
    const auto g = plan_gemm_grid(64, 6144, 16, 2048, MR, 6);
    REQUIRE(g.mc == 11);
    REQUIRE(g.nib == 6);
    REQUIRE(g.njb == 3);
    REQUIRE(g.ic_nt * g.jc_nt == 6);         // was 4
    REQUIRE(g.ic_nt <= g.nib);
    REQUIRE(g.jc_nt <= g.njb);
}

TEST_CASE("the exhaustive search is load-bearing, not decoration",
          "[detail][gemm][grid]") {
    // This case exists because the obvious ones do NOT test the search. Where the
    // mc cap can reach the budget it produces nib >= budget, and a greedy ic-first
    // split then lands on the same grid -- true for both the Jetson case above
    // and an earlier version of this test. Deleting the search would have left
    // every one of them green.
    //
    // The register-tile floor is what forces the issue: at m = 24 the cap wants
    // ceil(24/6) = 4, below mr, so mc stops at 6 and nib = 4 < budget. Greedy
    // then takes ic = 4, jc = 6/4 = 1 -> 4, while 3 x 2 = 6 is legal.
    const auto g = plan_gemm_grid(24, 6144, 100, 2048, MR, 6);
    REQUIRE(g.mc == MR);                                  // floored, cannot shrink further
    REQUIRE(g.nib == 4);
    REQUIRE(g.njb == 3);
    REQUIRE(greedy_grid(g.nib, g.njb, 6) == 4);           // what the old code did
    REQUIRE(g.ic_nt * g.jc_nt == 6);                      // what the search finds
    REQUIRE(g.ic_nt * g.jc_nt > greedy_grid(g.nib, g.njb, 6));
}

TEST_CASE("the planner is never worse than the greedy split it replaced",
          "[detail][gemm][grid]") {
    // Property form of the above, so the search cannot silently regress on shapes
    // no one thought to enumerate.
    for (std::size_t m : {std::size_t{16}, std::size_t{24}, std::size_t{64},
                          std::size_t{96}, std::size_t{400}, std::size_t{1024},
                          std::size_t{4096}})
        for (std::size_t n : {std::size_t{2048}, std::size_t{6144}, std::size_t{16384}})
            for (std::size_t mc : {std::size_t{16}, std::size_t{64}, std::size_t{100},
                                   std::size_t{256}})
                for (unsigned b : {2u, 6u, 8u, 12u}) {
                    const auto g = plan_gemm_grid(m, n, mc, 2048, MR, b);
                    INFO("m=" << m << " n=" << n << " mc=" << mc << " budget=" << b
                         << " -> " << g.ic_nt << "x" << g.jc_nt
                         << " vs greedy " << greedy_grid(g.nib, g.njb, b));
                    REQUIRE(g.ic_nt * g.jc_nt >= greedy_grid(g.nib, g.njb, b));
                }
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
