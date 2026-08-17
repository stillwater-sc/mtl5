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

// --- gemm_plan_for: what the nest RUNS, not what was configured -------------
//
// The blocking parameter under test travels through three stages before any
// loop steps by it: the L2 bound in blocking_params, the budget cap in
// plan_gemm_grid, and the round-off in balanced_mc. The A/B harness recorded
// only the first, so the committed data reports mc=213 for i7 runs that stepped
// 210 serially and 128 on eight threads (#430). gemm_plan_for is the single
// implementation the nest and the benchmark now share; these tests pin the
// properties that make it worth recording.
#include <mtl/simd/blocking.hpp>
#include <mtl/detail/thread_pool.hpp>

TEST_CASE("gemm_plan_for reports a derived mc, not the configured one",
          "[detail][gemm][grid][plan]") {
    const auto& bp = mtl::simd::runtime_blocking<double>();

    SECTION("serial: rounded down to whole register panels") {
        const auto p = mtl::detail::gemm_plan_for<double>(1024, 1024, 1);
        REQUIRE(p.ic_nt == 1);
        REQUIRE(p.jc_nt == 1);
        REQUIRE(p.mc >= 1);
        REQUIRE(p.mc <= bp.mc);
        // The property that makes the recorded number meaningful: it is a whole
        // number of mr-row panels, which the configured bound need not be (the
        // shipped fp64 mc=64 with mr=6 runs at 60).
        if (bp.mc >= bp.mr) REQUIRE(p.mc % bp.mr == 0);
        REQUIRE(p.nib == (1024 + p.mc - 1) / p.mc);
    }

    SECTION("threaded: never past the pool, never past the cache bound") {
        const unsigned pool = mtl::detail::thread_pool::instance().size();
        for (unsigned req : {2u, 6u, 8u, 64u}) {
            const auto p = mtl::detail::gemm_plan_for<double>(1024, 1024, req);
            INFO("requested " << req << " of a " << pool << "-thread pool -> "
                 << p.ic_nt << "x" << p.jc_nt << " mc=" << p.mc);
            REQUIRE(p.budget <= pool);
            REQUIRE(p.budget <= req);
            REQUIRE(p.ic_nt * p.jc_nt <= p.budget);
            REQUIRE(p.mc >= 1);
            REQUIRE(p.mc <= bp.mc);          // the L2 bound is never exceeded
            REQUIRE(p.nib == (1024 + p.mc - 1) / p.mc);
            REQUIRE(p.ic_nt <= p.nib);
            REQUIRE(p.jc_nt <= p.njb);
        }
    }

    SECTION("a tall problem on many threads shrinks mc rather than idling") {
        // The i7 case from #430: with the cache bound alone, m/mc yields fewer
        // blocks than threads. Whatever the pool, the plan must never report
        // more ic threads than it has blocks for.
        const auto p = mtl::detail::gemm_plan_for<double>(1024, 1024, 8);
        REQUIRE(p.ic_nt <= p.nib);
        if (p.budget >= 8) REQUIRE(p.nib >= p.ic_nt);
    }
}

// --- the C-strip bound on mc (#453) ----------------------------------------
//
// derive_blocking sizes mc so the packed A block fills ~half of L2 and counts
// nothing else, while the same cache holds the strip of C being accumulated.
// Adding the C term to the COMPILE-TIME model does not work, and the arithmetic
// says so rather than a measurement: the static strip has to be `mc x nc`, since
// n is unknown there, and reproducing the shipped mc = 64 then needs
// `mc*(kc+nc)*sizeof <= 8.5*L2` -- not a residency statement, and a different
// hardware family needs 2.5 instead. The strip is `mc x min(n, nc)`, and n is
// known at the call.
TEST_CASE("the C-strip cap bounds mc by what L2 must actually hold",
          "[detail][gemm][grid][cstrip]") {
    using mtl::detail::c_strip_mc_cap;
    constexpr std::size_t SD = 8;                 // double

    SECTION("i7-12700K: binds where the model overshot, not where it was fine") {
        constexpr std::size_t L2 = 1310720;       // 1.25 MB, one P-core
        // n <= nc: the strip is the problem width, so a narrow problem allows a
        // large mc and the shipped 64 is untouched.
        REQUIRE(c_strip_mc_cap(L2, 256, 1024, 4096, SD) == 128);
        // As n grows the strip does too, and the bound tightens below 64 --
        // which is the direction the 4096^2 measurements preferred (#430).
        REQUIRE(c_strip_mc_cap(L2, 256, 2048, 4096, SD) == 71);
        REQUIRE(c_strip_mc_cap(L2, 256, 4096, 4096, SD) == 37);
        // Beyond nc the strip stops growing: jc blocks the problem at nc.
        REQUIRE(c_strip_mc_cap(L2, 256, 8192, 4096, SD)
                == c_strip_mc_cap(L2, 256, 4096, 4096, SD));
    }

    SECTION("a larger kc tightens the bound, but weakly") {
        // The kc term is small next to the strip, so the cap is nearly
        // independent of kc -- unlike the current model, where mc is inversely
        // proportional to it. That difference in SHAPE is what the ccap arm
        // measures.
        constexpr std::size_t L2 = 1310720;
        const auto lo = c_strip_mc_cap(L2, 256, 1024, 4096, SD);
        const auto hi = c_strip_mc_cap(L2, 384, 1024, 4096, SD);
        REQUIRE(hi < lo);
        REQUIRE(hi * 10 > lo * 8);      // within ~20%, not the 1.5x of mc ~ 1/kc
    }

    SECTION("degenerate input yields no bound rather than a zero step") {
        REQUIRE(c_strip_mc_cap(0, 256, 1024, 4096, SD) == 0);      // L2 unknown
        REQUIRE(c_strip_mc_cap(1310720, 256, 1024, 4096, 0) == 0); // sizeof 0
        // A cache too small for even one row must still leave a usable step.
        REQUIRE(c_strip_mc_cap(64, 256, 4096, 4096, SD) == 1);
    }
}

TEST_CASE("plan_gemm_grid honours the extra cap and still never starves",
          "[detail][gemm][grid][cstrip]") {
    // The cap is applied BEFORE the budget cap, so both can bind and the smaller
    // wins -- and neither may push mc to 0, which is a loop step.
    for (std::size_t cap : {std::size_t{0}, std::size_t{16}, std::size_t{37},
                            std::size_t{128}, std::size_t{4096}})
        for (std::size_t m : {std::size_t{64}, std::size_t{1024}, std::size_t{4096}})
            for (unsigned b : {1u, 8u}) {
                const auto g = plan_gemm_grid(m, 4096, 213, NC, MR, b, cap);
                INFO("cap=" << cap << " m=" << m << " budget=" << b
                     << " -> mc=" << g.mc << " grid=" << g.ic_nt << "x" << g.jc_nt);
                REQUIRE(g.mc >= 1);
                REQUIRE(g.mc <= 213);                       // the L2 bound holds
                if (cap != 0 && cap < 213) REQUIRE(g.mc <= cap);
                REQUIRE(g.nib == (m + g.mc - 1) / g.mc);
                REQUIRE(g.ic_nt * g.jc_nt <= (b < 1 ? 1 : b));
            }

    // A cap looser than the cache bound changes nothing.
    const auto uncapped = plan_gemm_grid(1024, 4096, 64, NC, MR, 8, 0);
    const auto loose    = plan_gemm_grid(1024, 4096, 64, NC, MR, 8, 4096);
    REQUIRE(uncapped.mc == loose.mc);
    REQUIRE(uncapped.ic_nt == loose.ic_nt);
}
