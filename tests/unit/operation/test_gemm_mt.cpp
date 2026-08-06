// Multithreaded native GEMM (#92): the ic-loop is partitioned across a
// std::thread team (C++ concurrency runtime, no OpenMP). Different ic-blocks
// write disjoint C rows and only read the shared packed B panel, so the threaded
// result is BIT-IDENTICAL to single-thread -- each C block gets the same FMAs in
// the same order regardless of which thread runs it. We assert exact equality
// (==) between nthreads=1 and several thread counts, across sizes that span many
// ic-blocks, both operand orientations, and alpha/beta.
//
// Run under TSan (-DMTL5_SANITIZE=thread) to confirm race-freedom.
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include <mtl/detail/gemm_blocked.hpp>

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>
#include <algorithm>
#include <cmath>

namespace {

// Fill row-major (rs=cols,cs=1) or col-major (rs=1,cs=rows) storage with random
// values; return the (ptr-relative) strides for gemm_blocked.
template <typename T>
void fill_random(std::vector<T>& buf, std::size_t rows, std::size_t cols,
                 bool rowmaj, std::uint64_t seed) {
    buf.assign(rows * cols, T(0));
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (std::size_t i = 0; i < rows; ++i)
        for (std::size_t j = 0; j < cols; ++j)
            buf[rowmaj ? i * cols + j : j * rows + i] = static_cast<T>(dist(rng));
}

// gemm_blocked with nthreads == 1 vs == nt must be bit-identical.
template <typename T>
bool mt_matches(std::size_t m, std::size_t n, std::size_t k, bool a_rowmaj, bool b_rowmaj,
                unsigned nt, T alpha, T beta, std::uint64_t seed) {
    std::vector<T> A, B;
    fill_random(A, m, k, a_rowmaj, seed);
    fill_random(B, k, n, b_rowmaj, seed + 1);
    const std::ptrdiff_t a_rs = a_rowmaj ? (std::ptrdiff_t)k : 1, a_cs = a_rowmaj ? 1 : (std::ptrdiff_t)m;
    const std::ptrdiff_t b_rs = b_rowmaj ? (std::ptrdiff_t)n : 1, b_cs = b_rowmaj ? 1 : (std::ptrdiff_t)k;

    // Identical preset C for both runs (matters when beta != 0).
    std::vector<T> C0(m * n);
    { std::mt19937_64 rng(seed + 2); std::uniform_real_distribution<double> d(-1.0, 1.0);
      for (auto& c : C0) c = static_cast<T>(d(rng)); }

    std::vector<T> C1 = C0, CN = C0;
    mtl::detail::gemm_blocked<T>(m, n, k, alpha, A.data(), a_rs, a_cs, B.data(), b_rs, b_cs,
                                 beta, C1.data(), n, 1);
    mtl::detail::gemm_blocked<T>(m, n, k, alpha, A.data(), a_rs, a_cs, B.data(), b_rs, b_cs,
                                 beta, CN.data(), n, nt);
    return C1 == CN;   // bit-exact
}

} // namespace

TEMPLATE_TEST_CASE("MT GEMM == single-thread, bit-exact, many ic-blocks", "[operation][gemm][mt]", float, double) {
    // Sizes large enough in m to span several ic (mc) blocks for any build width.
    const std::size_t ms[] = {65, 128, 200, 333};
    const std::size_t ns[] = {1, 8, 40, 96};
    const std::size_t ks[] = {1, 7, 64, 129};
    std::uint64_t seed = 1;
    for (std::size_t m : ms)
        for (std::size_t n : ns)
            for (std::size_t k : ks)
                for (unsigned nt : {2u, 3u, 4u, 8u}) {
                    INFO("m=" << m << " n=" << n << " k=" << k << " nt=" << nt);
                    CHECK(mt_matches<TestType>(m, n, k, true, true, nt, TestType(1), TestType(0), seed++));
                }
}

TEMPLATE_TEST_CASE("MT GEMM == single-thread: orientations + alpha/beta", "[operation][gemm][mt]", float, double) {
    const std::size_t m = 257, n = 53, k = 71;   // many ic-blocks, odd/rectangular
    std::uint64_t s = 1000;
    const unsigned nt = 4;
    // orientation combos
    CHECK(mt_matches<TestType>(m, n, k, true,  true,  nt, TestType(1), TestType(0), s++));
    CHECK(mt_matches<TestType>(m, n, k, false, true,  nt, TestType(1), TestType(0), s++));
    CHECK(mt_matches<TestType>(m, n, k, true,  false, nt, TestType(1), TestType(0), s++));
    CHECK(mt_matches<TestType>(m, n, k, false, false, nt, TestType(1), TestType(0), s++));
    // alpha/beta
    CHECK(mt_matches<TestType>(m, n, k, true, true, nt, TestType(2.5),  TestType(0),    s++));
    CHECK(mt_matches<TestType>(m, n, k, true, true, nt, TestType(1),    TestType(-1.5), s++));
    CHECK(mt_matches<TestType>(m, n, k, true, true, nt, TestType(-0.5), TestType(0.75), s++));
}

TEMPLATE_TEST_CASE("MT GEMM: crosses mc/kc blocks", "[operation][gemm][mt]", float, double) {
    constexpr auto bp = mtl::simd::default_blocking<TestType>;
    const std::size_t m = bp.mc * 3 + bp.mr + 1;   // several mc blocks across threads
    const std::size_t k = bp.kc + 5;               // > kc -> multiple pc iterations
    const std::size_t n = bp.nr * 2 + 1;
    INFO("m=" << m << " n=" << n << " k=" << k << " (mc=" << bp.mc << " kc=" << bp.kc << ")");
    CHECK(mt_matches<TestType>(m, n, k, true, true, 4, TestType(1), TestType(0), 7));
    CHECK(mt_matches<TestType>(m, n, k, true, true, 8, TestType(1.25), TestType(-0.5), 9));
}

// #297 batch 9: BLIS multi-loop (2D jc_nt x ic_nt grid). A wide/short matrix has
// few ic (mc) blocks, so the ic-only parallelization can't use all threads --
// the jc loop must be partitioned too. Bit-identity must hold for ANY grid shape.
TEMPLATE_TEST_CASE("MT GEMM 2D grid: wide/short spans few ic-blocks -> jc-parallel",
                   "[operation][gemm][mt]", float, double) {
    constexpr auto bp = mtl::simd::default_blocking<TestType>;
    // m spans a single ic-block (nib == 1): with ic-only this would run 1 thread;
    // the 2D grid must fall to pure jc-parallelism. n crosses nc (njb == 2) so the
    // jc loop actually partitions (nc is L3-sized, so one crossing is plenty).
    const std::size_t m = bp.mr + 1;               // < mc -> one ic-block
    const std::size_t n = bp.nc + bp.nr + 1;       // 2 nc (jc) blocks
    const std::size_t k = bp.kc + 3;               // multiple pc iterations
    INFO("m=" << m << " n=" << n << " k=" << k << " (mc=" << bp.mc << " nc=" << bp.nc << ")");
    // gemm_blocked caps the grid to the pool size, so a pool < 2 would exercise
    // only the serial fallback -- flag that the jc-parallel path went untested.
    if (mtl::detail::thread_pool::instance().size() < 2)
        WARN("pool < 2 workers: jc-parallel path not exercised (set MTL5_NUM_THREADS)");
    for (unsigned nt : {2u, 4u}) {
        CHECK(mt_matches<TestType>(m, n, k, true,  true,  nt, TestType(1),    TestType(0),   100 + nt));
        CHECK(mt_matches<TestType>(m, n, k, false, true,  nt, TestType(1.5),  TestType(-0.5),200 + nt));
    }
}

// A problem with TWO ic-blocks AND TWO jc-blocks forces a genuine 2D grid
// (ic_nt > 1 AND jc_nt > 1 when the budget is >= 4), exercising the per-team
// barrier and the shared packed-B panel with multi-member teams. Kept small: nc
// is L3-sized, so just crossing it once (njb == 2) is already a wide panel;
// nib == 2 keeps ic_nt <= 2 so the budget spills into jc_nt. Bit-identity vs
// single-thread must hold for the resulting grid.
TEMPLATE_TEST_CASE("MT GEMM 2D grid: rectangular exercises ic_nt>1 && jc_nt>1",
                   "[operation][gemm][mt]", float, double) {
    constexpr auto bp = mtl::simd::default_blocking<TestType>;
    const std::size_t m = bp.mc + bp.mr + 1;   // 2 ic-blocks (nib == 2)
    const std::size_t n = bp.nc + bp.nr + 1;   // 2 jc-blocks (njb == 2)
    const std::size_t k = bp.kc + 7;           // 2 pc iterations (leader repacks B)
    INFO("m=" << m << " n=" << n << " k=" << k << " (mc=" << bp.mc << " nc=" << bp.nc << ")");
    // A genuine 2x2 grid needs >= 4 pool workers; below that the grid degenerates
    // (1D or serial) and the multi-member-team barrier path goes untested.
    if (mtl::detail::thread_pool::instance().size() < 4)
        WARN("pool < 4 workers: 2x2 grid not formed (set MTL5_NUM_THREADS>=4)");
    for (unsigned nt : {4u, 8u}) {   // budget>=4 -> 2x2 grid; pool caps to its size
        CHECK(mt_matches<TestType>(m, n, k, true,  true,  nt, TestType(1),    TestType(0),    300 + nt));
        CHECK(mt_matches<TestType>(m, n, k, true,  false, nt, TestType(-0.75),TestType(1.25), 400 + nt));
    }
}

// ---------------------------------------------------------------------------
// balanced_mc: the m-partition must divide evenly across the team (#408)
//
// mc used to be rounded down to a multiple of mr in derive_blocking, coupling a
// CACHE quantity to the REGISTER TILE. #382 moved mr 4 -> 6, which moved mc
// 64 -> 60, which moved the ic-block count at m=1024 from 16 (2.00 blocks per
// thread on 8 threads) to 18 (2.25) -- a 1.41x critical path that turned a
// +21.5% single-thread win into a -7.4% eight-thread REGRESSION.
//
// These assert the property that prevents a recurrence, not the specific
// constants of one machine: whatever mc the cache yields, the resulting block
// count divides evenly across the team and mc never exceeds the cache bound.

namespace {
/// Rows given to the most-loaded thread under the round-robin ic assignment
/// (`for ib = ic_id; ib < nib; ib += ic_nt`) -- i.e. the critical path.
std::size_t critical_rows(std::size_t m, std::size_t mc, unsigned nt) {
    const std::size_t nib = (m + mc - 1) / mc;
    std::vector<std::size_t> rows(nt, 0);
    for (std::size_t ib = 0; ib < nib; ++ib) {
        const std::size_t beg = ib * mc, end = std::min(m, beg + mc);
        rows[ib % nt] += end - beg;
    }
    return *std::max_element(rows.begin(), rows.end());
}
}  // namespace

TEST_CASE("balanced_mc keeps the ic partition even (#408)", "[operation][gemm][mt][regression]") {
    using mtl::detail::balanced_mc;

    SECTION("serial is untouched -- nothing to balance, biggest legal block wins") {
        CHECK(balanced_mc(1024, 64, 1) == 64);
        CHECK(balanced_mc(1024, 64, 0) == 64);
    }

    SECTION("degenerate inputs return the bound rather than dividing by zero") {
        CHECK(balanced_mc(0, 64, 8) == 64);
        CHECK(balanced_mc(1024, 0, 8) == 0);
    }

    SECTION("the exact case #382 regressed") {
        // mc_max = 60 (what the old mr-rounding produced) must still balance.
        CHECK(critical_rows(1024, 60, 8) == 180);                    // 1.41x ideal 128
        const std::size_t mc = balanced_mc(1024, 60, 8);
        INFO("balanced mc = " << mc << ", critical = " << critical_rows(1024, mc, 8));
        CHECK(mc <= 60);
        CHECK(critical_rows(1024, mc, 8) < 180);                     // strictly better
        CHECK(critical_rows(1024, mc, 8) <= 1024 / 8 + mc);          // within one block of ideal
    }

    SECTION("never exceeds the cache bound, and stays within a block of ideal") {
        for (std::size_t m : {512u, 777u, 1000u, 1024u, 1500u, 2048u, 3000u, 4096u})
            for (std::size_t mc_max : {48u, 60u, 64u, 96u})
                for (unsigned nt : {2u, 4u, 8u, 16u}) {
                    const std::size_t mc = balanced_mc(m, mc_max, nt);
                    INFO("m=" << m << " mc_max=" << mc_max << " nt=" << nt << " -> mc=" << mc);
                    REQUIRE(mc >= 1);
                    REQUIRE(mc <= mc_max);                  // still cache-legal
                    // The point of the exercise: the critical path is within one
                    // block of a perfectly even split. The old rounding could be
                    // 41% over, which is what -7.4% at 8 threads looked like.
                    REQUIRE(critical_rows(m, mc, nt) <= m / nt + mc);
                }
    }
}

// Multi-block m, SERIAL path (#408 regression).
//
// The #408 change introduced MC_eff (the ic-block size actually used, which may
// be below the cache bound MC). An early revision left serial_nest stepping
// `ic += MC` while do_ic_block sized its block as min(MC_eff, m - ic) -- so each
// block computed MC_eff rows but the loop advanced MC, silently dropping
// (MC - MC_eff) rows of C per block.
//
// The whole existing GEMM suite passed. Every case used an m that fits inside a
// single ic-block, where min(MC_eff, m) == m and the step never matters. Spanning
// several blocks is what makes the step observable, so that is what this asserts
// -- and it checks the SERIAL path specifically, since the threaded nest rebuilds
// its block-start list and was never affected.
TEMPLATE_TEST_CASE("serial GEMM spans multiple ic-blocks correctly (#408)",
                   "[operation][gemm][regression]", float, double) {
    constexpr auto bp = mtl::simd::default_blocking<TestType>;
    // >= 4 ic-blocks, and deliberately NOT a multiple of mc, so both the
    // interior blocks and the ragged final one are exercised.
    const std::size_t m = bp.mc * 4 + bp.mr + 1;
    const std::size_t n = 40, k = 24;
    INFO("m=" << m << " (mc=" << bp.mc << ", " << (m + bp.mc - 1) / bp.mc << " ic-blocks)");

    std::vector<TestType> A(m * k), B(k * n), C(m * n, TestType(0)), Cref(m * n, TestType(0));
    std::mt19937 gen(4080);
    std::uniform_real_distribution<double> d(-1.0, 1.0);
    for (auto& v : A) v = static_cast<TestType>(d(gen));
    for (auto& v : B) v = static_cast<TestType>(d(gen));

    // Triple-loop reference, independent of the blocked nest.
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t p = 0; p < k; ++p) {
            const TestType a = A[i * k + p];
            for (std::size_t j = 0; j < n; ++j) Cref[i * n + j] += a * B[p * n + j];
        }

    mtl::detail::gemm_blocked<TestType>(m, n, k, TestType(1),
                                        A.data(), static_cast<std::ptrdiff_t>(k), 1,
                                        B.data(), static_cast<std::ptrdiff_t>(n), 1,
                                        TestType(0), C.data(), static_cast<std::ptrdiff_t>(n),
                                        /*nthreads=*/1);

    // A dropped block leaves whole ROWS of C at zero, so report the first bad
    // row rather than 40 000 individual mismatches.
    std::size_t bad_row = m;
    for (std::size_t i = 0; i < m && bad_row == m; ++i)
        for (std::size_t j = 0; j < n; ++j)
            if (!(std::abs(static_cast<double>(C[i * n + j] - Cref[i * n + j]))
                  <= 1e-4 * (1.0 + std::abs(static_cast<double>(Cref[i * n + j]))))) {
                bad_row = i; break;
            }
    INFO("first mismatching row = " << bad_row);
    REQUIRE(bad_row == m);
}
