// `balanced_mc` must never return a block shorter than the register tile (#488).
//
// `plan_gemm_grid` floors `mc` at `mr` and says why: "a block shorter than mr
// would waste the micro-kernel". `balanced_mc` then ran and could lower it back
// through that floor, because balancing works by RAISING the block count and
// that lowers `mc`:
//
//     balanced_mc(m=18, mc_max=6, ic_nt=2, mr=6) = 5
//
// nib = ceil(18/6) = 3, rounded up to a multiple of 2 -> 4, mc = ceil(18/4) = 5.
//
// WHY IT COSTS MORE THAN A RAGGED LAST BLOCK. The nest sizes each ic block as
// `min(MC_eff, m - ic)` and the micro-kernel writes whole `mr x nr` tiles;
// anything shorter goes through the edge path -- a zeroed temporary, a copy in
// and a copy out. At `mc = 5` against `mr = 6` EVERY block is 5 rows, so every
// block takes that path, not just the trailing one it was written for.
//
// THE BALANCE IT BOUGHT WAS AN ILLUSION, which is the real argument for this
// change and is stronger than "sub-tile is wasteful". Exhaustively over 383934
// parameter combinations where the answer moves, the padded rows on the busiest
// thread -- what the micro-kernel is actually asked to compute -- are IDENTICAL
// before and after. The old rule looked better balanced only in raw rows, and
// the padding erased exactly that difference. What it did change was the total:
// the new answer does less work across all threads in 94.9% of those cases,
// 2.28x less on average and never more.
//
// So this is not a trade of balance for tiles. It is the same critical path with
// fewer edge blocks and less redundant work.
//
// Measured cost of the defect, from the #479 nc-model timing runs (i7-12700K and
// Ryzen 9 8945HS, both dtypes, m=18, T=8): where a re-plan moved `ic_nt` 3 -> 2
// and `mc` 6 -> 5, throughput fell to x0.818-0.950 against a 4.3-5.5% noise
// floor. Those arms also moved `jc_nt` and the packed-B working set, so they do
// not attribute the loss to `mc` alone; the sub-tile block is the part that is a
// defect rather than a trade.
#include <catch2/catch_test_macros.hpp>

#include <mtl/detail/gemm_blocked.hpp>

#include <algorithm>
#include <cstddef>
#include <vector>

using mtl::detail::balanced_mc;

namespace {

/// Rows the busiest thread is actually asked to compute: the micro-kernel writes
/// whole `mr x nr` tiles, so a block of `b` rows costs `ceil(b/mr)*mr`.
std::size_t padded_critical(std::size_t m, std::size_t mc, unsigned ic_nt,
                            std::size_t mr) {
    const std::size_t nib = (m + mc - 1) / mc;
    std::vector<std::size_t> load(ic_nt, 0);
    for (std::size_t b = 0; b < nib; ++b) {
        const std::size_t sz = std::min(mc, m - b * mc);
        load[b % ic_nt] += ((sz + mr - 1) / mr) * mr;
    }
    return *std::max_element(load.begin(), load.end());
}

/// The pre-#488 rule, kept verbatim so the tests below compare against what
/// actually shipped rather than against a paraphrase of it.
std::size_t legacy_balanced_mc(std::size_t m, std::size_t mc_max, unsigned ic_nt,
                               std::size_t mr) {
    if (ic_nt <= 1 || m == 0 || mc_max == 0) {
        if (mr > 1 && mc_max >= mr) return (mc_max / mr) * mr;
        return mc_max;
    }
    std::size_t nib = (m + mc_max - 1) / mc_max;
    nib = ((nib + ic_nt - 1) / ic_nt) * ic_nt;
    if (nib == 0) return mc_max;
    const std::size_t mc = (m + nib - 1) / nib;
    return mc == 0 ? mc_max : mc;
}

} // namespace

TEST_CASE("the #488 reproduction no longer goes sub-tile", "[detail][gemm][mc]") {
    // m = 18, mr = 6: three whole tiles. No ic_nt may split it below one.
    CHECK(legacy_balanced_mc(18, 6, 2, 6) == 5);      // what shipped, for contrast
    for (unsigned ic : {1u, 2u, 3u, 4u, 6u, 8u}) {
        INFO("ic_nt = " << ic);
        CHECK(balanced_mc(18, 6, ic, 6) >= 6);
    }
}

TEST_CASE("balanced_mc never returns a sub-tile block it could avoid",
          "[detail][gemm][mc]") {
    // Sub-tile is unavoidable only when the problem or the cache bound is itself
    // smaller than one tile; everywhere else it is a defect.
    std::size_t violations = 0;
    for (std::size_t m = 1; m <= 200; ++m)
        for (std::size_t mx = 1; mx <= 96; ++mx)
            for (unsigned ic = 1; ic <= 16; ++ic)
                for (std::size_t mr : {std::size_t{2}, std::size_t{4}, std::size_t{6},
                                       std::size_t{8}, std::size_t{16}}) {
                    const std::size_t mc = balanced_mc(m, mx, ic, mr);
                    if (m >= mr && mx >= mr && mc < mr) {
                        if (violations++ == 0)
                            INFO("first: m=" << m << " mc_max=" << mx
                                             << " ic_nt=" << ic << " mr=" << mr
                                             << " -> " << mc);
                        CHECK(mc >= mr);
                    }
                }
    CHECK(violations == 0);
}

TEST_CASE("balanced_mc stays within the cache bound and still covers m",
          "[detail][gemm][mc]") {
    // The two invariants the nest depends on. `mc` is a loop step: 0 would not
    // terminate, and a block list that does not span `m` silently drops rows
    // from C -- the failure the MC_eff comment in gemm_blocked warns about.
    for (std::size_t m = 1; m <= 200; ++m)
        for (std::size_t mx = 1; mx <= 96; ++mx)
            for (unsigned ic = 1; ic <= 16; ++ic)
                for (std::size_t mr : {std::size_t{1}, std::size_t{6}, std::size_t{16}}) {
                    const std::size_t mc = balanced_mc(m, mx, ic, mr);
                    REQUIRE(mc > 0);
                    REQUIRE(mc <= mx);
                    REQUIRE(((m + mc - 1) / mc) * mc >= m);
                }
}

TEST_CASE("the fix never lengthens the padded critical path",
          "[detail][gemm][mc]") {
    // The claim that makes this a fix rather than a trade. Where the answer
    // moved, the work the busiest thread is actually asked for is unchanged --
    // the old rule's extra "balance" was in rows the micro-kernel padded away.
    std::size_t moved = 0, worse = 0;
    for (std::size_t m = 1; m <= 200; ++m)
        for (std::size_t mx = 1; mx <= 96; ++mx)
            for (unsigned ic = 2; ic <= 16; ++ic)
                for (std::size_t mr : {std::size_t{2}, std::size_t{4}, std::size_t{6},
                                       std::size_t{8}, std::size_t{16}}) {
                    const std::size_t now = balanced_mc(m, mx, ic, mr);
                    const std::size_t was = legacy_balanced_mc(m, mx, ic, mr);
                    if (now == was) continue;
                    ++moved;
                    const std::size_t pw = padded_critical(m, was, ic, mr);
                    const std::size_t pn = padded_critical(m, now, ic, mr);
                    if (pn > pw) {
                        ++worse;
                        INFO("m=" << m << " mc_max=" << mx << " ic_nt=" << ic
                                  << " mr=" << mr << ": " << was << " -> " << now
                                  << " padded critical " << pw << " -> " << pn);
                        CHECK(pn <= pw);
                    }
                }
    CHECK(moved > 0);        // the comparison must not be vacuous
    CHECK(worse == 0);
}

TEST_CASE("balanced_mc still balances where a whole tile allows it",
          "[detail][gemm][mc]") {
    // #408's own shape must be untouched: mc 64 at m = 1024 on 8 threads gives
    // nib = 16, exactly 2.00 blocks per thread. If this moves, the fix has
    // damaged the repair it sits inside.
    CHECK(balanced_mc(1024, 64, 8, 6) == 64);
    CHECK(balanced_mc(1024, 64, 8, 6) == legacy_balanced_mc(1024, 64, 8, 6));

    // And a case where balancing genuinely lowers mc while staying tile-legal:
    // it must still do so rather than refusing every reduction.
    const std::size_t mc = balanced_mc(1000, 64, 8, 6);
    CHECK(mc <= 64);
    CHECK(mc >= 6);
    CHECK((1000 + mc - 1) / mc % 8 == 0);   // still an even partition
}

TEST_CASE("the partition does not collapse onto one thread",
          "[detail][gemm][mc]") {
    // The defect the first draft of this fix introduced: minimising BLOCK COUNT
    // rather than rows let m=16, ic_nt=16, mr=2 choose a single 16-row block and
    // idle 15 threads -- 16x worse than the sub-tile answer it replaced.
    CHECK(balanced_mc(16, 16, 16, 2) == 2);

    // Generally: where m affords at least ic_nt whole tiles, the block count
    // must reach the team.
    for (unsigned ic : {2u, 4u, 8u}) {
        const std::size_t mr = 6, m = mr * ic * 2, mx = m;
        const std::size_t mc = balanced_mc(m, mx, ic, mr);
        INFO("ic_nt=" << ic << " m=" << m << " -> mc=" << mc);
        CHECK((m + mc - 1) / mc >= ic);
    }
}
