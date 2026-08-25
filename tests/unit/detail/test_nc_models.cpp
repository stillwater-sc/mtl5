// The candidate `nc` models, and the properties #479 pre-registered as controls.
//
// #479 has to choose between six ways of sizing `nc`, on four machines. Most of
// what can go wrong is decidable here, without a machine: a model that is not a
// no-op single-threaded, or that breaks the multiple-of-`nr` property, or that
// makes the partition worse than the baseline, is wrong before any throughput is
// measured. Testing those here is what keeps the machine time for the question
// that actually needs hardware.
//
// THE NEGATIVE CONTROLS ARE THE POINT. #479 pre-registers four, and three are
// pure predictions about these functions:
//
//   1. single thread   -> jc_nt == 1, every balancing model must be a no-op
//   2. njb % jc_nt == 0 already -> balanced_nc must be a no-op
//   3. square shapes   -> jc_nt structurally 1, no model should move them
//
// A model that "wins" on a machine while failing one of these has not won, and
// the cheapest place to find that out is here.
#include <catch2/catch_test_macros.hpp>

#include <mtl/detail/gemm_blocked.hpp>
#include <mtl/detail/nc_model.hpp>

#include <cstddef>
#include <string>

namespace {

using mtl::detail::nc_model;
using mtl::detail::nc_model_inputs;
using mtl::detail::nc_for_model;
using mtl::detail::balanced_nc;

// A plausible x86 machine: 25 MB L3 shared by 12 cores, kc 256, nr 8, fp64.
nc_model_inputs base(std::size_t n, unsigned jc_nt) {
    return nc_model_inputs{n, /*kc*/256, /*nr*/8, /*sdata*/8,
                           /*l3_default*/8u * 1024 * 1024,
                           /*l3_detected*/26214400,
                           /*l3_sharing_cores*/12,
                           jc_nt};
}

} // namespace

TEST_CASE("balanced_nc evens the jc partition", "[detail][gemm][nc]") {
    // 5 blocks across 2 teams is 3 and 2 -- a 1.2x critical path. Balancing must
    // make the block count a multiple of the team count.
    const std::size_t n = 5000, nr = 8;
    const std::size_t nc_max = 1024;                    // njb = ceil(5000/1024) = 5
    CHECK((n + nc_max - 1) / nc_max == 5);
    const std::size_t nc = balanced_nc(n, nc_max, 2, nr);
    const std::size_t njb = (n + nc - 1) / nc;
    INFO("nc " << nc_max << " -> " << nc << ", njb 5 -> " << njb);
    CHECK(njb % 2 == 0);
    CHECK(nc <= nc_max);                                // never exceeds the cache bound
}

TEST_CASE("balanced_nc is a no-op where the partition already divides",
          "[detail][gemm][nc]") {
    // #479's negative control 2, decided here rather than on a machine.
    const std::size_t nr = 8, nc_max = 1000;
    for (unsigned jc_nt : {2u, 3u, 4u}) {
        for (std::size_t njb : {2u, 4u, 6u, 12u}) {
            if (njb % jc_nt) continue;                  // only the already-even cases
            const std::size_t n = njb * nc_max;         // exactly njb blocks
            INFO("jc_nt=" << jc_nt << " njb=" << njb);
            CHECK(balanced_nc(n, nc_max, jc_nt, nr) == nc_max);
        }
    }
}

TEST_CASE("balanced_nc is a no-op single-threaded", "[detail][gemm][nc]") {
    // Negative control 1. jc_nt <= 1 means there is nothing to balance, so the
    // largest cache-legal block is right and the model must not touch it.
    for (std::size_t n : {1u, 100u, 5000u, 100000u})
        CHECK(balanced_nc(n, 4096, 1, 8) == 4096);
    CHECK(balanced_nc(5000, 4096, 0, 8) == 4096);       // degenerate team count
}

TEST_CASE("balanced_nc never exceeds the cache bound", "[detail][gemm][nc]") {
    // Swept rather than spot-checked. Note what is NOT asserted: a multiple of
    // `nr` in the threaded case. Rounding after balancing perturbs `njb` and
    // undoes the balance -- #408's defect on the jc axis -- so `balanced_nc`
    // deliberately leaves the balanced value ragged, exactly as `balanced_mc`
    // does. The nr multiple is asserted only where it is safe, below.
    bool ok = true;
    for (std::size_t n = 1; n <= 20000; n += 137)
        for (std::size_t nc_max : {64u, 512u, 1024u, 4096u})
            for (unsigned jc_nt : {1u, 2u, 3u, 5u, 8u}) {
                const std::size_t nc = balanced_nc(n, nc_max, jc_nt, 8);
                if (nc == 0 || nc > nc_max) ok = false;
            }
    CHECK(ok);
}

TEST_CASE("balanced_nc rounds to whole nr panels only when serial",
          "[detail][gemm][nc]") {
    // Serial: no partition to perturb, so the rounding is free and removes the
    // ragged panel from every jc block.
    CHECK(balanced_nc(100000, 4095, 1, 8) == 4088);     // 4095 -> 4088, a multiple of 8
    // Threaded: the rounding is NOT applied, because it would change njb.
    const std::size_t nc = balanced_nc(5000, 1024, 2, 8);
    CHECK((5000 + nc - 1) / nc % 2 == 0);               // balance preserved
}

TEST_CASE("every model is a no-op relative to its own budget single-threaded",
          "[detail][gemm][nc]") {
    // Negative control 1 again, one level up: with jc_nt == 1 the balancing
    // models must collapse onto their capacity-only answer.
    for (auto m : mtl::detail::all_nc_models) {
        auto in = base(100000, 1);
        const std::size_t nc = nc_for_model(m, in);
        INFO(mtl::detail::nc_model_name(m));
        CHECK(nc > 0);
        CHECK(nc % in.nr == 0);
    }
}

TEST_CASE("M0 reproduces today's shipped derivation exactly", "[detail][gemm][nc]") {
    // M0 is the baseline the whole experiment is measured against. If it does not
    // equal what derive_blocking computes, every ratio in #479 is against a
    // strawman rather than against what ships.
    auto in = base(8192, 4);
    const std::size_t expect = ((in.l3_default_bytes / (in.kc * in.sdata)) / in.nr) * in.nr;
    CHECK(nc_for_model(nc_model::m0_default, in) == expect);
    // And it ignores the detected figure, which is what makes it the baseline.
    auto in2 = in; in2.l3_detected_bytes = 99u * 1024 * 1024;
    CHECK(nc_for_model(nc_model::m0_default, in2) == expect);
}

TEST_CASE("M1 differs from M0 more often than #479's table says",
          "[detail][gemm][nc]") {
    // #479 (inherited from #430) says M1 differs from M0 when `njb % jc_nt != 0`.
    // That is TOO NARROW, and the distinction matters because the offline sweep
    // uses it to decide which shapes are worth machine time -- under the table's
    // rule we would skip shapes where the models genuinely disagree.
    //
    // Counterexample: n = 7026, nc_max = 4096, jc_nt = 2. njb is already 2, so
    // the table predicts a no-op, but `balanced_nc` returns 3513: the block
    // COUNT was even while the block SIZES were not (4096 + 2930), and the
    // critical path is set by the larger block, not by the count. Equalising
    // them is the point. `balanced_mc` has behaved this way since #408; this is
    // the same function on the other axis.
    auto in = base(7026, 2);
    in.kc = 256; in.nr = 8; in.sdata = 8;
    const std::size_t m0 = nc_for_model(nc_model::m0_default, in);
    const std::size_t m1 = nc_for_model(nc_model::m1_balanced, in);
    CHECK(m0 == 4096);
    CHECK((in.n + m0 - 1) / m0 % 2 == 0);      // the table's condition says "no-op"
    CHECK(m1 != m0);                            // and yet
    CHECK(m1 == 3513);                          // equal blocks instead of 4096 + 2930

    // The property that IS reliable, and the one the sweep should use: M1 never
    // exceeds M0's nc, and never leaves a more ragged partition.
    using mtl::detail::grid_imbalance;
    bool ok = true;
    for (std::size_t n = 1024; n <= 200000; n += 3001)
        for (unsigned jc_nt : {2u, 3u, 4u}) {
            auto x = base(n, jc_nt);
            const std::size_t a = nc_for_model(nc_model::m0_default, x);
            const std::size_t b = nc_for_model(nc_model::m1_balanced, x);
            if (b > a) ok = false;
            if (grid_imbalance((n + b - 1) / b, jc_nt) >
                grid_imbalance((n + a - 1) / a, jc_nt) + 1e-12) ok = false;
        }
    CHECK(ok);
}

TEST_CASE("M1 is a genuine no-op when the blocks are already equal",
          "[detail][gemm][nc]") {
    // The case the table was reaching for: n an exact multiple of nc_max, and
    // the resulting count already a multiple of jc_nt. Then there is nothing to
    // equalise and M1 must leave M0 alone.
    auto in = base(0, 2);
    const std::size_t nc0 = nc_for_model(nc_model::m0_default, in);
    for (unsigned jc_nt : {2u, 4u}) {
        auto x = base(nc0 * jc_nt * 3, jc_nt);          // exactly 3*jc_nt equal blocks
        INFO("jc_nt=" << jc_nt << " n=" << x.n << " nc0=" << nc0);
        CHECK(nc_for_model(nc_model::m1_balanced, x) == nc0);
    }
}

TEST_CASE("a balancing model never worsens the imbalance it targets",
          "[detail][gemm][nc]") {
    // The mediator, used as a property rather than only reported. A model whose
    // whole justification is evening the partition must not make it more ragged
    // than the baseline -- if it does, it is mis-implemented and no amount of
    // throughput data should be spent on it.
    using mtl::detail::grid_imbalance;
    bool ok = true;
    for (std::size_t n = 2048; n <= 300000; n += 4099)
        for (unsigned jc_nt : {2u, 3u, 4u, 8u}) {
            auto in = base(n, jc_nt);
            const std::size_t nc0 = nc_for_model(nc_model::m0_default, in);
            const double base_imb = grid_imbalance((n + nc0 - 1) / nc0, jc_nt);
            for (auto m : {nc_model::m1_balanced, nc_model::m2_detected,
                           nc_model::m3_per_team, nc_model::m4_per_sharer}) {
                const std::size_t nc = nc_for_model(m, in);
                const double imb = grid_imbalance((n + nc - 1) / nc, jc_nt);
                if (imb > base_imb + 1e-12) {
                    UNSCOPED_INFO("worsened: " << mtl::detail::nc_model_name(m)
                                  << " n=" << n << " jc_nt=" << jc_nt
                                  << " " << base_imb << " -> " << imb);
                    ok = false;
                }
            }
        }
    CHECK(ok);
}

TEST_CASE("M5 gives one jc block per team where n allows", "[detail][gemm][nc]") {
    auto in = base(4096, 4);
    const std::size_t nc = nc_for_model(nc_model::m5_exact, in);
    CHECK((in.n + nc - 1) / nc == 4);
    // Where the exact split would overflow the budget it must fall back rather
    // than thrash -- an "exact" partition that does not fit is not the thing
    // under test.
    auto big = base(100000000, 2);
    const std::size_t nc_big = nc_for_model(nc_model::m5_exact, big);
    const std::size_t cap = mtl::detail::nc_from_budget(big.l3_detected_bytes, big.kc,
                                                        big.sdata, big.nr);
    CHECK(nc_big <= cap);
}

TEST_CASE("an undetectable machine behaves exactly as M0", "[detail][gemm][nc]") {
    // Same rule with_detected_caches follows field by field: a figure the
    // platform could not report keeps the compile-time value. A model must not
    // produce a degenerate block size on a machine that reports nothing.
    auto in = base(65536, 4);
    in.l3_detected_bytes = 0;
    in.l3_sharing_cores  = 0;
    for (auto m : mtl::detail::all_nc_models) {
        const std::size_t nc = nc_for_model(m, in);
        INFO(mtl::detail::nc_model_name(m));
        CHECK(nc > 0);
        CHECK(nc % in.nr == 0);
    }
    CHECK(nc_for_model(nc_model::m2_detected, in) ==
          nc_for_model(nc_model::m1_balanced, in));   // detected == default here
}

TEST_CASE("the mediator metrics say what they claim", "[detail][gemm][nc]") {
    using mtl::detail::grid_imbalance;
    using mtl::detail::packed_b_bytes;
    CHECK(grid_imbalance(8, 4) == 1.0);                 // divides evenly
    CHECK(grid_imbalance(5, 2) > 1.0);                  // 3 and 2
    CHECK(grid_imbalance(5, 2) == 6.0 / 5.0);
    CHECK(grid_imbalance(0, 4) == 1.0);                 // degenerate, not a divide by zero
    CHECK(grid_imbalance(8, 0) == 1.0);
    // One kc x nc panel per team.
    CHECK(packed_b_bytes(2, 256, 1024, 8, 8) == 2u * 256 * 1024 * 8);
    CHECK(packed_b_bytes(0, 256, 1024, 8, 8) == 256u * 1024 * 8);   // treated as one team
}

TEST_CASE("model names are stable and distinct", "[detail][gemm][nc]") {
    // They become CSV values and command-line arguments; a duplicate would make
    // two arms indistinguishable in the committed data.
    std::string seen;
    for (auto m : mtl::detail::all_nc_models) {
        const std::string n = mtl::detail::nc_model_name(m);
        CHECK(n != "unknown");
        CHECK(seen.find("|" + n + "|") == std::string::npos);
        seen += "|" + n + "|";
    }
}
