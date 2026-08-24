// M6: the guarded `balanced_nc` that #429 ships, and the data that justifies it.
//
// #479 timed six candidate `nc` models on four microarchitectures and two
// dtypes. M1 -- plain `balanced_nc`, which is what #429 originally proposed --
// wins on 42 of 44 arms and loses ~18% on two, both on a Zen 4. M6 is M1 with
// one guard, and the guard is the mechanism rather than a curve fit:
//
//   `balanced_nc` lowers `nc`, which raises `njb`, which can let
//   `plan_gemm_grid` re-factor the thread grid and take `jc_nt` 2 -> 4. Each
//   team holds its own `kc x nc` packed-B panel, so the resident set moves
//   16 MB -> 24 MB. An i7-12700K (25 MB L3) absorbs that and GAINS x1.02; a
//   Ryzen 9 8945HS (16 MB) does not and loses to x0.82.
//
// THE MACHINE TABLE BELOW IS THE POINT OF THIS FILE. A guard chosen from
// measurement can drift away from that measurement silently -- the code still
// compiles, the tests still pass, and nothing says the rule no longer does what
// the data said. So the four machines' recorded blocking parameters are pinned
// here, straight from their committed `.sysinfo` sidecars, and the guard is
// replayed against them. It must decline exactly two arms, and they must be the
// two that actually regressed.
//
// This is host-independent: it uses the recorded parameters, not this machine's.
#include <catch2/catch_test_macros.hpp>

#include <mtl/detail/gemm_blocked.hpp>
#include <mtl/detail/nc_model.hpp>
#include <mtl/simd/blocking.hpp>

#include <cstddef>
#include <string>
#include <vector>

using mtl::detail::all_nc_models;
using mtl::detail::gemm_grid;
using mtl::detail::nc_from_budget;
using mtl::detail::nc_guard_declines;
using mtl::detail::nc_model;
using mtl::detail::nc_model_inputs;
using mtl::detail::nc_model_name;
using mtl::detail::nc_model_selection;
using mtl::detail::packed_b_bytes;
using mtl::detail::plan_gemm_grid;

namespace {

/// One machine+dtype exactly as its committed sidecar reports it (#479).
struct recorded {
    const char* machine;
    const char* dtype;
    std::size_t mr, nr, kc, mc, nc;   // blocking the run compiled to
    std::size_t sdata;
    std::size_t l3_detected, l3_sharers;
    unsigned    threads;
};

// From benchmarks/data/*/nc_model_timing_*.csv.sysinfo. The Xeon has no float
// session; everything else ran both.
constexpr recorded MACHINES[] = {
    {"xeon-e5-2420",       "double", 6,  4, 512,  32,  2048, 8, 15728640,  6, 6},
    {"i7-12700k",          "double", 6,  8, 256,  64,  4096, 8, 26214400, 12, 8},
    {"i7-12700k",          "float",  6, 16, 256, 128,  8192, 4, 26214400, 12, 8},
    {"ryzen-9-8945hs",     "double", 6, 16, 128, 128,  8192, 8, 16777216,  8, 8},
    {"ryzen-9-8945hs",     "float",  6, 32, 128, 256, 16384, 4, 16777216,  8, 8},
    {"jetson-orin-nano",   "double", 6,  4, 512,  32,  2048, 8,  2097152,  4, 6},
    {"jetson-orin-nano",   "float",  6,  8, 512,  64,  4096, 4,  2097152,  4, 6},
};

/// The compile-time L3 the models take as their baseline budget.
constexpr std::size_t L3_DEFAULT = 8388608;

struct verdict { std::size_t nc_m0, nc_m1; bool declined; };

/// `nc_for_plan`'s M6 path, expressed against recorded parameters instead of
/// this host's. Kept in step with the implementation by the shared helpers it
/// calls -- `plan_gemm_grid`, `nc_for_model`, `packed_b_bytes`,
/// `nc_guard_declines` -- so a change to any of them shows up here.
verdict replay(const recorded& r, std::size_t m, std::size_t n) {
    const gemm_grid g0 = plan_gemm_grid(m, n, r.mc, r.nc, r.mr, r.threads);
    const nc_model_inputs in{n, r.kc, r.nr, r.sdata, L3_DEFAULT,
                             r.l3_detected, r.l3_sharers, g0.jc_nt};
    const std::size_t nc = nc_for_model(nc_model::m1_balanced, in);
    if (nc == 0 || nc == r.nc) return {r.nc, r.nc, false};
    const gemm_grid g1 = plan_gemm_grid(m, n, r.mc, nc, r.mr, r.threads);
    const std::size_t pb_new = packed_b_bytes(g1.jc_nt, r.kc, nc, r.sdata);
    const std::size_t pb_old = packed_b_bytes(g0.jc_nt, r.kc, r.nc, r.sdata);
    return {r.nc, nc, nc_guard_declines(pb_new, pb_old, r.l3_detected)};
}

} // namespace

TEST_CASE("the guard declines exactly the two arms that regressed",
          "[detail][gemm][nc][guard]") {
    // The shapes are the ones derive_nc_shapes nominates: m near mr*T, n a
    // multiple of nc. Replayed for every machine and dtype that was measured.
    std::vector<std::string> declined;
    std::size_t considered = 0;
    for (const recorded& r : MACHINES)
        for (std::size_t m : {std::size_t{6}, std::size_t{12}, std::size_t{18}})
            for (unsigned mult : {2u, 3u, 5u, 7u, 8u}) {
                const std::size_t n = r.nc * mult;
                const verdict v = replay(r, m, n);
                if (v.nc_m1 == v.nc_m0) continue;      // M1 is a no-op here
                ++considered;
                if (v.declined)
                    declined.push_back(std::string(r.machine) + " [" + r.dtype +
                                       "] m=" + std::to_string(m) +
                                       " n=" + std::to_string(n));
            }

    INFO("considered " << considered << " arms where M1's nc differs");
    for (const std::string& d : declined) INFO("declined: " << d);

    // Exactly the two measured regressions, both on the Zen 4:
    //   ryzen double m=18 n=24576  x0.8203 (three sessions: .8181/.8181/.8203)
    //   ryzen float  m=18 n=49152  x0.8853
    CHECK(considered == 44);
    REQUIRE(declined.size() == 2);
    CHECK(declined[0] == "ryzen-9-8945hs [double] m=18 n=24576");
    CHECK(declined[1] == "ryzen-9-8945hs [float] m=18 n=49152");
}

TEST_CASE("both terms of the guard are load-bearing", "[detail][gemm][nc][guard]") {
    // Either term alone would decline arms that MEASURED AS GAINS -- 2 extra for
    // "grew", 33 for "exceeds L3". This is the check that would fail if someone
    // simplified the conjunction away.
    std::size_t both = 0, grew_only = 0, over_only = 0;
    for (const recorded& r : MACHINES)
        for (std::size_t m : {std::size_t{6}, std::size_t{12}, std::size_t{18}})
            for (unsigned mult : {2u, 3u, 5u, 7u, 8u}) {
                const std::size_t n = r.nc * mult;
                const gemm_grid g0 = plan_gemm_grid(m, n, r.mc, r.nc, r.mr, r.threads);
                const nc_model_inputs in{n, r.kc, r.nr, r.sdata, L3_DEFAULT,
                                         r.l3_detected, r.l3_sharers, g0.jc_nt};
                const std::size_t nc = nc_for_model(nc_model::m1_balanced, in);
                if (nc == 0 || nc == r.nc) continue;
                const gemm_grid g1 = plan_gemm_grid(m, n, r.mc, nc, r.mr, r.threads);
                const std::size_t pn = packed_b_bytes(g1.jc_nt, r.kc, nc, r.sdata);
                const std::size_t po = packed_b_bytes(g0.jc_nt, r.kc, r.nc, r.sdata);
                if (pn > po) ++grew_only;
                if (pn > r.l3_detected) ++over_only;
                if (nc_guard_declines(pn, po, r.l3_detected)) ++both;
            }
    INFO("grew=" << grew_only << " over=" << over_only << " both=" << both);
    CHECK(both == 2);
    CHECK(grew_only > both);     // "grew" alone over-declines
    CHECK(over_only > both);     // "exceeds L3" alone over-declines by far
}

TEST_CASE("nc_guard_declines is the stated conjunction", "[detail][gemm][nc][guard]") {
    CHECK(nc_guard_declines(24, 16, 16));        // grew and exceeds -> decline
    CHECK_FALSE(nc_guard_declines(24, 16, 25));  // grew but still fits
    CHECK_FALSE(nc_guard_declines(20, 32, 16));  // exceeds but SHRANK
    CHECK_FALSE(nc_guard_declines(16, 16, 8));   // unchanged
    CHECK_FALSE(nc_guard_declines(8, 16, 4));    // shrank
}

TEST_CASE("M6 is the default, and M0 still restores the pre-#429 behaviour",
          "[detail][gemm][nc][guard]") {
    // #429 makes the measured winner the default. M0 is kept so an arm labelled
    // "the old behaviour" IS the old behaviour rather than a reconstruction.
    CHECK(nc_model_selection() == nc_model::m6_guarded);

    constexpr auto bp = mtl::simd::default_blocking<double>;
    for (unsigned t : {1u, 2u, 4u, 8u}) {
        INFO("threads=" << t);
        CHECK(mtl::detail::nc_for_plan<double>(4096, 8192, t, nc_model::m0_default)
              == bp.nc);
    }
}

TEST_CASE("M6 never invents a third answer", "[detail][gemm][nc][guard]") {
    // It either takes M1's nc or falls back to M0's. Anything else means the
    // guard has started computing rather than deciding.
    constexpr auto bp = mtl::simd::default_blocking<double>;
    for (std::size_t m : {std::size_t{6}, std::size_t{18}, std::size_t{64},
                          std::size_t{1024}})
        for (unsigned mult : {1u, 2u, 3u, 5u, 8u})
            for (unsigned t : {1u, 2u, 4u, 6u, 8u}) {
                const std::size_t n = bp.nc * mult;
                const std::size_t m6 =
                    mtl::detail::nc_for_plan<double>(m, n, t, nc_model::m6_guarded);
                const std::size_t m1 =
                    mtl::detail::nc_for_plan<double>(m, n, t, nc_model::m1_balanced);
                const std::size_t m0 =
                    mtl::detail::nc_for_plan<double>(m, n, t, nc_model::m0_default);
                INFO("m=" << m << " n=" << n << " T=" << t
                          << ": m6=" << m6 << " m1=" << m1 << " m0=" << m0);
                CHECK((m6 == m1 || m6 == m0));
                CHECK(m6 > 0);
            }
}

TEST_CASE("M6 changes nothing where M1 changes nothing",
          "[detail][gemm][nc][guard]") {
    // Square, tall/thin and serial shapes: `balanced_nc` is a no-op when `njb`
    // already divides `jc_nt`, so the new default cannot touch them. This is why
    // #479's square controls could not regress under M1 by construction, and it
    // is the property that bounds the blast radius of making M6 the default.
    constexpr auto bp = mtl::simd::default_blocking<double>;
    struct sh { std::size_t m, n; unsigned t; };
    for (const sh& s : {sh{1024, 1024, 8}, sh{2048, 2048, 8}, sh{8192, 64, 8},
                        sh{64, bp.nc * 2, 1}, sh{64, bp.nc * 5, 1}}) {
        INFO("m=" << s.m << " n=" << s.n << " T=" << s.t);
        CHECK(mtl::detail::nc_for_plan<double>(s.m, s.n, s.t, nc_model::m6_guarded)
              == mtl::detail::nc_for_plan<double>(s.m, s.n, s.t, nc_model::m0_default));
    }
}

TEST_CASE("m6_guarded round-trips through its name", "[detail][gemm][nc][guard]") {
    bool ok = false;
    CHECK(mtl::detail::nc_model_from_name("m6_guarded", ok) == nc_model::m6_guarded);
    CHECK(ok);
    CHECK(std::string(nc_model_name(nc_model::m6_guarded)) == "m6_guarded");
    // and it is in the sweep/harness list, so both tools measure the default
    bool present = false;
    for (nc_model mo : all_nc_models) if (mo == nc_model::m6_guarded) present = true;
    CHECK(present);
}
