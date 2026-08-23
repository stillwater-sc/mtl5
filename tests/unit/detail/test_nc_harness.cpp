// The nc-model HARNESS ARM: does asking for a model actually change what the
// nest does, and does it change nothing else? (#479, feeding #429)
//
// test_nc_models covers the models as pure functions. This covers the WIRING --
// `nc_model_selection`, `nc_for_plan`, and the path from `gemm_blocked`'s new
// `model` parameter down to the jc loop's step. Three properties, and each of
// them has already failed once somewhere in this codebase:
//
//   1. THE DEFAULT IS THE SHIPPED PATH. #426 merged runtime cache detection on
//      the argument that measured sizes must beat constants tuned for a Haswell
//      core, and lost on every machine that ran it. Six unmeasured models are
//      six chances to repeat that, so the arm has to be inert until asked.
//
//   2. THE ARGUMENT REACHES THE BLOCK SIZE. #470 shipped a benchmark whose two
//      int8 arms were secretly one arm, because kernel selection was inferred
//      from the element types and (i8,i8) is valid input to both. It was caught
//      by the two arms reporting the same number to three digits -- a full green
//      test suite had not noticed. A selector that stops selecting is invisible
//      to every test that only checks answers.
//
//   3. THE ANSWER DOES NOT MOVE. `nc` is a blocking parameter, not a semantic
//      one: the FMA order for a given C element is fixed by the pc loop, and jc
//      blocking only decides which COLUMNS are grouped. So all six models must
//      agree BIT FOR BIT. This is the property that says the wiring is right
//      rather than merely different, and it is the one that would catch a
//      mis-sized packed-B buffer -- which produces wrong answers, not crashes.
//
// Property 2 is asserted against `nc_for_plan` rather than `gemm_plan_for`
// because the latter clamps its budget to the thread pool, and the pool is size
// 1 unless MTL5_NUM_THREADS is set -- which it is not, in CI. At jc_nt == 1
// every balancing model is a no-op BY CONSTRUCTION, so a plan-level test would
// pass while asserting nothing. `nc_for_plan` takes the budget as an argument,
// so the jc-parallel regime is reachable from a single-threaded test process.
#include <catch2/catch_test_macros.hpp>

#include <mtl/detail/gemm_blocked.hpp>
#include <mtl/detail/nc_model.hpp>
#include <mtl/simd/blocking.hpp>

#include <cstddef>
#include <cstring>
#include <vector>

using mtl::detail::all_nc_models;
using mtl::detail::gemm_plan_for;
using mtl::detail::nc_for_plan;
using mtl::detail::nc_model;
using mtl::detail::nc_model_from_name;
using mtl::detail::nc_model_name;
using mtl::detail::nc_model_selection;

TEST_CASE("M0 computed equals the constant the nest ships", "[detail][gemm][nc]") {
    // `nc_for_plan` SHORT-CIRCUITS M0 to `bp.nc` instead of computing it, so the
    // baseline arm is the shipped code rather than a reimplementation that
    // merely agrees. That makes the equality below a test rather than a
    // load-bearing assumption -- and this is the test.
    //
    // It matters beyond tidiness: the four committed sweeps (#481, #483) take M0
    // as their baseline. If `nc_from_budget` and `derive_blocking` ever drifted,
    // every "M1 differs from M0 on N shapes" count in those CSVs would be
    // answering a question about a baseline the library does not use.
    auto check = [](auto tag, std::size_t l3_default) {
        using T = decltype(tag);
        constexpr auto bp = mtl::simd::default_blocking<T>;
        const mtl::detail::nc_model_inputs in{
            4096, bp.kc, bp.nr, sizeof(T), l3_default, 0, 0, 1u};
        INFO("kc=" << bp.kc << " nr=" << bp.nr << " sizeof=" << sizeof(T));
        CHECK(mtl::detail::nc_for_model(nc_model::m0_default, in) == bp.nc);
    };
    const std::size_t l3 = mtl::simd::default_hw_traits.l3_bytes;
    check(double{}, l3);
    check(float{}, l3);
}

TEST_CASE("the harness arm is inert until asked", "[detail][gemm][nc]") {
    // No MTL5_NC_MODEL in the environment -> M0, and a plan that reports exactly
    // the compile-time constant. If this fails, some model became the default
    // without the measurement #479 exists to produce.
    CHECK(nc_model_selection() == nc_model::m0_default);

    constexpr auto bpd = mtl::simd::default_blocking<double>;
    for (unsigned t : {1u, 2u, 4u, 8u}) {
        INFO("threads=" << t);
        CHECK(gemm_plan_for<double>(4096, 8192, t).nc == bpd.nc);
        CHECK(nc_for_plan<double>(4096, 8192, t, nc_model::m0_default) == bpd.nc);
    }
}

TEST_CASE("the model argument reaches the block size", "[detail][gemm][nc]") {
    // The #470 guard, on this axis: a selector that quietly stopped selecting
    // would leave every arm running M0 while the CSV's `model` column claimed
    // otherwise, and no answer-checking test could tell.
    //
    // Asserted at a jc-parallel shape, because that is where M1 is defined to do
    // anything. The shape is the one the Xeon sweep nominates (m ~ mr*T, n a
    // multiple of nc); `nc_for_plan` takes the budget directly, so this holds
    // whatever the test process's pool size is.
    constexpr auto bp = mtl::simd::default_blocking<double>;
    const std::size_t m = bp.mr, n = bp.nc * 8;
    const unsigned budget = 6;

    const std::size_t nc0 = nc_for_plan<double>(m, n, budget, nc_model::m0_default);
    std::size_t moved = 0;
    for (nc_model mo : all_nc_models) {
        const std::size_t nc = nc_for_plan<double>(m, n, budget, mo);
        INFO(nc_model_name(mo) << " -> nc=" << nc << " (m0 -> " << nc0 << ")");
        CHECK(nc > 0);                       // never a zero loop step
        if (mo != nc_model::m0_default && nc != nc0) ++moved;
    }
    // Not "all five differ" -- M2 and M5 coincide on some machines, and on a
    // machine whose detected L3 equals the compile-time figure several collapse
    // onto M0 legitimately. But if NOTHING moves, the argument is not connected.
    INFO("models differing from M0 at m=" << m << " n=" << n << " T=" << budget);
    CHECK(moved > 0);
}

TEST_CASE("every model name round-trips", "[detail][gemm][nc]") {
    for (nc_model mo : all_nc_models) {
        bool ok = false;
        INFO(nc_model_name(mo));
        CHECK(nc_model_from_name(nc_model_name(mo), ok) == mo);
        CHECK(ok);
    }
}

TEST_CASE("an unrecognised model name is refused, not defaulted",
          "[detail][gemm][nc]") {
    // `nc_model_selection` aborts on a bad name and cannot be tested in-process;
    // this pins the predicate it aborts on. A typo'd MTL5_NC_MODEL that silently
    // ran M0 would forge the provenance of every row of the CSV that followed.
    for (const char* bad : {"m1_balnced", "M1_BALANCED", "m1", "balanced", "", "m6"}) {
        bool ok = true;
        INFO("name=\"" << bad << "\"");
        CHECK(nc_model_from_name(bad, ok) == nc_model::m0_default);
        CHECK_FALSE(ok);
    }
    bool ok = true;
    CHECK(nc_model_from_name(nullptr, ok) == nc_model::m0_default);
    CHECK_FALSE(ok);
}

TEST_CASE("all six models compute the same C, bit for bit",
          "[detail][gemm][nc]") {
    // The correctness property of the whole arm. `nc` groups columns; it does not
    // reorder any C element's FMA chain, which the pc loop fixes. So a model that
    // changed the ANSWER would mean the nest was mis-stepping or the packed-B
    // buffer was mis-sized -- both of which corrupt silently rather than crash.
    //
    // Compared with memcmp, not an epsilon: "close" is the wrong bar here, and a
    // tolerance would hide exactly the partial-block bug this targets.
    //
    // Shapes deliberately include ragged n (1031, not a multiple of nc or nr) and
    // a square case where jc parallelism cannot appear, so the models that still
    // move nc there -- the capacity ones -- are exercised too. This runs at the
    // process's own thread count, so in CI (pool = 1) it covers M2..M5, whose nc
    // moves with the DETECTED L3 and is independent of the partition.
    struct shape { std::size_t m, n, k; };
    const shape shapes[] = {
        {6, 4096, 256}, {64, 5120, 300}, {257, 1031, 129}, {512, 512, 96},
    };
    const unsigned nt = mtl::detail::gemm_default_threads();

    for (const shape& s : shapes) {
        std::vector<double> A(s.m * s.k), B(s.k * s.n);
        for (std::size_t i = 0; i < A.size(); ++i) A[i] = double((i * 37) % 101) - 50.0;
        for (std::size_t i = 0; i < B.size(); ++i) B[i] = double((i * 53) % 97) - 48.0;

        std::vector<double> ref(s.m * s.n), got(s.m * s.n);
        bool first = true;
        for (nc_model mo : all_nc_models) {
            std::vector<double>& out = first ? ref : got;
            std::fill(out.begin(), out.end(), 0.0);
            mtl::detail::gemm_blocked<double>(
                s.m, s.n, s.k, 1.0,
                A.data(), static_cast<std::ptrdiff_t>(s.k), 1,
                B.data(), static_cast<std::ptrdiff_t>(s.n), 1,
                0.0, out.data(), s.n, nt, mo);
            if (first) { first = false; continue; }
            INFO("m=" << s.m << " n=" << s.n << " k=" << s.k << " threads=" << nt
                      << " model=" << nc_model_name(mo)
                      << " nc=" << nc_for_plan<double>(s.m, s.n, nt, mo)
                      << " vs m0 nc=" << nc_for_plan<double>(s.m, s.n, nt,
                                                             nc_model::m0_default));
            CHECK(std::memcmp(ref.data(), got.data(), ref.size() * sizeof(double)) == 0);
        }
    }
}
