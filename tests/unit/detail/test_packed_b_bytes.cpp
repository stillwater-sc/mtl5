// `packed_b_bytes` must report the panel the packer actually builds (#486).
//
// It is the mediator #479 records per point, and the left-hand side of M6's
// guard (#429), so "approximately right" is not good enough: the guard compares
// it against L3 and decides whether to take a blocking change.
//
// It was wrong in three ways, all of which made it report a panel larger than
// the nest builds:
//
//   1. IT USED THE ACCUMULATOR TYPE. The panel is a `packed_buffer<TB>`.
//      Blocking is chosen for `TC` because the C microtile maps to `TC`
//      registers, and the callers inherited `sizeof(TC)` from that. For mixed
//      pairs the two differ -- 2x for float operands into an fp64 accumulator
//      (#176), 4x for i8 into i32 (#451).
//
//   2. IT IGNORED THE COLUMN PADDING. `packed_B_size` reserves
//      `ceil(n/NR)*NR*k`, and `balanced_nc` deliberately does NOT round `nc` to
//      a multiple of `nr` on the threaded path -- rounding after balancing would
//      undo the balance -- so a ragged `nc` is the normal case here.
//
//   3. IT IGNORED THE QUAD k-PADDING. `packed_B_quad_size` reserves
//      `quad_depth(k)`, i.e. `k` rounded up to a multiple of 4.
//
// THE STRONGEST TEST HERE IS THE CROSS-CHECK against the packers themselves:
// `packed_b_bytes` must equal `packed_B_size_for<Quad>(...) * sizeof(TB)` for
// every shape. Anything weaker restates the formula rather than testing it, and
// a formula that agrees with itself is what let this drift in the first place.
//
// WHAT THIS DELIBERATELY DOES NOT CHANGE: `nc`. Correcting the same `sizeof`
// inside `nc_from_budget` would ENLARGE `nc` by 2-4x for the mixed pairs, and
// #479 measured that direction and falsified it -- M2 enlarged `nc` about 1.9x
// via the detected L3 and lost up to 45%, with `balanced_nc` already included.
// The accounting is a fact; the budget is a policy, and the policy stays where
// the measurement left it.
#include <catch2/catch_test_macros.hpp>

#include <mtl/detail/gemm_blocked.hpp>
#include <mtl/detail/gemm_pack.hpp>
#include <mtl/detail/gemm_quad_pack.hpp>

#include <cstddef>
#include <cstdint>

using mtl::detail::packed_b_bytes;
using mtl::detail::packed_B_quad_size;
using mtl::detail::packed_B_size;

TEST_CASE("packed_b_bytes agrees with the non-quad packer", "[detail][gemm][packb]") {
    // The cross-check. Ragged `nc` values are included on purpose: they are what
    // `balanced_nc` produces on the threaded path.
    for (std::size_t kc : {std::size_t{128}, std::size_t{512}, std::size_t{2048}})
        for (std::size_t nc : {std::size_t{1}, std::size_t{7}, std::size_t{1024},
                               std::size_t{1195}, std::size_t{6144}})
            for (std::size_t nr : {std::size_t{1}, std::size_t{4}, std::size_t{8},
                                   std::size_t{16}, std::size_t{32}})
                for (std::size_t sz : {std::size_t{1}, std::size_t{4}, std::size_t{8}})
                    for (unsigned teams : {1u, 2u, 4u, 8u}) {
                        INFO("kc=" << kc << " nc=" << nc << " nr=" << nr
                                   << " sizeof=" << sz << " teams=" << teams);
                        REQUIRE(packed_b_bytes(teams, kc, nc, nr, sz)
                                == teams * packed_B_size(kc, nc, nr) * sz);
                    }
}

TEST_CASE("packed_b_bytes agrees with the quad packer", "[detail][gemm][packb]") {
    // `k_group = 4` is what the quad layout pads to. `kc` values that are not
    // multiples of 4 are the point of the test.
    for (std::size_t kc : {std::size_t{1}, std::size_t{3}, std::size_t{129},
                           std::size_t{512}, std::size_t{2047}})
        for (std::size_t nc : {std::size_t{5}, std::size_t{1024}, std::size_t{1195}})
            for (std::size_t nr : {std::size_t{4}, std::size_t{16}})
                for (unsigned teams : {1u, 4u}) {
                    INFO("kc=" << kc << " nc=" << nc << " nr=" << nr
                               << " teams=" << teams);
                    REQUIRE(packed_b_bytes(teams, kc, nc, nr, sizeof(std::int8_t), 4)
                            == teams * packed_B_quad_size(kc, nc, nr)
                                     * sizeof(std::int8_t));
                }
}

TEST_CASE("the operand type is what counts, not the accumulator",
          "[detail][gemm][packb]") {
    // #486's table, as arithmetic. Same blocking, same panel, different element.
    const std::size_t kc = 2048, nc = 1024, nr = 4;

    // i8 operands into an i32 accumulator: the model used to claim 4x the truth.
    const std::size_t as_acc  = packed_b_bytes(1, kc, nc, nr, sizeof(std::int32_t), 4);
    const std::size_t as_oper = packed_b_bytes(1, kc, nc, nr, sizeof(std::int8_t), 4);
    CHECK(as_acc == 4 * as_oper);
    CHECK(as_oper == packed_B_quad_size(kc, nc, nr) * sizeof(std::int8_t));

    // float operands into an fp64 accumulator (#176): 2x.
    CHECK(packed_b_bytes(1, kc, nc, nr, sizeof(double))
          == 2 * packed_b_bytes(1, kc, nc, nr, sizeof(float)));

    // Homogeneous pairs are unaffected -- which is why this never bit.
    CHECK(packed_b_bytes(2, 256, 1024, 8, 8) == 2u * 256 * 1024 * 8);
}

TEST_CASE("padding is counted on both axes", "[detail][gemm][packb]") {
    // A ragged nc costs a whole extra nr-panel, not a fraction of one.
    CHECK(packed_b_bytes(1, 512, 1195, 4, 8) == 512u * 1196 * 8);
    CHECK(packed_b_bytes(1, 512, 1196, 4, 8) == 512u * 1196 * 8);   // already aligned
    CHECK(packed_b_bytes(1, 512, 1197, 4, 8) == 512u * 1200 * 8);   // next panel

    // A ragged kc costs a whole quad group under the quad layout, and nothing
    // without it.
    CHECK(packed_b_bytes(1, 129, 1024, 4, 1, 4) == 132u * 1024 * 1);
    CHECK(packed_b_bytes(1, 129, 1024, 4, 1, 1) == 129u * 1024 * 1);
}

TEST_CASE("degenerate arguments do not produce a zero or a division fault",
          "[detail][gemm][packb]") {
    // The guard divides nothing, but it COMPARES this against L3, and a zero
    // would read as "the panel fits" for any cache. jc_nt == 0 is documented to
    // mean one team.
    CHECK(packed_b_bytes(0, 256, 1024, 8, 8) == 256u * 1024 * 8);
    CHECK(packed_b_bytes(1, 256, 1024, 0, 8) > 0);      // nr = 0 must not divide by zero
    CHECK(packed_b_bytes(1, 256, 1024, 8, 8, 0) > 0);   // k_group = 0 likewise
}
