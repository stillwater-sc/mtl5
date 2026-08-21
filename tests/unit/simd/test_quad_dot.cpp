// int8 -> int32 dot via the quad multiply-accumulate -- #451 phase 3.
//
// This is the case the epic calls "the one that matters", and the reason is
// headroom, not lane count. Against phase 2's 16-bit kernel, from the same
// 32-bit accumulator:
//
//   pairing        max |product|   terms before the sum can overflow
//   int16 x int16     2^30         3 worst case, ~36 random
//    int8 x int8      2^14         ~131 000
//   uint8 x  int8      ~2^15       ~65 000
//   uint8 x uint8      ~2^16       ~66 000
//
// Four to five orders of magnitude, purely because halving the operand width
// quarters the product. That is why the quantized-inference instructions are
// 8-bit, and why phase 2's kernel is for small operands while this one is for
// real vectors. The headroom claims are tested below, not just asserted.
//
// THE SIGNEDNESS ASYMMETRY IS THE HARDWARE'S. VNNI implements unsigned x signed
// (vpdpbusd) -- unsigned activations against signed weights -- and the symmetric
// i8 x i8 form only from AVX10.2. `(int8, uint8)` is rejected rather than
// silently reordered, since a dot product is symmetric and swapping the
// arguments gets the native instruction.
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include <mtl/interface/dispatch_traits.hpp>
#include <mtl/operation/dot.hpp>
#include <mtl/simd/algorithm.hpp>
#include <mtl/simd/batch.hpp>
#include <mtl/simd/blocking.hpp>
#include <mtl/vec/dense_vector.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

namespace {

/// Exact sum of products reduced to 32 bits, computed in 64.
template <typename W, typename A, typename B>
W ref_quad(const A* a, const B* b, std::size_t n) {
    std::uint64_t acc = 0;
    for (std::size_t i = 0; i < n; ++i)
        acc += static_cast<std::uint64_t>(static_cast<std::uint32_t>(static_cast<W>(a[i]))) *
               static_cast<std::uint32_t>(static_cast<W>(b[i]));
    return static_cast<W>(static_cast<std::uint32_t>(acc));
}

template <typename T> T gen_a(std::size_t i) { return static_cast<T>(static_cast<std::uint8_t>(0x9Eu * (i + 1) + 0x37u)); }
template <typename T> T gen_b(std::size_t i) { return static_cast<T>(static_cast<std::uint8_t>(0x85u * (i + 3) + 0xEBu)); }

// Lengths straddling the quad step on every backend, plus ragged tails.
const std::size_t kLengths[] = {0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33,
                                63, 64, 65, 127, 128, 129, 255, 256, 257, 1031};

} // namespace

TEST_CASE("the quad-widenable types and their pairings", "[simd][quad]") {
    using namespace mtl::simd;
    STATIC_REQUIRE(is_quad_widenable_v<std::int8_t>);
    STATIC_REQUIRE(is_quad_widenable_v<std::uint8_t>);
    STATIC_REQUIRE_FALSE(is_quad_widenable_v<std::int16_t>);   // phase 2's op
    STATIC_REQUIRE_FALSE(is_quad_widenable_v<std::int32_t>);   // already a lane

    // The three pairings the hardware op accepts...
    STATIC_REQUIRE(QuadPair<std::int8_t,  std::int8_t>);
    STATIC_REQUIRE(QuadPair<std::uint8_t, std::int8_t>);       // VNNI's native shape
    STATIC_REQUIRE(QuadPair<std::uint8_t, std::uint8_t>);
    // ... and the one it does not. A dot product is symmetric, so the caller
    // swaps and gets the native instruction; being told beats being detoured.
    STATIC_REQUIRE_FALSE(QuadPair<std::int8_t, std::uint8_t>);

    STATIC_REQUIRE(std::is_same_v<quad_accumulator_t<std::int8_t,  std::int8_t>,  std::int32_t>);
    STATIC_REQUIRE(std::is_same_v<quad_accumulator_t<std::uint8_t, std::int8_t>,  std::int32_t>);
    STATIC_REQUIRE(std::is_same_v<quad_accumulator_t<std::uint8_t, std::uint8_t>, std::uint32_t>);

    // 8-bit types are operands, not lanes -- x86 has no 8-bit vector multiply.
    STATIC_REQUIRE_FALSE(is_lane_v<std::int8_t>);
    STATIC_REQUIRE_FALSE(is_lane_v<std::uint8_t>);

    // Four products per accumulator lane, against the pairwise op's two.
    STATIC_REQUIRE(batch<std::int32_t>::quad_step == 4 * batch<std::int32_t>::size);
    STATIC_REQUIRE(batch<std::int32_t>::quad_step == 2 * batch<std::int32_t>::widen_step);
}

TEST_CASE("int8 x int8 dot is exact mod 2^32 at every length", "[simd][quad]") {
    for (std::size_t n : kLengths) {
        std::vector<std::int8_t> a(n), b(n);
        for (std::size_t i = 0; i < n; ++i) { a[i] = gen_a<std::int8_t>(i); b[i] = gen_b<std::int8_t>(i); }
        INFO("n=" << n);
        CHECK(mtl::simd::reduce_dot_widen<std::int32_t, std::int8_t>(a.data(), b.data(), n) ==
              (ref_quad<std::int32_t, std::int8_t, std::int8_t>(a.data(), b.data(), n)));
    }
}

TEST_CASE("uint8 x int8 -- VNNI's native shape -- is exact at every length", "[simd][quad]") {
    for (std::size_t n : kLengths) {
        std::vector<std::uint8_t> a(n);
        std::vector<std::int8_t>  b(n);
        for (std::size_t i = 0; i < n; ++i) { a[i] = gen_a<std::uint8_t>(i); b[i] = gen_b<std::int8_t>(i); }
        INFO("n=" << n);
        CHECK((mtl::simd::reduce_dot_widen<std::int32_t, std::uint8_t, std::int8_t>(a.data(), b.data(), n)) ==
              (ref_quad<std::int32_t, std::uint8_t, std::int8_t>(a.data(), b.data(), n)));
    }
}

TEST_CASE("uint8 x uint8 is exact at every length", "[simd][quad]") {
    for (std::size_t n : kLengths) {
        std::vector<std::uint8_t> a(n), b(n);
        for (std::size_t i = 0; i < n; ++i) { a[i] = gen_a<std::uint8_t>(i); b[i] = gen_b<std::uint8_t>(i); }
        INFO("n=" << n);
        CHECK((mtl::simd::reduce_dot_widen<std::uint32_t, std::uint8_t>(a.data(), b.data(), n)) ==
              (ref_quad<std::uint32_t, std::uint8_t, std::uint8_t>(a.data(), b.data(), n)));
    }
}

TEST_CASE("quad dot is invariant under partitioning", "[simd][quad]") {
    // Cut anywhere, add the halves, get the whole back exactly -- the same
    // associativity that makes the horizontal reduce order-independent. Here it
    // also has to survive splits that land mid-quad, where the two sides pick up
    // different scalar tails.
    constexpr std::size_t n = 137;
    std::vector<std::int8_t> a(n), b(n);
    for (std::size_t i = 0; i < n; ++i) { a[i] = gen_a<std::int8_t>(i); b[i] = gen_b<std::int8_t>(i); }

    const auto whole = mtl::simd::reduce_dot_widen<std::int32_t, std::int8_t>(a.data(), b.data(), n);
    for (std::size_t k = 0; k <= n; ++k) {
        const auto lo = mtl::simd::reduce_dot_widen<std::int32_t, std::int8_t>(a.data(), b.data(), k);
        const auto hi = mtl::simd::reduce_dot_widen<std::int32_t, std::int8_t>(a.data() + k, b.data() + k, n - k);
        INFO("split at k=" << k);
        CHECK(static_cast<std::int32_t>(static_cast<std::uint32_t>(lo) +
                                        static_cast<std::uint32_t>(hi)) == whole);
    }
}

TEST_CASE("the headroom that makes 8-bit the useful width", "[simd][quad]") {
    // The whole point of phase 3 over phase 2, tested rather than asserted.
    // Worst case throughout: every term maximal and same-signed, so nothing
    // cancels and the sum grows as fast as it possibly can.

    SECTION("int8 x int8 survives 100 000 worst-case terms") {
        // (-128)*(-128) = 16384 = 2^14, so the sum reaches 2^31 only at
        // k = 131 072. 100 000 terms must therefore still be exact and positive.
        constexpr std::size_t n = 100000;
        std::vector<std::int8_t> a(n, -128), b(n, -128);
        const auto got = mtl::simd::reduce_dot_widen<std::int32_t, std::int8_t>(a.data(), b.data(), n);
        CHECK(got == static_cast<std::int32_t>(n * 16384ull));      // 1 638 400 000 < 2^31
        CHECK(got > 0);                                             // did not wrap
    }

    SECTION("uint8 x int8 survives 60 000 worst-case terms") {
        // 255 * 127 = 32385, so the sum reaches 2^31 near k = 66 000.
        constexpr std::size_t n = 60000;
        std::vector<std::uint8_t> a(n, 255);
        std::vector<std::int8_t>  b(n, 127);
        const auto got = mtl::simd::reduce_dot_widen<std::int32_t, std::uint8_t, std::int8_t>(
            a.data(), b.data(), n);
        CHECK(got == static_cast<std::int32_t>(n * 32385ull));       // 1 943 100 000 < 2^31
        CHECK(got > 0);
    }

    SECTION("the same worst case in 16-bit storage wraps almost immediately") {
        // The contrast that motivates this phase: identical accumulator, operands
        // one width wider, and three terms is already too many.
        constexpr auto lo = std::numeric_limits<std::int16_t>::lowest();
        std::vector<std::int16_t> p(3, lo), q(3, lo);
        const auto got = mtl::simd::reduce_dot_widen<std::int32_t, std::int16_t>(p.data(), q.data(), 3);
        CHECK(got == static_cast<std::int32_t>(static_cast<std::uint32_t>(3ull << 30)));
        CHECK(got < 0);                                             // i.e. it wrapped
    }
}

TEST_CASE("beyond the headroom it wraps, it does not saturate", "[simd][quad]") {
    // 200 000 terms of 2^14 exceeds 2^31; the answer is the true sum mod 2^32,
    // not a clamp to INT32_MAX and not a trap.
    constexpr std::size_t n = 200000;
    std::vector<std::int8_t> a(n, -128), b(n, -128);
    const auto got = mtl::simd::reduce_dot_widen<std::int32_t, std::int8_t>(a.data(), b.data(), n);
    CHECK(got == static_cast<std::int32_t>(static_cast<std::uint32_t>(n * 16384ull)));
    CHECK(got != std::numeric_limits<std::int32_t>::max());         // not saturated
}

// ---------------------------------------------------------------------------

TEST_CASE("dot selects the quad kernel for the pairings that have one",
          "[simd][quad][operation][dot]") {
    using i8v = mtl::vec::dense_vector<std::int8_t>;
    using u8v = mtl::vec::dense_vector<std::uint8_t>;

    STATIC_REQUIRE(mtl::interface::SimdQuadVector<i8v>);
    STATIC_REQUIRE(mtl::quad_int_dot<std::int32_t,  std::int32_t,  i8v, i8v>);
    STATIC_REQUIRE(mtl::quad_int_dot<std::int32_t,  std::int32_t,  u8v, i8v>);   // VNNI native
    STATIC_REQUIRE(mtl::quad_int_dot<std::uint32_t, std::uint32_t, u8v, u8v>);

    // (int8, uint8) has no hardware pairing, so the direct predicate is false --
    // but dot does NOT fall to the generic loop for it. A dot product is
    // symmetric, so the operands are swapped and the native instruction runs.
    STATIC_REQUIRE_FALSE(mtl::quad_int_dot<std::int32_t, std::int32_t, i8v, u8v>);
    STATIC_REQUIRE(mtl::quad_int_dot_swapped<std::int32_t, std::int32_t, i8v, u8v>);
    // The swap applies only where it is needed: a pairing the hardware already
    // has must not be re-routed through it.
    STATIC_REQUIRE_FALSE(mtl::quad_int_dot_swapped<std::int32_t, std::int32_t, u8v, i8v>);
    STATIC_REQUIRE_FALSE(mtl::quad_int_dot_swapped<std::int32_t, std::int32_t, i8v, i8v>);
    // A wider accumulator than the instruction implements also falls back.
    STATIC_REQUIRE_FALSE(mtl::quad_int_dot<std::int64_t, std::int64_t, i8v, i8v>);
    // And 8-bit types are not lanes, so the plain dot path must still reject them.
    STATIC_REQUIRE_FALSE(mtl::interface::SimdDenseVector<i8v>);
}

TEST_CASE("dot<int32_t> over 8-bit vectors agrees with the exact sum",
          "[simd][quad][operation][dot]") {
    constexpr std::size_t n = 259;
    mtl::vec::dense_vector<std::uint8_t> a(n);
    mtl::vec::dense_vector<std::int8_t>  b(n);
    std::int64_t exact = 0;
    for (std::size_t i = 0; i < n; ++i) {
        a(i) = gen_a<std::uint8_t>(i);
        b(i) = gen_b<std::int8_t>(i);
        exact += std::int64_t(a(i)) * b(i);
    }
    // These operands and this length are inside the headroom, so the 32-bit
    // answer is not merely congruent to the true sum -- it IS the true sum.
    CHECK((mtl::dot<std::int32_t>(a, b)) == exact);
    CHECK((mtl::dot_real<std::int32_t>(a, b)) == exact);
    CHECK((mtl::dot_real<std::int64_t>(a, b)) == exact);   // wider accumulator agrees
}

// The other half of what phase 3 owes: "re-deriving the register tile for a
// 1-byte operand and a 4-byte accumulator". The answer is that there is nothing
// to re-derive, and it is worth pinning so a future int8 GEMM does not go
// looking for a tile that does not need deriving.
TEST_CASE("the int8 GEMM register tile is the float tile", "[simd][quad][blocking]") {
    using namespace mtl::simd;
    for (std::size_t nvec : {std::size_t{4}, std::size_t{8}, std::size_t{16}}) {
        const auto f = derive_blocking<float>(nvec);
        const auto i = derive_blocking<std::int32_t>(nvec);
        INFO("nvec=" << nvec);
        // derive_blocking sizes the tile from the ACCUMULATOR -- which is what
        // gemm_blocked<TC, TAB> already passes it -- and the 1-byte operand
        // never enters. int32 and float share a lane width and a register file,
        // so they share a tile.
        CHECK(i.mr == f.mr);
        CHECK(i.nr == f.nr);
        CHECK(i.kc == f.kc);

        // The real constraint a 1-byte operand imposes is on the K GRANULARITY,
        // not the tile: VNNI and SDOT consume FOUR k-values per instruction, so
        // kc must be a multiple of 4 and the pack layout quad-interleaved.
        CHECK(i.kc % 4 == 0);
    }
}

TEST_CASE("the reversed 8-bit pairing is swapped, not demoted",
          "[simd][quad][operation][dot]") {
    // dot(i8, u8) and dot(u8, i8) are the same mathematical quantity, and both
    // must take the native instruction rather than one of them silently
    // dropping to the generic loop an order of magnitude away.
    constexpr std::size_t n = 259;
    mtl::vec::dense_vector<std::int8_t>  a(n);
    mtl::vec::dense_vector<std::uint8_t> b(n);
    std::int64_t exact = 0;
    for (std::size_t i = 0; i < n; ++i) {
        a(i) = gen_b<std::int8_t>(i);
        b(i) = gen_a<std::uint8_t>(i);
        exact += std::int64_t(a(i)) * b(i);
    }
    CHECK((mtl::dot_real<std::int32_t>(a, b)) == exact);          // reversed order
    CHECK((mtl::dot_real<std::int32_t>(b, a)) == exact);          // native order
    CHECK((mtl::dot_real<std::int32_t>(a, b)) == (mtl::dot_real<std::int32_t>(b, a)));
    // conj is the identity on the integers, so the Hermitian spelling agrees too.
    CHECK((mtl::dot<std::int32_t>(a, b)) == exact);
    CHECK((mtl::dot<std::int32_t>(b, a)) == exact);
}
