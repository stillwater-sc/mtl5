// Widening integer dot: 16-bit operands accumulated in 32 bits -- #451 phase 2.
//
// The interesting property is what the tests DO NOT have to say. The hardware
// instruction behind this (`vpmaddwd` on x86, `SMLAL` on NEON) is free to permute
// which product lands in which accumulator lane; Highway promises only that the
// total is right. A floating-point kernel could not use it without qualifying
// every result by ISA, because the grouping would change the rounding. Here the
// permutation is unobservable -- addition mod 2^32 is associative and
// commutative -- so every case below asserts exact equality against a closed form
// and never mentions lane order, backend or vector width.
//
// The reference is computed in 64-bit and truncated. Products of two 16-bit
// values are exact in 32 bits, so only the running sum can wrap, and reduction
// mod 2^32 commutes with the 64-bit accumulation.
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include <mtl/interface/dispatch_traits.hpp>
#include <mtl/operation/dot.hpp>
#include <mtl/simd/algorithm.hpp>
#include <mtl/simd/batch.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/vec/strided_vector_ref.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

namespace {

template <typename Narrow>
using Wide = mtl::simd::widen_accumulator_t<Narrow>;

/// Exact sum of products, reduced to the accumulator width.
template <typename Narrow>
Wide<Narrow> ref_widen_dot(const Narrow* a, const Narrow* b, std::size_t n) {
    std::uint64_t acc = 0;
    for (std::size_t i = 0; i < n; ++i)
        acc += static_cast<std::uint64_t>(static_cast<std::uint32_t>(static_cast<Wide<Narrow>>(a[i]))) *
               static_cast<std::uint32_t>(static_cast<Wide<Narrow>>(b[i]));
    return static_cast<Wide<Narrow>>(static_cast<std::uint32_t>(acc));
}

/// Full-range 16-bit data: every product is near 2^30, so the sum leaves the
/// accumulator within a handful of terms. That is the regime the contract has to
/// define rather than avoid.
template <typename Narrow>
Narrow gen_a(std::size_t i) { return static_cast<Narrow>(static_cast<std::uint16_t>(0x9E37u * (i + 1) + 0x1234u)); }
template <typename Narrow>
Narrow gen_b(std::size_t i) { return static_cast<Narrow>(static_cast<std::uint16_t>(0x85EBu * (i + 3) + 0x5678u)); }

const std::size_t kLengths[] = {0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 23, 31,
                                32, 33, 63, 64, 65, 100, 127, 128, 129, 257, 1031};

} // namespace

TEST_CASE("the widenable types and their accumulators", "[simd][widen]") {
    using namespace mtl::simd;
    STATIC_REQUIRE(is_widenable_v<std::int16_t>);
    STATIC_REQUIRE(is_widenable_v<std::uint16_t>);
    STATIC_REQUIRE_FALSE(is_widenable_v<std::int8_t>);      // phase 3
    STATIC_REQUIRE_FALSE(is_widenable_v<std::int32_t>);     // already a lane
    STATIC_REQUIRE_FALSE(is_widenable_v<float>);

    STATIC_REQUIRE(std::is_same_v<widen_accumulator_t<std::int16_t>,  std::int32_t>);
    STATIC_REQUIRE(std::is_same_v<widen_accumulator_t<std::uint16_t>, std::uint32_t>);

    // 16-bit types are operands, NOT lanes: batch<int16_t> is deliberately absent,
    // because a 16-bit lane's value is this accumulate, not a plain multiply.
    STATIC_REQUIRE_FALSE(is_lane_v<std::int16_t>);
    STATIC_REQUIRE_FALSE(is_lane_v<std::uint16_t>);

    // widen_step is a compile-time constant, like size, and consumes a full
    // narrow vector -- twice as many 16-bit lanes as the batch has 32-bit ones.
    STATIC_REQUIRE(batch<std::int32_t>::widen_step == 2 * batch<std::int32_t>::size);
}

TEMPLATE_TEST_CASE("widening dot is exact mod 2^32 at every length", "[simd][widen]",
                   std::int16_t, std::uint16_t) {
    using N = TestType;
    using W = Wide<N>;
    for (std::size_t n : kLengths) {
        std::vector<N> a(n), b(n);
        for (std::size_t i = 0; i < n; ++i) { a[i] = gen_a<N>(i); b[i] = gen_b<N>(i); }
        INFO("n=" << n);
        CHECK(mtl::simd::reduce_dot_widen<W, N>(a.data(), b.data(), n) ==
              ref_widen_dot<N>(a.data(), b.data(), n));
    }
}

TEMPLATE_TEST_CASE("widening dot is invariant under partitioning", "[simd][widen]",
                   std::int16_t, std::uint16_t) {
    // The same associativity argument as the plain integer dot, and the reason
    // the hardware's lane permutation is harmless: cut the reduction anywhere,
    // add the two halves, get the whole back exactly.
    using N = TestType;
    using W = Wide<N>;
    constexpr std::size_t n = 137;                     // prime: no split aligns
    std::vector<N> a(n), b(n);
    for (std::size_t i = 0; i < n; ++i) { a[i] = gen_a<N>(i); b[i] = gen_b<N>(i); }

    const W whole = mtl::simd::reduce_dot_widen<W, N>(a.data(), b.data(), n);
    for (std::size_t k = 0; k <= n; ++k) {
        const W lo = mtl::simd::reduce_dot_widen<W, N>(a.data(), b.data(), k);
        const W hi = mtl::simd::reduce_dot_widen<W, N>(a.data() + k, b.data() + k, n - k);
        INFO("split at k=" << k);
        CHECK(static_cast<W>(static_cast<std::uint32_t>(
                  static_cast<std::uint32_t>(lo) + static_cast<std::uint32_t>(hi))) == whole);
    }
}

TEST_CASE("products are exact even at the extremes; only the sum wraps",
          "[simd][widen]") {
    // -2^15 * -2^15 = 2^30, which fits an int32 with a bit to spare. A single
    // term is therefore never lost, however extreme the operands.
    constexpr auto lo = std::numeric_limits<std::int16_t>::lowest();   // -32768
    std::vector<std::int16_t> a{lo}, b{lo};
    CHECK(mtl::simd::reduce_dot_widen<std::int32_t, std::int16_t>(a.data(), b.data(), 1)
          == 1073741824);                                              // 2^30, exact

    // Two of them still fit (2^31 does not, but 2^30 + 2^30 wraps to exactly
    // INT32_MIN, which is the defined answer rather than an accident).
    a.push_back(lo); b.push_back(lo);
    CHECK(mtl::simd::reduce_dot_widen<std::int32_t, std::int16_t>(a.data(), b.data(), 2)
          == std::numeric_limits<std::int32_t>::lowest());

    // Three overflows -- the worst case the contract names -- and the result is
    // the true sum 3 * 2^30 reduced mod 2^32, not a trap or a saturation.
    a.push_back(lo); b.push_back(lo);
    CHECK(mtl::simd::reduce_dot_widen<std::int32_t, std::int16_t>(a.data(), b.data(), 3)
          == static_cast<std::int32_t>(static_cast<std::uint32_t>(3ull * (1ull << 30))));
}

TEST_CASE("small operands have the headroom the contract claims", "[simd][widen]") {
    // The useful regime is small operands, not short vectors: at b bits of
    // magnitude the headroom is about 2^(31-2b) terms. With 8-bit operands
    // (b = 8) that is ~2^15, so 20 000 terms of the worst case -- every term
    // maximal and same-signed -- must still be exact.
    constexpr std::size_t n = 20000;
    std::vector<std::int16_t> a(n, 127), b(n, 127);
    const auto got = mtl::simd::reduce_dot_widen<std::int32_t, std::int16_t>(a.data(), b.data(), n);
    CHECK(got == static_cast<std::int32_t>(n * 127 * 127));            // 322 580 000 < 2^31
    CHECK(got > 0);                                                    // i.e. it did not wrap
}

// ---------------------------------------------------------------------------
// Reachability from the public API. The kernel above is only useful if the
// mixed-precision dot() spelling selects it, so assert the predicate directly
// rather than inferring it from a value that the generic loop would also get
// right.

TEST_CASE("dot<int32_t> over int16 vectors selects the widening kernel",
          "[simd][widen][operation][dot]") {
    using v16 = mtl::vec::dense_vector<std::int16_t>;
    using u16 = mtl::vec::dense_vector<std::uint16_t>;

    STATIC_REQUIRE(mtl::interface::SimdNarrowVector<v16>);
    STATIC_REQUIRE(mtl::widening_int_dot<std::int32_t, std::int32_t, v16, v16>);
    STATIC_REQUIRE(mtl::widening_int_dot<std::uint32_t, std::uint32_t, u16, u16>);

    // Everything else keeps the generic accumulator_traits loop, which is
    // correct for all of these: a wider accumulator than the kernel implements,
    // a mismatched operand pair, and a view whose stride the kernel cannot see.
    STATIC_REQUIRE_FALSE(mtl::widening_int_dot<std::int64_t, std::int64_t, v16, v16>);
    STATIC_REQUIRE_FALSE(mtl::widening_int_dot<std::int32_t, std::int32_t, v16, u16>);
    STATIC_REQUIRE_FALSE((mtl::widening_int_dot<std::int32_t, std::int32_t,
                          mtl::vec::strided_vector_ref<std::int16_t>,
                          mtl::vec::strided_vector_ref<std::int16_t>>));
    // 16-bit operands are not lanes, so the PLAIN dot path must still reject them.
    STATIC_REQUIRE_FALSE(mtl::interface::SimdDenseVector<v16>);
}

TEST_CASE("dot<int32_t> over int16 vectors agrees with the generic loop",
          "[simd][widen][operation][dot]") {
    constexpr std::size_t n = 259;
    mtl::vec::dense_vector<std::int16_t> a(n), b(n);
    for (std::size_t i = 0; i < n; ++i) { a(i) = gen_a<std::int16_t>(i); b(i) = gen_b<std::int16_t>(i); }

    const auto expected = ref_widen_dot<std::int16_t>(&a(0), &b(0), n);
    CHECK(mtl::dot<std::int32_t>(a, b) == expected);
    CHECK(mtl::dot_real<std::int32_t>(a, b) == expected);

    // A wider accumulator does NOT wrap: int64 has room for every one of these
    // terms, so it must differ from the 32-bit answer and match the true sum.
    std::int64_t exact = 0;
    for (std::size_t i = 0; i < n; ++i) exact += std::int64_t(a(i)) * b(i);
    CHECK(mtl::dot_real<std::int64_t>(a, b) == exact);
    CHECK(exact != std::int64_t(expected));          // i.e. the 32-bit path really wrapped
}
