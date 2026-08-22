// The BROADCAST form of the quad multiply-accumulate -- #451 phase 5.
//
// `quad_dot_accumulate` walks two vectors in step, which is a dot product. A
// GEMM does not: one C row needs one A element -- four of them here, for four k
// values -- multiplied against a whole row of B. So the left operand is the same
// four narrow values in every lane, and lane j must accumulate
//
//     sum[j] += sum_{q<4} a4[q] * b[4*j + q]
//
// WHY THIS NEEDS ITS OWN TEST, and a strict one. The broadcast is implemented as
// a 32-bit splat of the four bytes as they lie in memory, then a reinterpret back
// to 8-bit lanes -- one `vpbroadcastd`, not four inserts. That ties the quad's
// lane order to the machine's BYTE order: on a little-endian target lane 0 gets
// a4[0], which is what pairs it with the `b` load. Get it wrong and the kernel
// still computes something -- a transposed quad -- with no crash and no
// diagnostic, and only a numeric check catches it. Every target that HAS the
// instruction is little-endian, so this is the assertion that would fail loudly
// on a big-endian port instead of silently transposing.
//
// The lane assignment is checked EXACTLY, per lane, not through a horizontal
// sum. `quad_dot_accumulate` gets away with "the total is right" because the
// instruction may permute products across lanes and a reduction cannot observe
// it. A GEMM can: lane j is a COLUMN of C, and a permutation there is a wrong
// answer, not a reordering. So this primitive's contract is stronger than its
// sibling's, and the tests have to be too.
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include <mtl/simd/batch.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

template <typename T> T gen(std::size_t i, std::uint8_t salt) {
    return static_cast<T>(static_cast<std::uint8_t>(0x9Eu * (i + 1) + salt));
}

/// Per-lane check for one operand pairing; the three pairings the hardware op
/// accepts each get a TEST_CASE below.
template <typename NA, typename NB>
void check_lanes() {
    using W = mtl::simd::quad_accumulator_t<NA, NB>;
    using Batch = mtl::simd::batch<W>;
    constexpr std::size_t L = Batch::size;          // lanes = C columns in a batch

    NA a4[4];
    std::vector<NB> b(4 * L);
    for (std::size_t q = 0; q < 4; ++q) a4[q] = gen<NA>(q, 0x37u);
    for (std::size_t i = 0; i < b.size(); ++i)      b[i] = gen<NB>(i, 0xEBu);

    Batch acc{};
    Batch::template quad_dot_broadcast_accumulate<NA, NB>(a4, b.data(), acc);

    std::vector<W> got(L);
    acc.store_unaligned(got.data());

    // Per-lane reference. Computed in 64 bits and reduced, so it is exact and
    // says nothing about the accumulator's own wrapping.
    bool all = true;
    for (std::size_t j = 0; j < L; ++j) {
        std::uint64_t want = 0;
        for (std::size_t q = 0; q < 4; ++q)
            want += static_cast<std::uint64_t>(static_cast<std::uint32_t>(static_cast<W>(a4[q]))) *
                    static_cast<std::uint32_t>(static_cast<W>(b[4 * j + q]));
        const W wantw = static_cast<W>(static_cast<std::uint32_t>(want));
        INFO("lane " << j << " of " << L << ": got " << std::int64_t(got[j])
                     << " want " << std::int64_t(wantw));
        if (got[j] != wantw) all = false;
        CHECK(got[j] == wantw);
    }
    CHECK(all);
}

} // namespace

TEST_CASE("i8 x i8: the broadcast quad lands each product in the right lane",
          "[simd][quad][broadcast]") {
    check_lanes<std::int8_t, std::int8_t>();
}

TEST_CASE("u8 x i8: the broadcast quad lands each product in the right lane",
          "[simd][quad][broadcast]") {
    // VNNI's native shape, and the one the widening load cannot express at all.
    check_lanes<std::uint8_t, std::int8_t>();
}

TEST_CASE("u8 x u8: the broadcast quad lands each product in the right lane",
          "[simd][quad][broadcast]") {
    check_lanes<std::uint8_t, std::uint8_t>();
}

TEST_CASE("the broadcast is a broadcast: every lane sees the same four values",
          "[simd][quad][broadcast]") {
    // A left operand of (1,0,0,0) against a b whose every 4-group starts with
    // the lane index isolates a4[0] * b[4j], so a mis-splatted A -- one that
    // varied by lane, or that shifted -- shows up as a lane-dependent factor.
    using Batch = mtl::simd::batch<std::int32_t>;
    constexpr std::size_t L = Batch::size;

    const std::int8_t a4[4] = {1, 0, 0, 0};
    std::vector<std::int8_t> b(4 * L, 0);
    for (std::size_t j = 0; j < L; ++j) b[4 * j] = static_cast<std::int8_t>(j + 1);

    Batch acc{};
    Batch::template quad_dot_broadcast_accumulate<std::int8_t, std::int8_t>(a4, b.data(), acc);

    std::vector<std::int32_t> got(L);
    acc.store_unaligned(got.data());
    for (std::size_t j = 0; j < L; ++j) {
        INFO("lane " << j);
        CHECK(got[j] == static_cast<std::int32_t>(j + 1));
    }
}

TEST_CASE("the quad position is preserved, not reversed", "[simd][quad][broadcast]") {
    // The endianness guard, stated as arithmetic. a4 = (1,2,4,8) against a b
    // that selects one position per lane group: lane j sees b[4j+q] = 1 only at
    // q == j % 4. A byte-reversed splat would return 8,4,2,1 instead.
    using Batch = mtl::simd::batch<std::int32_t>;
    constexpr std::size_t L = Batch::size;

    const std::int8_t a4[4] = {1, 2, 4, 8};
    std::vector<std::int8_t> b(4 * L, 0);
    for (std::size_t j = 0; j < L; ++j) b[4 * j + (j % 4)] = 1;

    Batch acc{};
    Batch::template quad_dot_broadcast_accumulate<std::int8_t, std::int8_t>(a4, b.data(), acc);

    std::vector<std::int32_t> got(L);
    acc.store_unaligned(got.data());
    const std::int32_t expect[4] = {1, 2, 4, 8};
    for (std::size_t j = 0; j < L; ++j) {
        INFO("lane " << j << " selects quad position " << (j % 4));
        CHECK(got[j] == expect[j % 4]);
    }
}

TEST_CASE("the broadcast accumulates rather than overwriting", "[simd][quad][broadcast]") {
    using Batch = mtl::simd::batch<std::int32_t>;
    constexpr std::size_t L = Batch::size;

    const std::int8_t a4[4] = {2, 3, 5, 7};
    std::vector<std::int8_t> b(4 * L);
    for (std::size_t i = 0; i < b.size(); ++i) b[i] = static_cast<std::int8_t>((i % 11) - 5);

    Batch once{}, thrice{};
    Batch::template quad_dot_broadcast_accumulate<std::int8_t, std::int8_t>(a4, b.data(), once);
    for (int r = 0; r < 3; ++r)
        Batch::template quad_dot_broadcast_accumulate<std::int8_t, std::int8_t>(a4, b.data(), thrice);

    std::vector<std::int32_t> g1(L), g3(L);
    once.store_unaligned(g1.data());
    thrice.store_unaligned(g3.data());
    for (std::size_t j = 0; j < L; ++j) CHECK(g3[j] == 3 * g1[j]);
}

TEST_CASE("a full k-group agrees with the dot form on the diagonal",
          "[simd][quad][broadcast]") {
    // Cross-check against the primitive that was already tested (#451 phase 3):
    // if the broadcast quad IS the b vector's own lane-j group, then lane j of
    // the broadcast result equals the dot form's lane j. Feeding the same b to
    // both, with a4 taken from group j, must agree there.
    using Batch = mtl::simd::batch<std::int32_t>;
    constexpr std::size_t L = Batch::size;

    std::vector<std::int8_t> a(4 * L), b(4 * L);
    for (std::size_t i = 0; i < a.size(); ++i) {
        a[i] = static_cast<std::int8_t>((i * 7) % 61 - 30);
        b[i] = static_cast<std::int8_t>((i * 5) % 53 - 26);
    }

    Batch dot{};
    Batch::template quad_dot_accumulate<std::int8_t, std::int8_t>(a.data(), b.data(), dot);
    std::vector<std::int32_t> gdot(L);
    dot.store_unaligned(gdot.data());

    for (std::size_t j = 0; j < L; ++j) {
        Batch bc{};
        Batch::template quad_dot_broadcast_accumulate<std::int8_t, std::int8_t>(
            a.data() + 4 * j, b.data(), bc);
        std::vector<std::int32_t> gbc(L);
        bc.store_unaligned(gbc.data());
        INFO("group " << j);
        CHECK(gbc[j] == gdot[j]);   // same operands in lane j -> same product sum
    }
}
