// Block-scaled integer dot -- #628.
//
// reduce_dot_block_scaled keeps each block's integer sum in vector lanes and
// scales it in-register, so unlike reduce_dot_widen it is NOT exact: the lane
// partial sums are converted to float one by one, and the blocks are
// accumulated in float. The tests therefore say three different things:
//
//   * against an exact reference (block sums in 64-bit integers, scaled and
//     summed in double) the result is within a float re-association bound;
//   * where no step can round -- small operands, power-of-two scales -- it is
//     exactly the reference, on every backend and block length;
//   * the order of the float steps is fixed, so the result does not move with
//     the alignment of the input or from call to call.
//
// Block lengths deliberately include ones below the SIMD step and ones that are
// not a multiple of it, which take the scalar-remainder path.
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include <mtl/simd/algorithm.hpp>
#include <mtl/simd/batch.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>

namespace {

template <typename A, typename B>
struct Pair { using NA = A; using NB = B; };

using PairI16 = Pair<std::int16_t, std::int16_t>;
using PairU16 = Pair<std::uint16_t, std::uint16_t>;
using PairI8  = Pair<std::int8_t, std::int8_t>;
using PairU8I8 = Pair<std::uint8_t, std::int8_t>;
using PairU8  = Pair<std::uint8_t, std::uint8_t>;

constexpr std::size_t kBlocks[] = {1, 3, 7, 16, 31, 32, 33, 64, 100, 256};
constexpr std::size_t kCounts[] = {0, 1, 2, 5, 13};

/// Operands that respect the overflow precondition at the longest block here
/// (256): the 16-bit pairs carry 12-bit weights against 8-bit activations, as a
/// GGML-style kernel does after folding sub-block scales; the 8-bit pairs use
/// their full range.
template <typename T>
std::vector<T> random_operand(std::size_t n, unsigned seed, bool activation) {
    std::mt19937 rng(seed);
    long lo, hi;
    if constexpr (sizeof(T) == 2) {
        hi = activation ? 127 : 4096;
        lo = std::is_signed_v<T> ? -hi : 0;
    } else {
        lo = std::numeric_limits<T>::min();
        hi = std::numeric_limits<T>::max();
    }
    std::uniform_int_distribution<long> d(lo, hi);
    std::vector<T> v(n);
    for (auto& e : v) e = static_cast<T>(d(rng));
    return v;
}

std::vector<float> random_scales(std::size_t n, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(-2.0f, 2.0f);
    std::vector<float> v(n);
    for (auto& e : v) e = d(rng);
    return v;
}

struct Reference {
    double value = 0;   // sum_b s_b * exact_block_sum_b, in double
    double magnitude = 0;   // sum_b |s_b| * sum_j |a_j x_j|: what the rounding scales with
};

template <typename NA, typename NB>
Reference reference(const NA* a, const NB* x, const float* s, std::size_t nb, std::size_t block) {
    Reference r;
    for (std::size_t b = 0; b < nb; ++b) {
        std::int64_t sum = 0, mag = 0;
        for (std::size_t j = 0; j < block; ++j) {
            const std::int64_t p = std::int64_t(a[b * block + j]) * std::int64_t(x[b * block + j]);
            sum += p;
            mag += p < 0 ? -p : p;
        }
        r.value += double(s[b]) * double(sum);
        r.magnitude += std::fabs(double(s[b])) * double(mag);
    }
    return r;
}

/// Each lane's conversion rounds by at most 2^-24 relative, and each float
/// accumulation step likewise: about n_blocks / 4 fmas per accumulator, the
/// combine, a horizontal reduce over W lanes and the remainder. The bound is
/// deliberately generous in the step count; what it must not be is loose in
/// the magnitude it scales.
double bound(const Reference& r, std::size_t nb) {
    constexpr double W = double(mtl::simd::batch<float>::size);
    return (double(nb) + W + 8.0) * std::ldexp(1.0, -23) * r.magnitude;
}

} // namespace

TEMPLATE_TEST_CASE("reduce_dot_block_scaled is within float rounding of the exact sum",
                   "[simd][block_scaled]", PairI16, PairU16, PairI8, PairU8I8, PairU8) {
    using NA = typename TestType::NA;
    using NB = typename TestType::NB;
    for (std::size_t block : kBlocks) {
        for (std::size_t nb : kCounts) {
            const std::size_t n = nb * block;
            const auto a = random_operand<NA>(n, unsigned(17 + block + nb), false);
            const auto x = random_operand<NB>(n, unsigned(91 + block * nb), true);
            const auto s = random_scales(nb, unsigned(5 + nb));
            const Reference ref = reference(a.data(), x.data(), s.data(), nb, block);
            const double got =
                mtl::simd::reduce_dot_block_scaled(a.data(), x.data(), s.data(), nb, block);
            INFO("block " << block << ", n_blocks " << nb);
            CHECK(std::fabs(got - ref.value) <= bound(ref, nb));
        }
    }
}

TEMPLATE_TEST_CASE("reduce_dot_block_scaled agrees with the per-block reduce_dot_widen loop",
                   "[simd][block_scaled]", PairI16, PairU16, PairI8, PairU8I8, PairU8) {
    // The loop it replaces: one exact integer dot and one horizontal reduce per
    // block, then a scalar scale. Both are within the bound of the exact value,
    // so they are within twice it of each other.
    using NA = typename TestType::NA;
    using NB = typename TestType::NB;
    using Wide = typename mtl::simd::detail::block_accumulator<NA, NB>::type;
    for (std::size_t block : {32u, 256u}) {
        const std::size_t nb = 13, n = nb * block;
        const auto a = random_operand<NA>(n, 3, false);
        const auto x = random_operand<NB>(n, 4, true);
        const auto s = random_scales(nb, 6);
        float loop = 0.0f;
        for (std::size_t b = 0; b < nb; ++b)
            loop += s[b] * static_cast<float>(mtl::simd::reduce_dot_widen<Wide>(
                               a.data() + b * block, x.data() + b * block, block));
        const Reference ref = reference(a.data(), x.data(), s.data(), nb, block);
        const float got =
            mtl::simd::reduce_dot_block_scaled(a.data(), x.data(), s.data(), nb, block);
        INFO("block " << block);
        CHECK(std::fabs(double(got) - double(loop)) <= 2 * bound(ref, nb));
    }
}

TEMPLATE_TEST_CASE("reduce_dot_block_scaled is exact when no step can round",
                   "[simd][block_scaled]", PairI16, PairU16, PairI8, PairU8I8, PairU8) {
    // Operands of magnitude <= 8 keep every block sum <= 64 * 256 = 2^14, so
    // every lane partial converts exactly; scales of 2^-3 .. 2^3 keep every
    // scaled term a multiple of 2^-3 below 2^17, and 13 of them sum below 2^21,
    // inside float's 24 bits. Every float step is then exact, so the result IS
    // the reference -- on every backend, block length and remainder split.
    using NA = typename TestType::NA;
    using NB = typename TestType::NB;
    std::mt19937 rng(11);
    std::uniform_int_distribution<int> va(std::is_signed_v<NA> ? -8 : 0, 8);
    std::uniform_int_distribution<int> vx(std::is_signed_v<NB> ? -8 : 0, 8);
    std::uniform_int_distribution<int> e(-3, 3);
    for (std::size_t block : kBlocks) {
        const std::size_t nb = 13, n = nb * block;
        std::vector<NA> a(n);
        std::vector<NB> x(n);
        std::vector<float> s(nb);
        for (auto& v : a) v = static_cast<NA>(va(rng));
        for (auto& v : x) v = static_cast<NB>(vx(rng));
        for (auto& v : s) v = std::ldexp(1.0f, e(rng)) * (rng() % 2 ? 1.0f : -1.0f);
        const Reference ref = reference(a.data(), x.data(), s.data(), nb, block);
        INFO("block " << block);
        CHECK(double(mtl::simd::reduce_dot_block_scaled(a.data(), x.data(), s.data(), nb,
                                                        block)) == ref.value);
    }
}

TEMPLATE_TEST_CASE("reduce_dot_block_scaled does not move with alignment or between calls",
                   "[simd][block_scaled]", PairI16, PairU16, PairI8, PairU8I8, PairU8) {
    // The determinism contract: the float order depends on the block index and
    // the target, never on where the data sits. The same values copied to four
    // different offsets must give the same bits.
    using NA = typename TestType::NA;
    using NB = typename TestType::NB;
    const std::size_t block = 32, nb = 29, n = nb * block;
    const auto a0 = random_operand<NA>(n, 21, false);
    const auto x0 = random_operand<NB>(n, 22, true);
    const auto s = random_scales(nb, 23);
    const float first =
        mtl::simd::reduce_dot_block_scaled(a0.data(), x0.data(), s.data(), nb, block);
    std::uint32_t first_bits;
    std::memcpy(&first_bits, &first, sizeof first);
    for (std::size_t off = 0; off < 4; ++off) {
        std::vector<NA> a(n + 4);
        std::vector<NB> x(n + 4);
        std::memcpy(a.data() + off, a0.data(), n * sizeof(NA));
        std::memcpy(x.data() + off, x0.data(), n * sizeof(NB));
        const float got =
            mtl::simd::reduce_dot_block_scaled(a.data() + off, x.data() + off, s.data(), nb, block);
        std::uint32_t bits;
        std::memcpy(&bits, &got, sizeof got);
        INFO("offset " << off);
        CHECK(bits == first_bits);
    }
}

TEST_CASE("reduce_dot_block_scaled on empty input is zero", "[simd][block_scaled]") {
    const std::int8_t a[1] = {5};
    const std::int8_t x[1] = {7};
    const float s[1] = {3.0f};
    CHECK(mtl::simd::reduce_dot_block_scaled(a, x, s, 0, 1) == 0.0f);
    CHECK(mtl::simd::reduce_dot_block_scaled(a, x, s, 1, 0) == 0.0f);
}

TEST_CASE("a block shorter than the step never touches the lane path", "[simd][block_scaled]") {
    // One-element blocks are below the step on every backend (the scalar
    // fallback's is 2 or 4). Adding a zero lane times an infinite scale would
    // give NaN; the remainder path alone gives the right infinity.
    const std::int8_t a[2] = {2, -1};
    const std::int8_t x[2] = {3, 5};
    const float s[2] = {std::numeric_limits<float>::infinity(), 1.0f};
    CHECK(mtl::simd::reduce_dot_block_scaled(a, x, s, 2, 1) ==
          std::numeric_limits<float>::infinity());
}

TEMPLATE_TEST_CASE("batch<float>::convert rounds each lane as static_cast does",
                   "[simd][block_scaled]", std::int32_t, std::uint32_t) {
    // Exact through 2^24, round-to-nearest-even above it, and the extremes.
    using BF = mtl::simd::batch<float>;
    using BI = mtl::simd::batch<TestType>;
    constexpr std::size_t W = BF::size;
    using L = std::numeric_limits<TestType>;
    std::vector<TestType> values = {0, 1, 2, 3, 1000, (1 << 24) - 1, 1 << 24, (1 << 24) + 1,
                                    (1 << 24) + 3, 123456789, L::max(), L::max() - 64};
    if constexpr (std::is_signed_v<TestType>)
        for (TestType v : {-1, -(1 << 24) - 1, -123456789, L::min(), L::min() + 1})
            values.push_back(v);
    else
        for (TestType v : {0x80000000u, 0x80000001u, 0xFFFFFF80u, 0xFFFFFF7Fu})
            values.push_back(v);
    while (values.size() % W != 0) values.push_back(7);
    for (std::size_t i = 0; i < values.size(); i += W) {
        float out[64];
        BF::convert(BI::load_unaligned(values.data() + i)).store_unaligned(out);
        for (std::size_t l = 0; l < W; ++l) {
            INFO("value " << values[i + l]);
            CHECK(out[l] == static_cast<float>(values[i + l]));
        }
    }
}
