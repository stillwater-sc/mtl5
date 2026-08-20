// Integer lanes for batch<> and the L1 kernels -- #451 phase 0.
//
// The whole point of these tests is that they can assert EXACT equality with a
// closed-form reference and never mention accumulation order. On floating lanes
// a reduction's value depends on how the four independent accumulators, the
// horizontal reduce and the scalar tail happen to group the additions, so the
// float tests either pick order-independent data or fall back to Approx. On
// integer lanes addition is associative and commutative mod 2^32, so:
//
//   * the SIMD result equals the scalar result bit for bit,
//   * splitting a reduction anywhere and adding the pieces gives the same value,
//   * and the answer does not change with lane width, backend or unroll factor.
//
// The reference is computed in uint64_t and truncated: two's-complement multiply
// and add agree with unsigned multiply and add in the low bits, and 2^32 divides
// 2^64, so wrapping the uint64 accumulator cannot disturb the low 32 bits.
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include <mtl/simd/algorithm.hpp>
#include <mtl/simd/batch.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

namespace {

/// Exact mod-2^32 truncation of a uint64 accumulator back into the lane type.
template <typename T>
T wrap32(std::uint64_t acc) {
    return static_cast<T>(static_cast<std::uint32_t>(acc));
}

/// Lane-type-agnostic view of a value as its 32 raw bits.
template <typename T>
std::uint64_t bits(T v) {
    return static_cast<std::uint64_t>(static_cast<std::uint32_t>(v));
}

// Data with a deliberately wide dynamic range: the products below overflow an
// int32 within a few terms, which is exactly the regime the wrapping contract
// has to define rather than avoid.
template <typename T>
T big_a(std::size_t i) { return static_cast<T>(static_cast<std::uint32_t>(0x9E3779B9u * (i + 1))); }
template <typename T>
T big_b(std::size_t i) { return static_cast<T>(static_cast<std::uint32_t>(0x85EBCA77u * (i + 3))); }

const std::size_t kLengths[] = {0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 33, 64, 100, 257};

/// Closed-form mod-2^32 reference for the first `n` products.
template <typename T>
std::uint64_t ref_dot(const T* a, const T* b, std::size_t n) {
    std::uint64_t acc = 0;
    for (std::size_t i = 0; i < n; ++i) acc += bits(a[i]) * bits(b[i]);
    return acc;
}

/// `reduce_dot` at a COMPILE-TIME-CONSTANT length, checked against both the
/// closed form and the same call with the length made opaque.
///
/// The second comparison is the one that earns its place. A constant trip count
/// is a different code path -- the compiler may unroll it, specialize the loop
/// bounds, or vectorize it differently from the runtime-length version -- and
/// the two must still agree. GCC 13 at -O3 -march=x86-64-v3 did not: it fused
/// the four interleaved integer accumulator chains into one contiguous
/// reduction and advanced three of them at half the required stride, so the
/// constant-length answer was silently wrong while the runtime-length answer
/// was right. Every test that reached reduce_dot through a runtime length
/// passed. See the note in simd/algorithm.hpp.
template <typename T, std::size_t N>
bool const_length_ok(const T* a, const T* b) {
    const T folded = mtl::simd::reduce_dot<T>(a, b, N);       // N is constant here
    volatile std::size_t opaque = N;                          // ... and not here
    const T runtime = mtl::simd::reduce_dot<T>(a, b, opaque);
    return folded == runtime && folded == wrap32<T>(ref_dot(a, b, N));
}

template <typename T, std::size_t N>
bool const_length_sq_ok(const T* a) {
    const T folded = mtl::simd::reduce_sum_squares<T>(a, N);
    volatile std::size_t opaque = N;
    const T runtime = mtl::simd::reduce_sum_squares<T>(a, opaque);
    return folded == runtime && folded == wrap32<T>(ref_dot(a, a, N));
}

template <typename T, std::size_t... Ns>
std::vector<std::size_t> failing_dot_lengths(const T* a, const T* b) {
    std::vector<std::size_t> bad;
    ((const_length_ok<T, Ns>(a, b) ? void() : bad.push_back(Ns)), ...);
    return bad;
}

template <typename T, std::size_t... Ns>
std::vector<std::size_t> failing_sq_lengths(const T* a) {
    std::vector<std::size_t> bad;
    ((const_length_sq_ok<T, Ns>(a) ? void() : bad.push_back(Ns)), ...);
    return bad;
}

/// Is `batch<T> / batch<T>` a valid expression?
///
/// Written as a variable TEMPLATE rather than inline in the assertion on
/// purpose: a requires-expression whose operands are non-dependent gets its
/// requirements checked eagerly, so `static_assert(!requires(batch<int> a,
/// batch<int> b){ a / b; })` is a hard "no match for operator/" on both GCC and
/// Clang instead of evaluating to false. Parameterizing on T makes the
/// expression dependent, which is what lets the detector answer rather than
/// abort.
template <typename T>
inline constexpr bool has_lane_division =
    requires(mtl::simd::batch<T> a, mtl::simd::batch<T> b) { a / b; };

} // namespace

TEST_CASE("is_lane_v admits exactly the supported lane types", "[simd][integer]") {
    using namespace mtl::simd;
    STATIC_REQUIRE(is_lane_v<float>);
    STATIC_REQUIRE(is_lane_v<double>);
    STATIC_REQUIRE(is_lane_v<std::int32_t>);
    STATIC_REQUIRE(is_lane_v<std::uint32_t>);

    // Not yet: 8- and 16-bit lanes are worth having only WITH the widening
    // accumulate they exist for, and 64-bit integer multiply is emulated on x86
    // before AVX512DQ (#451 phases 2-3).
    STATIC_REQUIRE_FALSE(is_lane_v<std::int8_t>);
    STATIC_REQUIRE_FALSE(is_lane_v<std::int16_t>);
    STATIC_REQUIRE_FALSE(is_lane_v<std::int64_t>);
    STATIC_REQUIRE_FALSE(is_lane_v<std::uint64_t>);
    STATIC_REQUIRE_FALSE(is_lane_v<bool>);

    // long double is NOT a lane, despite being a floating-point type. Highway
    // has no long double vector, so batch<long double> is a hard compile error
    // there -- and because this trait gates SimdDenseVector, admitting it sent
    // dot(dense_vector<long double>, ...) into reduce_dot<long double> and broke
    // a build that had worked, since the older BlasDenseVector gate (float and
    // double only) kept it on the generic loop.
    STATIC_REQUIRE_FALSE(is_lane_v<long double>);

    STATIC_REQUIRE(is_integer_lane_v<std::int32_t>);
    STATIC_REQUIRE(is_integer_lane_v<std::uint32_t>);
    STATIC_REQUIRE_FALSE(is_integer_lane_v<double>);
}

TEST_CASE("division is available on floating lanes and absent on integer lanes",
          "[simd][integer]") {
    // The constrained overload, stated as the claim it is. #450 made integer
    // division a compile error one level up (Field excludes the integers, the
    // factorizations require FieldMatrix); this pins the same answer underneath,
    // so generic code cannot reach a truncating quotient through batch<>.
    STATIC_REQUIRE(has_lane_division<float>);
    STATIC_REQUIRE(has_lane_division<double>);
    STATIC_REQUIRE_FALSE(has_lane_division<std::int32_t>);
    STATIC_REQUIRE_FALSE(has_lane_division<std::uint32_t>);
}

TEMPLATE_TEST_CASE("integer batch: lane count and load/store round-trip",
                   "[simd][integer]", std::int32_t, std::uint32_t) {
    using B = mtl::simd::batch<TestType>;
    constexpr std::size_t N = B::size;
    STATIC_REQUIRE(N >= 1);
    STATIC_REQUIRE(mtl::simd::width<TestType> == N);

    // Same guard the float tests carry: an -DMTL5_WITH_HIGHWAY=ON build that
    // silently resolves to the size == 1 fallback would pass everything here
    // while covering nothing. 32-bit integer lanes are in the SSE2 baseline.
#if defined(MTL5_HAS_HIGHWAY) && (defined(__x86_64__) || defined(_M_X64))
    STATIC_REQUIRE(N > 1);
#endif

    alignas(64) TestType in[64], out[64];
    for (std::size_t i = 0; i < N; ++i) in[i] = big_a<TestType>(i);

    B::load_unaligned(in).store_unaligned(out);
    for (std::size_t i = 0; i < N; ++i) CHECK(out[i] == in[i]);
    B::load_aligned(in).store_aligned(out);
    for (std::size_t i = 0; i < N; ++i) CHECK(out[i] == in[i]);

    B(TestType(7)).store_unaligned(out);
    for (std::size_t i = 0; i < N; ++i) CHECK(out[i] == TestType(7));

    B{}.store_unaligned(out);                      // default ctor zeroes
    for (std::size_t i = 0; i < N; ++i) CHECK(out[i] == TestType(0));
}

TEMPLATE_TEST_CASE("integer batch arithmetic wraps mod 2^32, exactly",
                   "[simd][integer]", std::int32_t, std::uint32_t) {
    using B = mtl::simd::batch<TestType>;
    constexpr std::size_t N = B::size;

    alignas(64) TestType a[64], b[64], c[64], r[64];
    for (std::size_t i = 0; i < N; ++i) {
        a[i] = big_a<TestType>(i);                 // ~2^31 magnitudes, so every
        b[i] = big_b<TestType>(i);                 // product below overflows
        c[i] = static_cast<TestType>(static_cast<std::uint32_t>(0xDEADBEEFu + i));
    }
    const auto va = B::load_aligned(a), vb = B::load_aligned(b), vc = B::load_aligned(c);

    (va + vb).store_aligned(r);
    for (std::size_t i = 0; i < N; ++i) CHECK(bits(r[i]) == ((bits(a[i]) + bits(b[i])) & 0xFFFFFFFFu));
    (va - vb).store_aligned(r);
    for (std::size_t i = 0; i < N; ++i) CHECK(bits(r[i]) == ((bits(a[i]) - bits(b[i])) & 0xFFFFFFFFu));
    (va * vb).store_aligned(r);
    for (std::size_t i = 0; i < N; ++i) CHECK(bits(r[i]) == ((bits(a[i]) * bits(b[i])) & 0xFFFFFFFFu));

    // fma on integer lanes is a multiply and an add -- nothing is fused, nothing
    // rounds, and the result is the exact product-sum reduced mod 2^32. So it
    // must agree with the separate operations bit for bit, which is a claim the
    // floating-point fma cannot make.
    fma(va, vb, vc).store_aligned(r);
    for (std::size_t i = 0; i < N; ++i)
        CHECK(bits(r[i]) == ((bits(a[i]) * bits(b[i]) + bits(c[i])) & 0xFFFFFFFFu));

    alignas(64) TestType mul_then_add[64];
    (va * vb + vc).store_aligned(mul_then_add);
    for (std::size_t i = 0; i < N; ++i) CHECK(mul_then_add[i] == r[i]);
}

TEMPLATE_TEST_CASE("integer batch horizontal reductions", "[simd][integer]",
                   std::int32_t, std::uint32_t) {
    using B = mtl::simd::batch<TestType>;
    constexpr std::size_t N = B::size;

    alignas(64) TestType a[64];
    std::uint64_t sum = 0;
    TestType lo = std::numeric_limits<TestType>::max();
    TestType hi = std::numeric_limits<TestType>::lowest();
    for (std::size_t i = 0; i < N; ++i) {
        // Keep magnitudes modest here: min/max are order-independent anyway, but
        // the sum is compared against the wrapped reference, so state it plainly.
        a[i] = static_cast<TestType>(static_cast<std::uint32_t>(i * 37u + 11u));
        sum += bits(a[i]);
        if (a[i] < lo) lo = a[i];
        if (a[i] > hi) hi = a[i];
    }
    const auto v = B::load_aligned(a);
    CHECK(reduce_add(v) == wrap32<TestType>(sum));
    CHECK(reduce_min(v) == lo);
    CHECK(reduce_max(v) == hi);
}

TEMPLATE_TEST_CASE("integer reduce_dot equals the exact mod-2^32 sum",
                   "[simd][integer][l1]", std::int32_t, std::uint32_t) {
    for (std::size_t n : kLengths) {
        std::vector<TestType> a(n), b(n);
        std::uint64_t acc = 0;
        for (std::size_t i = 0; i < n; ++i) {
            a[i] = big_a<TestType>(i);
            b[i] = big_b<TestType>(i);
            acc += bits(a[i]) * bits(b[i]);        // uint64 wrap keeps the low 32 bits
        }
        INFO("n=" << n);
        CHECK(mtl::simd::reduce_dot<TestType>(a.data(), b.data(), n) == wrap32<TestType>(acc));
    }
}

TEMPLATE_TEST_CASE("integer reductions agree at compile-time-constant lengths",
                   "[simd][integer][l1]", std::int32_t, std::uint32_t) {
    // Lengths chosen to straddle the SIMD width and the unroll factor on every
    // backend, and to include the two that GCC 13 miscompiled (137, 257).
    constexpr std::size_t kMax = 1031;
    std::vector<TestType> a(kMax), b(kMax);
    for (std::size_t i = 0; i < kMax; ++i) { a[i] = big_a<TestType>(i); b[i] = big_b<TestType>(i); }

    const auto bad_dot = failing_dot_lengths<TestType,
        1, 2, 3, 7, 8, 12, 13, 16, 17, 20, 25, 31, 33, 40, 64, 100,
        128, 137, 200, 256, 257, 512, 1031>(a.data(), b.data());
    for (std::size_t n : bad_dot)
        UNSCOPED_INFO("reduce_dot disagrees at compile-time-constant length " << n);
    CHECK(bad_dot.empty());

    const auto bad_sq = failing_sq_lengths<TestType,
        1, 2, 3, 7, 8, 12, 13, 16, 17, 20, 25, 31, 33, 40, 64, 100,
        128, 137, 200, 256, 257, 512, 1031>(a.data());
    for (std::size_t n : bad_sq)
        UNSCOPED_INFO("reduce_sum_squares disagrees at compile-time-constant length " << n);
    CHECK(bad_sq.empty());
}

TEMPLATE_TEST_CASE("integer reduce_dot is invariant under partitioning",
                   "[simd][integer][l1]", std::int32_t, std::uint32_t) {
    // The property that makes integer lanes a better testbed than float ones:
    // wrapping addition is associative, so cutting the reduction at ANY point
    // and adding the two halves reproduces the whole exactly. Every split is
    // checked, including ones that land mid-SIMD-body and leave both halves with
    // different tail lengths -- which is where a grouping-sensitive kernel would
    // disagree. The same argument is why a threaded integer reduction would be
    // bit-identical to the serial one.
    constexpr std::size_t n = 137;                 // prime: no split aligns nicely
    std::vector<TestType> a(n), b(n);
    for (std::size_t i = 0; i < n; ++i) { a[i] = big_a<TestType>(i); b[i] = big_b<TestType>(i); }

    const TestType whole = mtl::simd::reduce_dot<TestType>(a.data(), b.data(), n);
    for (std::size_t k = 0; k <= n; ++k) {
        const TestType lo = mtl::simd::reduce_dot<TestType>(a.data(), b.data(), k);
        const TestType hi = mtl::simd::reduce_dot<TestType>(a.data() + k, b.data() + k, n - k);
        INFO("split at k=" << k);
        CHECK(wrap32<TestType>(bits(lo) + bits(hi)) == whole);
    }
}

TEMPLATE_TEST_CASE("integer reduce_sum_squares equals the exact mod-2^32 sum",
                   "[simd][integer][l1]", std::int32_t, std::uint32_t) {
    for (std::size_t n : kLengths) {
        std::vector<TestType> a(n);
        std::uint64_t acc = 0;
        for (std::size_t i = 0; i < n; ++i) { a[i] = big_a<TestType>(i); acc += bits(a[i]) * bits(a[i]); }
        INFO("n=" << n);
        CHECK(mtl::simd::reduce_sum_squares<TestType>(a.data(), n) == wrap32<TestType>(acc));
    }
}

TEMPLATE_TEST_CASE("integer axpy and scal match the wrapping reference",
                   "[simd][integer][l1]", std::int32_t, std::uint32_t) {
    const TestType alpha = static_cast<TestType>(static_cast<std::uint32_t>(0xC2B2AE35u));
    for (std::size_t n : kLengths) {
        std::vector<TestType> x(n), y(n), yref(n), xs(n), xsref(n);
        for (std::size_t i = 0; i < n; ++i) {
            x[i] = big_a<TestType>(i);
            y[i] = big_b<TestType>(i);
            yref[i]  = wrap32<TestType>(bits(alpha) * bits(x[i]) + bits(y[i]));
            xs[i]    = x[i];
            xsref[i] = wrap32<TestType>(bits(alpha) * bits(x[i]));
        }
        mtl::simd::axpy<TestType>(alpha, x.data(), y.data(), n);
        mtl::simd::scal<TestType>(alpha, xs.data(), n);
        INFO("n=" << n);
        for (std::size_t i = 0; i < n; ++i) {
            CHECK(y[i] == yref[i]);
            CHECK(xs[i] == xsref[i]);
        }
    }
}
