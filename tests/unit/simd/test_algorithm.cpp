// Tests for vectorized L1 kernels (#86): reduce_dot, reduce_sum_squares, axpy, scal.
// Integer-valued data keeps reductions exact regardless of accumulation order
// (SIMD multi-accumulator vs scalar), so we can compare exactly; sqrt uses Approx.
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <mtl/simd/algorithm.hpp>

#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

using Catch::Approx;

namespace {
const std::size_t kLengths[] = {0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 33, 64, 100, 257};
}

TEMPLATE_TEST_CASE("reduce_dot matches scalar reference", "[simd][l1]", float, double) {
    for (std::size_t n : kLengths) {
        std::vector<TestType> a(n), b(n);
        TestType ref = 0;
        for (std::size_t i = 0; i < n; ++i) {
            a[i] = TestType((i % 7) - 3);     // small integers
            b[i] = TestType((i % 5) - 2);
            ref += a[i] * b[i];
        }
        CHECK(mtl::simd::reduce_dot<TestType>(a.data(), b.data(), n) == ref);
    }
}

TEMPLATE_TEST_CASE("reduce_sum_squares matches scalar reference", "[simd][l1]", float, double) {
    for (std::size_t n : kLengths) {
        std::vector<TestType> a(n);
        TestType ref = 0;
        for (std::size_t i = 0; i < n; ++i) { a[i] = TestType((i % 9) - 4); ref += a[i] * a[i]; }
        CHECK(mtl::simd::reduce_sum_squares<TestType>(a.data(), n) == ref);
    }
}

TEMPLATE_TEST_CASE("axpy matches scalar reference over all lengths", "[simd][l1]", float, double) {
    const TestType alpha = TestType(2.5);
    for (std::size_t n : kLengths) {
        std::vector<TestType> x(n), y(n), ref(n);
        for (std::size_t i = 0; i < n; ++i) {
            x[i] = TestType(i % 11) - 5;
            y[i] = TestType(i % 6) - 3;
            ref[i] = alpha * x[i] + y[i];
        }
        mtl::simd::axpy<TestType>(alpha, x.data(), y.data(), n);
        for (std::size_t i = 0; i < n; ++i) CHECK(y[i] == Approx(ref[i]));
    }
}

TEMPLATE_TEST_CASE("scal matches scalar reference over all lengths", "[simd][l1]", float, double) {
    const TestType alpha = TestType(-1.5);
    for (std::size_t n : kLengths) {
        std::vector<TestType> x(n), ref(n);
        for (std::size_t i = 0; i < n; ++i) { x[i] = TestType(i % 13) - 6; ref[i] = alpha * x[i]; }
        mtl::simd::scal<TestType>(alpha, x.data(), n);
        for (std::size_t i = 0; i < n; ++i) CHECK(x[i] == Approx(ref[i]));
    }
}

// -- Issue 878 (QA sweep #783) --------------------------------------------------
//
// The cases above use small integers, whose reductions are exact in any order.
// The ones below use dyadic fractions (k / 64 with |k| < 64): products are
// multiples of 2^-12 below 1, and every partial sum over the lengths here is
// a multiple of 2^-12 below 2^9, exact in float's 24 bits. So the SIMD body,
// its scalar tail and a plain loop must agree exactly, with no Approx, while
// the values are not integers. They also cover NaN and Inf in the body and in
// the tail, and pointers that are not vector-aligned.

namespace {

template <typename T>
T dyadic(std::size_t i, std::size_t salt) {
    return T(static_cast<int>((i * 37 + salt * 11) % 127) - 63) / T(64);
}

} // namespace

TEMPLATE_TEST_CASE("reduce_dot and reduce_sum_squares on non-integer data, body and tail exact",
                   "[simd][l1][878]", float, double) {
    using T = TestType;
    for (std::size_t n : kLengths) {
        for (std::size_t offset : {0u, 1u, 3u}) {   // offsets 1 and 3: not vector-aligned
            INFO("n = " << n << ", offset = " << offset);
            std::vector<T> abuf(n + offset), bbuf(n + offset);
            T* a = abuf.data() + offset;
            T* b = bbuf.data() + offset;
            T dot = 0, sq = 0;
            for (std::size_t i = 0; i < n; ++i) {
                a[i] = dyadic<T>(i, 1);
                b[i] = dyadic<T>(i, 2);
                dot += a[i] * b[i];
                sq += a[i] * a[i];
            }
            CHECK(mtl::simd::reduce_dot<T>(a, b, n) == dot);
            CHECK(mtl::simd::reduce_sum_squares<T>(a, n) == sq);
        }
    }
}

TEMPLATE_TEST_CASE("reduce_dot and reduce_sum_squares propagate NaN and Inf from body and tail",
                   "[simd][l1][878]", float, double) {
    using T = TestType;
    const T qnan = std::numeric_limits<T>::quiet_NaN();
    const T inf = std::numeric_limits<T>::infinity();
    for (std::size_t n : {1u, 3u, 9u, 17u, 33u, 100u}) {
        // Position 0 is in the body when n >= the SIMD width; n - 1 is in the
        // tail whenever n is not a multiple of it.
        for (std::size_t at : {std::size_t{0}, n / 2, n - 1}) {
            INFO("n = " << n << ", special at " << at);
            std::vector<T> a(n), b(n);
            for (std::size_t i = 0; i < n; ++i) { a[i] = dyadic<T>(i, 1); b[i] = T(1); }
            a[at] = qnan;
            CHECK(std::isnan(mtl::simd::reduce_dot<T>(a.data(), b.data(), n)));
            CHECK(std::isnan(mtl::simd::reduce_sum_squares<T>(a.data(), n)));
            a[at] = inf;
            CHECK(mtl::simd::reduce_dot<T>(a.data(), b.data(), n) == inf);
            CHECK(mtl::simd::reduce_sum_squares<T>(a.data(), n) == inf);
            b[at] = T(0);   // inf * 0 is NaN
            CHECK(std::isnan(mtl::simd::reduce_dot<T>(a.data(), b.data(), n)));
        }
    }
}

TEMPLATE_TEST_CASE("scalar_fma keeps NaN and Inf: its reduce_max runs over identical lanes",
                   "[simd][batch][878]", float, double) {
    // The tail's scalar_fma broadcasts to a batch, applies fma and takes a lane
    // back with reduce_max. Every lane holds the same value, so a NaN must come
    // back as NaN on every backend (the Highway build runs this too).
    using T = TestType;
    const T qnan = std::numeric_limits<T>::quiet_NaN();
    const T inf = std::numeric_limits<T>::infinity();
    CHECK(std::isnan(mtl::simd::scalar_fma<T>(qnan, T(1), T(1))));
    CHECK(std::isnan(mtl::simd::scalar_fma<T>(T(1), T(1), qnan)));
    CHECK(std::isnan(mtl::simd::scalar_fma<T>(inf, T(0), T(1))));
    CHECK(std::isnan(mtl::simd::scalar_fma<T>(inf, T(1), -inf)));
    CHECK(mtl::simd::scalar_fma<T>(inf, T(2), T(1)) == inf);
    CHECK(mtl::simd::scalar_fma<T>(T(0.5), T(0.25), T(-0.125)) == T(0));
}
