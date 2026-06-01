// Tests for vectorized L1 kernels (#86): reduce_dot, reduce_sum_squares, axpy, scal.
// Integer-valued data keeps reductions exact regardless of accumulation order
// (SIMD multi-accumulator vs scalar), so we can compare exactly; sqrt uses Approx.
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <mtl/simd/algorithm.hpp>

#include <cstddef>
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
