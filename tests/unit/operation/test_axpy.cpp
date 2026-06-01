// Tests for mtl::axpy and the vectorized mtl::scale path (#86), exercised
// through dense_vector (which routes to the SIMD/BLAS L1 kernels).
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <mtl/operation/axpy.hpp>
#include <mtl/operation/scale.hpp>
#include <mtl/operation/dot.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/vec/dense_vector.hpp>

#include <cmath>
#include <cstddef>

using Catch::Approx;

TEMPLATE_TEST_CASE("axpy: y += alpha*x on dense_vector", "[operation][l1]", float, double) {
    const std::size_t sizes[] = {0, 1, 7, 16, 31, 100, 257};
    const TestType alpha = TestType(3);
    for (std::size_t n : sizes) {
        mtl::vec::dense_vector<TestType> x(n), y(n);
        for (std::size_t i = 0; i < n; ++i) { x(i) = TestType(i % 7) - 3; y(i) = TestType(i % 4); }
        mtl::axpy(alpha, x, y);
        for (std::size_t i = 0; i < n; ++i)
            CHECK(y(i) == Approx(alpha * (TestType(i % 7) - 3) + TestType(i % 4)));
    }
}

TEMPLATE_TEST_CASE("scale: x *= alpha on dense_vector", "[operation][l1]", float, double) {
    const std::size_t sizes[] = {0, 1, 7, 16, 33, 100, 257};
    const TestType alpha = TestType(-2);
    for (std::size_t n : sizes) {
        mtl::vec::dense_vector<TestType> x(n);
        for (std::size_t i = 0; i < n; ++i) x(i) = TestType(i % 9) - 4;
        mtl::scale(alpha, x);
        for (std::size_t i = 0; i < n; ++i) CHECK(x(i) == Approx(alpha * (TestType(i % 9) - 4)));
    }
}

// dot / two_norm now route through the SIMD L1 path for dense_vector; confirm
// they are still numerically correct end-to-end.
TEMPLATE_TEST_CASE("dot and two_norm via dense_vector are correct", "[operation][l1]", float, double) {
    const std::size_t n = 257;
    mtl::vec::dense_vector<TestType> a(n), b(n);
    TestType dotref = 0, ssq = 0;
    for (std::size_t i = 0; i < n; ++i) {
        a(i) = TestType(i % 5) - 2; b(i) = TestType(i % 3) - 1;
        dotref += a(i) * b(i); ssq += a(i) * a(i);
    }
    CHECK(mtl::dot(a, b) == Approx(dotref));
    CHECK(mtl::dot_real(a, b) == Approx(dotref));
    CHECK(mtl::two_norm(a) == Approx(std::sqrt(ssq)));
}
