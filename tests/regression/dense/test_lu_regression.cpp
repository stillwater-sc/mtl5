#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/lu.hpp>
#include <mtl/operation/operators.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/generators/frank.hpp>
#include <mtl/generators/moler.hpp>
#include <mtl/generators/lehmer.hpp>

using namespace mtl;

namespace {

/// Materialize an implicit generator into a dense2D matrix.
template <typename Gen>
mat::dense2D<typename Gen::value_type> materialize(const Gen& g) {
    using V = typename Gen::value_type;
    std::size_t n = g.num_rows();
    std::size_t m = g.num_cols();
    mat::dense2D<V> A(n, m);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < m; ++j)
            A(i, j) = g(i, j);
    return A;
}

/// Copy a dense2D (for LU which modifies in place).
mat::dense2D<double> copy_matrix(const mat::dense2D<double>& A) {
    std::size_t n = A.num_rows(), m = A.num_cols();
    mat::dense2D<double> B(n, m);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < m; ++j)
            B(i, j) = A(i, j);
    return B;
}

/// Compute relative backward error: ||Ax - b|| / (||A|| * ||x||)
double backward_error(const mat::dense2D<double>& A,
                      const vec::dense_vector<double>& x,
                      const vec::dense_vector<double>& b) {
    auto r = A * x;
    double res_norm = 0.0;
    for (std::size_t i = 0; i < b.size(); ++i) {
        double d = r(i) - b(i);
        res_norm += d * d;
    }
    res_norm = std::sqrt(res_norm);

    double A_norm = frobenius_norm(A);
    double x_norm = two_norm(x);

    if (A_norm * x_norm == 0.0) return res_norm;
    return res_norm / (A_norm * x_norm);
}

/// LU solve helper: factor and solve Ax = b, return backward error.
double lu_solve_and_check(const mat::dense2D<double>& A,
                          const vec::dense_vector<double>& b) {
    std::size_t n = A.num_rows();
    auto LU = copy_matrix(A);

    std::vector<std::size_t> pivot;
    int info = lu_factor(LU, pivot);
    REQUIRE(info == 0);

    vec::dense_vector<double> x(n);
    lu_solve(LU, pivot, x, b);

    return backward_error(A, x, b);
}

} // anonymous namespace

TEST_CASE("LU regression: Frank matrix", "[regression][dense][lu]") {
    auto n = GENERATE(100, 500, 1000);
    CAPTURE(n);

    // Frank returns dense2D directly
    auto A = generators::frank<double>(n);

    vec::dense_vector<double> x_exact(n, 1.0);
    auto b = A * x_exact;

    double be = lu_solve_and_check(A, b);
    double tol = static_cast<double>(n) * std::numeric_limits<double>::epsilon();
    REQUIRE(be < tol);
}

TEST_CASE("LU regression: Moler matrix", "[regression][dense][lu]") {
    auto n = GENERATE(100, 500);
    CAPTURE(n);

    // Moler returns dense2D directly
    auto A = generators::moler<double>(n);

    vec::dense_vector<double> x_exact(n, 1.0);
    auto b = A * x_exact;

    double be = lu_solve_and_check(A, b);
    // Moler is moderately ill-conditioned
    double tol = static_cast<double>(n) * 1000.0 * std::numeric_limits<double>::epsilon();
    REQUIRE(be < tol);
}

TEST_CASE("LU regression: Lehmer matrix", "[regression][dense][lu]") {
    auto n = GENERATE(100, 500, 1000);
    CAPTURE(n);

    // Lehmer is an implicit generator — materialize it
    auto A = materialize(generators::lehmer<double>(n));

    vec::dense_vector<double> x_exact(n, 1.0);
    auto b = A * x_exact;

    double be = lu_solve_and_check(A, b);
    double tol = static_cast<double>(n) * std::numeric_limits<double>::epsilon();
    REQUIRE(be < tol);
}
