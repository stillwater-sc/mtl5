#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
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
#include <mtl/generators/pascal.hpp>
#include <mtl/generators/hilbert.hpp>

using namespace mtl;

namespace {

template <typename Gen>
mat::dense2D<typename Gen::value_type> materialize(const Gen& g) {
    using V = typename Gen::value_type;
    auto n = g.num_rows(), m = g.num_cols();
    mat::dense2D<V> A(n, m);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < m; ++j)
            A(i, j) = g(i, j);
    return A;
}

mat::dense2D<double> copy_matrix(const mat::dense2D<double>& A) {
    auto n = A.num_rows(), m = A.num_cols();
    mat::dense2D<double> B(n, m);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < m; ++j)
            B(i, j) = A(i, j);
    return B;
}

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

/// Solve Ax=b via LU, report backward error, return it for assertion.
double lu_solve_report(const char* matrix_name, int n,
                       const mat::dense2D<double>& A,
                       const vec::dense_vector<double>& b,
                       double tol) {
    auto LU = copy_matrix(A);
    std::vector<std::size_t> pivot;
    int info = lu_factor(LU, pivot);
    REQUIRE(info == 0);
    vec::dense_vector<double> x(A.num_rows());
    lu_solve(LU, pivot, x, b);
    double be = backward_error(A, x, b);

    std::cout << std::left << std::setw(12) << matrix_name
              << "  n=" << std::setw(6) << n
              << "  backward_err=" << std::scientific << std::setprecision(3) << be
              << "  tol=" << tol
              << (be < tol ? "  PASS" : "  FAIL")
              << std::defaultfloat << std::endl;
    return be;
}

} // anonymous namespace

TEST_CASE("LU regression: Frank matrix", "[regression][dense][lu]") {
    auto n = GENERATE(100, 500, 1000);
    auto A = generators::frank<double>(n);
    vec::dense_vector<double> x_exact(n, 1.0);
    auto b = A * x_exact;
    double tol = double(n) * std::numeric_limits<double>::epsilon();
    double be = lu_solve_report("Frank", n, A, b, tol);
    REQUIRE(be < tol);
}

TEST_CASE("LU regression: Moler matrix", "[regression][dense][lu]") {
    auto n = GENERATE(100, 500);
    auto A = generators::moler<double>(n);
    vec::dense_vector<double> x_exact(n, 1.0);
    auto b = A * x_exact;
    double tol = double(n) * 1000.0 * std::numeric_limits<double>::epsilon();
    double be = lu_solve_report("Moler", n, A, b, tol);
    REQUIRE(be < tol);
}

TEST_CASE("LU regression: Lehmer matrix (SPD)", "[regression][dense][lu]") {
    auto n = GENERATE(100, 500, 1000);
    auto A = materialize(generators::lehmer<double>(n));
    vec::dense_vector<double> x_exact(n, 1.0);
    auto b = A * x_exact;
    double tol = double(n) * std::numeric_limits<double>::epsilon();
    double be = lu_solve_report("Lehmer", n, A, b, tol);
    REQUIRE(be < tol);
}

TEST_CASE("LU regression: Pascal matrix", "[regression][dense][lu]") {
    auto n = GENERATE(50, 100);
    auto A = generators::pascal<double>(n);
    vec::dense_vector<double> x_exact(n, 1.0);
    auto b = A * x_exact;
    double tol = double(n) * 1e8 * std::numeric_limits<double>::epsilon();
    double be = lu_solve_report("Pascal", n, A, b, tol);
    REQUIRE(be < tol);
}

TEST_CASE("LU regression: Hilbert matrix (ill-conditioned)", "[regression][dense][lu]") {
    // Hilbert is notoriously ill-conditioned; cond(H_n) ~ O((1+sqrt(2))^{4n})
    auto n = GENERATE(50, 100);
    auto A = materialize(generators::hilbert<double>(n));
    vec::dense_vector<double> x_exact(n, 1.0);
    auto b = A * x_exact;
    // Very relaxed tolerance due to extreme ill-conditioning
    double tol = double(n) * 1e10 * std::numeric_limits<double>::epsilon();
    double be = lu_solve_report("Hilbert", n, A, b, tol);
    REQUIRE(be < tol);
}
