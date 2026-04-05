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
#include <mtl/operation/cholesky.hpp>
#include <mtl/operation/operators.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/generators/randspd.hpp>
#include <mtl/generators/lehmer.hpp>
#include <mtl/generators/moler.hpp>
#include <mtl/generators/minij.hpp>

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

double cholesky_solve_report(const char* matrix_name, int n,
                             const mat::dense2D<double>& A,
                             const vec::dense_vector<double>& b,
                             double tol) {
    auto L = copy_matrix(A);
    int info = cholesky_factor(L);
    REQUIRE(info == 0);

    vec::dense_vector<double> x(A.num_rows());
    cholesky_solve(L, x, b);

    auto r = A * x;
    double res_norm = 0.0;
    for (std::size_t i = 0; i < A.num_rows(); ++i) {
        double d = r(i) - b(i);
        res_norm += d * d;
    }
    res_norm = std::sqrt(res_norm);
    double A_norm = frobenius_norm(A);
    double x_norm = two_norm(x);
    double be = (A_norm * x_norm > 0.0) ? res_norm / (A_norm * x_norm) : res_norm;

    std::cout << std::left << std::setw(12) << matrix_name
              << "  n=" << std::setw(6) << n
              << "  backward_err=" << std::scientific << std::setprecision(3) << be
              << "  tol=" << tol
              << (be < tol ? "  PASS" : "  FAIL")
              << std::defaultfloat << std::endl;
    return be;
}

} // anonymous namespace

TEST_CASE("Cholesky regression: randspd (kappa=100)", "[regression][dense][cholesky]") {
    auto n = GENERATE(100, 500, 1000);
    auto A = generators::randspd<double>(static_cast<std::size_t>(n), 100.0);
    vec::dense_vector<double> x_exact(n, 1.0);
    auto b = A * x_exact;
    double tol = double(n) * std::numeric_limits<double>::epsilon();
    double be = cholesky_solve_report("randspd", n, A, b, tol);
    REQUIRE(be < tol);
}

TEST_CASE("Cholesky regression: Lehmer matrix", "[regression][dense][cholesky]") {
    auto n = GENERATE(100, 500, 1000);
    auto A = materialize(generators::lehmer<double>(n));
    vec::dense_vector<double> x_exact(n, 1.0);
    auto b = A * x_exact;
    double tol = double(n) * std::numeric_limits<double>::epsilon();
    double be = cholesky_solve_report("Lehmer", n, A, b, tol);
    REQUIRE(be < tol);
}

TEST_CASE("Cholesky regression: Moler matrix", "[regression][dense][cholesky]") {
    auto n = GENERATE(100, 500);
    auto A = generators::moler<double>(n);
    vec::dense_vector<double> x_exact(n, 1.0);
    auto b = A * x_exact;
    double tol = double(n) * 1000.0 * std::numeric_limits<double>::epsilon();
    double be = cholesky_solve_report("Moler", n, A, b, tol);
    REQUIRE(be < tol);
}

TEST_CASE("Cholesky regression: Minij matrix (SPD)", "[regression][dense][cholesky]") {
    auto n = GENERATE(100, 500, 1000);
    auto A = materialize(generators::minij<double>(n));
    vec::dense_vector<double> x_exact(n, 1.0);
    auto b = A * x_exact;
    double tol = double(n) * std::numeric_limits<double>::epsilon();
    double be = cholesky_solve_report("Minij", n, A, b, tol);
    REQUIRE(be < tol);
}
