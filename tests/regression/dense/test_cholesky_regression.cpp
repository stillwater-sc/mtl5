#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <cmath>
#include <cstddef>
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

double cholesky_solve_check(const mat::dense2D<double>& A,
                            const vec::dense_vector<double>& b) {
    auto L = copy_matrix(A);
    int info = cholesky_factor(L);
    REQUIRE(info == 0);

    auto n = A.num_rows();
    vec::dense_vector<double> x(n);
    cholesky_solve(L, x, b);

    auto r = A * x;
    double res_norm = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double d = r(i) - b(i);
        res_norm += d * d;
    }
    res_norm = std::sqrt(res_norm);
    double A_norm = frobenius_norm(A);
    double x_norm = two_norm(x);
    if (A_norm * x_norm == 0.0) return res_norm;
    return res_norm / (A_norm * x_norm);
}

} // anonymous namespace

TEST_CASE("Cholesky regression: randspd (kappa=100)", "[regression][dense][cholesky]") {
    auto n = GENERATE(100, 500, 1000);
    CAPTURE(n);

    // randspd(n, kappa) generates SPD with condition number ~kappa
    auto A = generators::randspd<double>(static_cast<std::size_t>(n), 100.0);
    vec::dense_vector<double> x_exact(n, 1.0);
    auto b = A * x_exact;

    double be = cholesky_solve_check(A, b);
    REQUIRE(be < double(n) * std::numeric_limits<double>::epsilon());
}

TEST_CASE("Cholesky regression: Lehmer matrix", "[regression][dense][cholesky]") {
    auto n = GENERATE(100, 500, 1000);
    CAPTURE(n);

    auto A = materialize(generators::lehmer<double>(n));
    vec::dense_vector<double> x_exact(n, 1.0);
    auto b = A * x_exact;

    double be = cholesky_solve_check(A, b);
    REQUIRE(be < double(n) * std::numeric_limits<double>::epsilon());
}

TEST_CASE("Cholesky regression: Moler matrix", "[regression][dense][cholesky]") {
    auto n = GENERATE(100, 500);
    CAPTURE(n);

    auto A = generators::moler<double>(n);
    vec::dense_vector<double> x_exact(n, 1.0);
    auto b = A * x_exact;

    double be = cholesky_solve_check(A, b);
    REQUIRE(be < double(n) * 1000.0 * std::numeric_limits<double>::epsilon());
}
