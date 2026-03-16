#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstddef>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/sparse_solve.hpp>

using namespace mtl;

static double sparse_residual(
    const mat::compressed2D<double>& A,
    const vec::dense_vector<double>& x,
    const vec::dense_vector<double>& b)
{
    std::size_t n = b.size();
    double res = 0.0, bnorm = 0.0;
    const auto& starts = A.ref_major();
    const auto& indices = A.ref_minor();
    const auto& data = A.ref_data();
    for (std::size_t i = 0; i < n; ++i) {
        double Ax_i = 0.0;
        for (std::size_t k = starts[i]; k < starts[i + 1]; ++k)
            Ax_i += data[k] * x(indices[k]);
        double ri = Ax_i - b(i);
        res += ri * ri;
        bnorm += b(i) * b(i);
    }
    if (bnorm == 0.0) return std::sqrt(res);
    return std::sqrt(res / bnorm);
}

static double dense_residual(
    const mat::dense2D<double>& A,
    const vec::dense_vector<double>& x,
    const vec::dense_vector<double>& b)
{
    std::size_t n = b.size();
    double res = 0.0, bnorm = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double Ax_i = 0.0;
        for (std::size_t j = 0; j < n; ++j)
            Ax_i += A(i, j) * x(j);
        double ri = Ax_i - b(i);
        res += ri * ri;
        bnorm += b(i) * b(i);
    }
    if (bnorm == 0.0) return std::sqrt(res);
    return std::sqrt(res / bnorm);
}

// ---- solve() dispatches to LU ----

TEST_CASE("solve() dispatches sparse LU", "[dispatch][lu]") {
    mat::compressed2D<double> A(3, 3);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 3.0; ins[0][1] << 1.0;
        ins[1][1] << 4.0; ins[1][2] << 2.0;
        ins[2][0] << 1.0; ins[2][2] << 5.0;
    }
    vec::dense_vector<double> b = {4.0, 10.0, 11.0};
    vec::dense_vector<double> x(3, 0.0);

    solve(A, x, b);
    REQUIRE(sparse_residual(A, x, b) < 1e-12);
}

TEST_CASE("solve() dispatches dense LU", "[dispatch][lu]") {
    mat::dense2D<double> A(3, 3);
    A(0, 0) = 3.0; A(0, 1) = 1.0; A(0, 2) = 0.0;
    A(1, 0) = 0.0; A(1, 1) = 4.0; A(1, 2) = 2.0;
    A(2, 0) = 1.0; A(2, 1) = 0.0; A(2, 2) = 5.0;

    vec::dense_vector<double> b = {4.0, 10.0, 11.0};
    vec::dense_vector<double> x(3, 0.0);

    solve(A, x, b);
    REQUIRE(dense_residual(A, x, b) < 1e-12);
}

// ---- cholesky_solve_dispatch ----

TEST_CASE("cholesky_solve_dispatch on sparse SPD", "[dispatch][cholesky]") {
    std::size_t n = 10;
    mat::compressed2D<double> A(n, n);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        for (std::size_t i = 0; i < n; ++i) {
            ins[i][i] << 4.0;
            if (i + 1 < n) {
                ins[i][i + 1] << -1.0;
                ins[i + 1][i] << -1.0;
            }
        }
    }
    vec::dense_vector<double> b(n, 1.0);
    vec::dense_vector<double> x(n, 0.0);

    cholesky_solve_dispatch(A, x, b);
    REQUIRE(sparse_residual(A, x, b) < 1e-12);
}

TEST_CASE("cholesky_solve_dispatch on dense SPD", "[dispatch][cholesky]") {
    mat::dense2D<double> A(3, 3);
    A(0, 0) = 4.0;  A(0, 1) = -1.0; A(0, 2) = 0.0;
    A(1, 0) = -1.0; A(1, 1) = 4.0;  A(1, 2) = -1.0;
    A(2, 0) = 0.0;  A(2, 1) = -1.0; A(2, 2) = 4.0;

    vec::dense_vector<double> b = {1.0, 2.0, 3.0};
    vec::dense_vector<double> x(3, 0.0);

    cholesky_solve_dispatch(A, x, b);
    REQUIRE(dense_residual(A, x, b) < 1e-12);
}

// ---- lu_solve_dispatch ----

TEST_CASE("lu_solve_dispatch on sparse unsymmetric", "[dispatch][lu]") {
    mat::compressed2D<double> A(3, 3);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 2.0;  ins[0][1] << 3.0;  ins[0][2] << 1.0;
        ins[1][0] << 4.0;  ins[1][1] << 7.0;  ins[1][2] << 5.0;
        ins[2][0] << 6.0;  ins[2][1] << 18.0; ins[2][2] << 22.0;
    }
    vec::dense_vector<double> b = {6.0, 16.0, 46.0};
    vec::dense_vector<double> x(3, 0.0);

    lu_solve_dispatch(A, x, b);
    REQUIRE(sparse_residual(A, x, b) < 1e-12);
}

TEST_CASE("lu_solve_dispatch on dense unsymmetric", "[dispatch][lu]") {
    mat::dense2D<double> A(3, 3);
    A(0, 0) = 2.0;  A(0, 1) = 3.0;  A(0, 2) = 1.0;
    A(1, 0) = 4.0;  A(1, 1) = 7.0;  A(1, 2) = 5.0;
    A(2, 0) = 6.0;  A(2, 1) = 18.0; A(2, 2) = 22.0;

    vec::dense_vector<double> b = {6.0, 16.0, 46.0};
    vec::dense_vector<double> x(3, 0.0);

    lu_solve_dispatch(A, x, b);
    REQUIRE(dense_residual(A, x, b) < 1e-12);
}

// ---- qr_solve_dispatch ----

TEST_CASE("qr_solve_dispatch on sparse square", "[dispatch][qr]") {
    mat::compressed2D<double> A(3, 3);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 4.0;  ins[0][1] << -1.0;
        ins[1][0] << -1.0; ins[1][1] << 4.0;  ins[1][2] << -1.0;
        ins[2][1] << -1.0; ins[2][2] << 4.0;
    }
    vec::dense_vector<double> b = {1.0, 2.0, 3.0};
    vec::dense_vector<double> x(3, 0.0);

    qr_solve_dispatch(A, x, b);
    REQUIRE(sparse_residual(A, x, b) < 1e-12);
}

TEST_CASE("qr_solve_dispatch on dense square", "[dispatch][qr]") {
    mat::dense2D<double> A(3, 3);
    A(0, 0) = 4.0;  A(0, 1) = -1.0; A(0, 2) = 0.0;
    A(1, 0) = -1.0; A(1, 1) = 4.0;  A(1, 2) = -1.0;
    A(2, 0) = 0.0;  A(2, 1) = -1.0; A(2, 2) = 4.0;

    vec::dense_vector<double> b = {1.0, 2.0, 3.0};
    vec::dense_vector<double> x(3, 0.0);

    qr_solve_dispatch(A, x, b);
    REQUIRE(dense_residual(A, x, b) < 1e-12);
}

// ---- Consistency: dispatch produces same answer as explicit call ----

TEST_CASE("Dispatch matches explicit sparse Cholesky", "[dispatch][consistency]") {
    std::size_t n = 8;
    mat::compressed2D<double> A(n, n);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        for (std::size_t i = 0; i < n; ++i) {
            ins[i][i] << 4.0;
            if (i + 1 < n) {
                ins[i][i + 1] << -1.0;
                ins[i + 1][i] << -1.0;
            }
        }
    }
    vec::dense_vector<double> b(n, 1.0);

    vec::dense_vector<double> x_dispatch(n, 0.0);
    cholesky_solve_dispatch(A, x_dispatch, b);

    vec::dense_vector<double> x_explicit(n, 0.0);
    sparse::factorization::sparse_cholesky_solve(A, x_explicit, b,
        sparse::ordering::amd{});

    for (std::size_t i = 0; i < n; ++i)
        REQUIRE_THAT(x_dispatch(i), Catch::Matchers::WithinAbs(x_explicit(i), 1e-12));
}

TEST_CASE("Dispatch matches explicit sparse LU", "[dispatch][consistency]") {
    mat::compressed2D<double> A(4, 4);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 3.0; ins[0][1] << 1.0;
        ins[1][1] << 4.0; ins[1][2] << 2.0;
        ins[2][0] << 1.0; ins[2][2] << 5.0; ins[2][3] << 1.0;
        ins[3][1] << 1.0; ins[3][3] << 3.0;
    }
    vec::dense_vector<double> b = {4.0, 10.0, 12.0, 7.0};

    vec::dense_vector<double> x_dispatch(4, 0.0);
    lu_solve_dispatch(A, x_dispatch, b);

    vec::dense_vector<double> x_explicit(4, 0.0);
    sparse::factorization::sparse_lu_solve(A, x_explicit, b,
        sparse::ordering::colamd{});

    for (std::size_t i = 0; i < 4; ++i)
        REQUIRE_THAT(x_dispatch(i), Catch::Matchers::WithinAbs(x_explicit(i), 1e-10));
}

// ---- Larger dispatch tests ----

TEST_CASE("solve() on 20x20 sparse system", "[dispatch][scale]") {
    std::size_t n = 20;
    mat::compressed2D<double> A(n, n);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        for (std::size_t i = 0; i < n; ++i) {
            ins[i][i] << 4.0;
            if (i + 1 < n) {
                ins[i][i + 1] << -1.0;
                ins[i + 1][i] << -0.5;
            }
        }
    }
    vec::dense_vector<double> b(n, 1.0);
    vec::dense_vector<double> x(n, 0.0);

    solve(A, x, b);
    REQUIRE(sparse_residual(A, x, b) < 1e-12);
}
