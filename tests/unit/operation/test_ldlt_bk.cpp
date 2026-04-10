#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <stdexcept>
#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/ldlt_bk.hpp>
#include <mtl/operation/operators.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/operation/trans.hpp>
#include <mtl/generators/randspd.hpp>

using namespace mtl;

// Helper: compute ||Ax - b|| / (||A||*||x||)
static double backward_error(const mat::dense2D<double>& A,
                             const vec::dense_vector<double>& x,
                             const vec::dense_vector<double>& b) {
    auto Ax = A * x;
    double res = 0.0;
    for (std::size_t i = 0; i < b.size(); ++i) {
        double d = Ax(i) - b(i);
        res += d * d;
    }
    res = std::sqrt(res);
    double Anorm = frobenius_norm(A);
    double xnorm = two_norm(x);
    return (Anorm * xnorm > 0.0) ? res / (Anorm * xnorm) : res;
}

TEST_CASE("Bunch-Kaufman on SPD matrix", "[operation][ldlt_bk]") {
    // SPD matrix: A = {{4,2,1},{2,5,3},{1,3,6}}
    mat::dense2D<double> A(3, 3);
    A(0,0) = 4; A(0,1) = 2; A(0,2) = 1;
    A(1,0) = 2; A(1,1) = 5; A(1,2) = 3;
    A(2,0) = 1; A(2,1) = 3; A(2,2) = 6;

    mat::dense2D<double> Aorig(3, 3);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            Aorig(i, j) = A(i, j);

    bk_pivot_info pivots;
    int info = ldlt_bk_factor(A, pivots);
    REQUIRE(info == 0);
    REQUIRE(pivots.ipiv.size() == 3);

    // Solve
    vec::dense_vector<double> b = {1.0, 2.0, 3.0};
    vec::dense_vector<double> x(3);
    ldlt_bk_solve(A, pivots, x, b);

    double be = backward_error(Aorig, x, b);
    REQUIRE(be < 1e-12);
}

TEST_CASE("Bunch-Kaufman on symmetric indefinite matrix", "[operation][ldlt_bk]") {
    // Indefinite: eigenvalues +/- sqrt(5)
    // A = {{1, 2}, {2, -1}}
    mat::dense2D<double> A(2, 2);
    A(0,0) = 1;  A(0,1) = 2;
    A(1,0) = 2;  A(1,1) = -1;

    mat::dense2D<double> Aorig(2, 2);
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 2; ++j)
            Aorig(i, j) = A(i, j);

    bk_pivot_info pivots;
    int info = ldlt_bk_factor(A, pivots);
    REQUIRE(info == 0);

    vec::dense_vector<double> b = {5.0, 3.0};
    vec::dense_vector<double> x(2);
    ldlt_bk_solve(A, pivots, x, b);

    double be = backward_error(Aorig, x, b);
    REQUIRE(be < 1e-12);
}

TEST_CASE("Bunch-Kaufman on matrix requiring 2x2 pivot", "[operation][ldlt_bk]") {
    // Matrix where diagonal is zero but off-diagonals are large
    // Forces 2x2 pivot selection
    // A = {{0, 1, 0}, {1, 0, 1}, {0, 1, 0}} — singular, but
    // Use: A = {{0, 3, 1}, {3, 0, 2}, {1, 2, 5}}
    // A(0,0) = 0 forces a pivot swap or 2x2
    mat::dense2D<double> A(3, 3);
    A(0,0) = 0; A(0,1) = 3; A(0,2) = 1;
    A(1,0) = 3; A(1,1) = 0; A(1,2) = 2;
    A(2,0) = 1; A(2,1) = 2; A(2,2) = 5;

    mat::dense2D<double> Aorig(3, 3);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            Aorig(i, j) = A(i, j);

    bk_pivot_info pivots;
    int info = ldlt_bk_factor(A, pivots);
    REQUIRE(info == 0);

    // Check that at least one 2x2 pivot was used (negative ipiv)
    bool has_2x2 = false;
    for (int p : pivots.ipiv)
        if (p < 0) has_2x2 = true;
    REQUIRE(has_2x2);

    vec::dense_vector<double> b = {4.0, 5.0, 12.0};
    vec::dense_vector<double> x(3);
    ldlt_bk_solve(A, pivots, x, b);

    double be = backward_error(Aorig, x, b);
    REQUIRE(be < 1e-12);
}

TEST_CASE("Bunch-Kaufman on 1x1 matrix", "[operation][ldlt_bk]") {
    mat::dense2D<double> A(1, 1);
    A(0, 0) = 7.0;

    bk_pivot_info pivots;
    int info = ldlt_bk_factor(A, pivots);
    REQUIRE(info == 0);

    vec::dense_vector<double> b = {21.0};
    vec::dense_vector<double> x(1);
    ldlt_bk_solve(A, pivots, x, b);
    REQUIRE_THAT(x(0), Catch::Matchers::WithinAbs(3.0, 1e-12));
}

TEST_CASE("Bunch-Kaufman on 4x4 indefinite matrix", "[operation][ldlt_bk]") {
    // Symmetric indefinite 4x4 — the type that arises in UKF covariance updates
    mat::dense2D<double> A(4, 4);
    A(0,0) =  2; A(0,1) =  1; A(0,2) = -1; A(0,3) =  0;
    A(1,0) =  1; A(1,1) = -3; A(1,2) =  2; A(1,3) =  1;
    A(2,0) = -1; A(2,1) =  2; A(2,2) =  4; A(2,3) = -2;
    A(3,0) =  0; A(3,1) =  1; A(3,2) = -2; A(3,3) =  1;

    mat::dense2D<double> Aorig(4, 4);
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            Aorig(i, j) = A(i, j);

    bk_pivot_info pivots;
    int info = ldlt_bk_factor(A, pivots);
    REQUIRE(info == 0);

    vec::dense_vector<double> b = {1.0, -2.0, 3.0, -1.0};
    vec::dense_vector<double> x(4);
    ldlt_bk_solve(A, pivots, x, b);

    double be = backward_error(Aorig, x, b);
    REQUIRE(be < 1e-12);
}

TEST_CASE("Bunch-Kaufman on 5x5 SPD matrix", "[operation][ldlt_bk]") {
    // Deterministic well-conditioned SPD: Lehmer matrix (small)
    constexpr std::size_t n = 5;
    mat::dense2D<double> A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A(i, j) = double(std::min(i, j) + 1) / double(std::max(i, j) + 1);

    mat::dense2D<double> Aorig(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            Aorig(i, j) = A(i, j);

    bk_pivot_info pivots;
    int info = ldlt_bk_factor(A, pivots);
    REQUIRE(info == 0);

    vec::dense_vector<double> b(n, 1.0);
    vec::dense_vector<double> x(n);
    ldlt_bk_solve(A, pivots, x, b);

    double be = backward_error(Aorig, x, b);
    REQUIRE(be < 1e-12);
}

TEST_CASE("Bunch-Kaufman on 8x8 diagonally dominant SPD", "[operation][ldlt_bk]") {
    // Deterministic: tridiagonal SPD with strong diagonal dominance
    constexpr std::size_t n = 8;
    mat::dense2D<double> A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A(i, j) = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        A(i, i) = 10.0;
        if (i > 0) { A(i, i-1) = -1.0; A(i-1, i) = -1.0; }
    }

    mat::dense2D<double> Aorig(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            Aorig(i, j) = A(i, j);

    bk_pivot_info pivots;
    int info = ldlt_bk_factor(A, pivots);
    REQUIRE(info == 0);

    vec::dense_vector<double> b(n, 1.0);
    vec::dense_vector<double> x(n);
    ldlt_bk_solve(A, pivots, x, b);

    double be = backward_error(Aorig, x, b);
    REQUIRE(be < 1e-12);
}

TEST_CASE("Bunch-Kaufman on large indefinite system", "[operation][ldlt_bk]") {
    // Construct a symmetric indefinite matrix: diag with alternating signs
    // plus off-diagonal coupling
    constexpr std::size_t n = 20;
    mat::dense2D<double> A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A(i, j) = 0.0;

    for (std::size_t i = 0; i < n; ++i) {
        A(i, i) = (i % 2 == 0) ? 10.0 : -5.0;
        if (i + 1 < n) {
            A(i, i + 1) = 2.0;
            A(i + 1, i) = 2.0;
        }
    }

    mat::dense2D<double> Aorig(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            Aorig(i, j) = A(i, j);

    bk_pivot_info pivots;
    int info = ldlt_bk_factor(A, pivots);
    REQUIRE(info == 0);

    vec::dense_vector<double> b(n, 1.0);
    vec::dense_vector<double> x(n);
    ldlt_bk_solve(A, pivots, x, b);

    double be = backward_error(Aorig, x, b);
    REQUIRE(be < 1e-10);
}

TEST_CASE("Bunch-Kaufman detects singular matrix", "[operation][ldlt_bk]") {
    // Singular: A = {{0, 0}, {0, 0}}
    mat::dense2D<double> A(2, 2);
    A(0,0) = 0; A(0,1) = 0;
    A(1,0) = 0; A(1,1) = 0;

    bk_pivot_info pivots;
    int info = ldlt_bk_factor(A, pivots);
    REQUIRE(info != 0);
}

TEST_CASE("Bunch-Kaufman on empty matrix", "[operation][ldlt_bk]") {
    mat::dense2D<double> A(0, 0);
    bk_pivot_info pivots;
    int info = ldlt_bk_factor(A, pivots);
    REQUIRE(info == 0);
}
