#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/lu.hpp>
#include <mtl/operation/inv.hpp>
#include <mtl/operation/operators.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/generators/frank.hpp>
#include <mtl/generators/moler.hpp>
#include <mtl/generators/hilbert.hpp>
#include <mtl/generators/pascal.hpp>

using namespace mtl;

TEST_CASE("LU factorization and solve", "[operation][lu]") {
    mat::dense2D<double> A(3, 3);
    A(0,0) = 2; A(0,1) = 1; A(0,2) = 1;
    A(1,0) = 4; A(1,1) = 3; A(1,2) = 3;
    A(2,0) = 8; A(2,1) = 7; A(2,2) = 9;

    vec::dense_vector<double> b = {4.0, 10.0, 24.0};

    // Copy A for factoring (it gets modified)
    mat::dense2D<double> LU(3, 3);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            LU(i, j) = A(i, j);

    std::vector<std::size_t> pivot;
    int info = lu_factor(LU, pivot);
    REQUIRE(info == 0);

    vec::dense_vector<double> x(3);
    lu_solve(LU, pivot, x, b);

    // Verify A*x = b
    auto r = A * x;
    for (std::size_t i = 0; i < 3; ++i)
        REQUIRE_THAT(r(i), Catch::Matchers::WithinAbs(b(i), 1e-10));
}

TEST_CASE("LU convenience function lu_apply", "[operation][lu]") {
    mat::dense2D<double> A(3, 3);
    A(0,0) = 1; A(0,1) = 2; A(0,2) = 3;
    A(1,0) = 4; A(1,1) = 5; A(1,2) = 6;
    A(2,0) = 7; A(2,1) = 8; A(2,2) = 10;

    // Save original
    mat::dense2D<double> Aorig(3, 3);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            Aorig(i, j) = A(i, j);

    vec::dense_vector<double> b = {1.0, 2.0, 3.0};
    vec::dense_vector<double> x(3);

    int info = lu_apply(A, x, b);
    REQUIRE(info == 0);

    auto r = Aorig * x;
    for (std::size_t i = 0; i < 3; ++i)
        REQUIRE_THAT(r(i), Catch::Matchers::WithinAbs(b(i), 1e-10));
}

TEST_CASE("Matrix inverse via LU", "[operation][inv]") {
    mat::dense2D<double> A(3, 3);
    A(0,0) = 4; A(0,1) = 7; A(0,2) = 2;
    A(1,0) = 3; A(1,1) = 6; A(1,2) = 1;
    A(2,0) = 2; A(2,1) = 5; A(2,2) = 3;

    auto Ainv = inv(A);

    // A * Ainv should be approximately I
    auto I_approx = A * Ainv;
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            REQUIRE_THAT(I_approx(i, j), Catch::Matchers::WithinAbs(expected, 1e-10));
        }
}

// -- Generator-based LU tests -----------------------------------------

TEST_CASE("LU solve on 8x8 Frank matrix", "[operation][lu][generator]") {
    constexpr std::size_t n = 8;
    auto A = generators::frank<double>(n);

    mat::dense2D<double> Aorig(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            Aorig(i, j) = A(i, j);

    // Known solution
    vec::dense_vector<double> x_true(n);
    for (std::size_t i = 0; i < n; ++i)
        x_true(i) = static_cast<double>(i + 1);

    auto b = Aorig * x_true;

    vec::dense_vector<double> x(n);
    int info = lu_apply(A, x, b);
    REQUIRE(info == 0);

    // Verify backward error
    auto residual = Aorig * x - b;
    double rel_residual = two_norm(residual) / two_norm(b);
    REQUIRE(rel_residual < 1e-10);
}

TEST_CASE("LU solve on Moler matrix", "[operation][lu][generator]") {
    constexpr std::size_t n = 6;
    auto A = generators::moler<double>(n);

    mat::dense2D<double> Aorig(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            Aorig(i, j) = A(i, j);

    vec::dense_vector<double> x_true(n);
    for (std::size_t i = 0; i < n; ++i)
        x_true(i) = static_cast<double>(i + 1);

    auto b = Aorig * x_true;

    vec::dense_vector<double> x(n);
    int info = lu_apply(A, x, b);
    REQUIRE(info == 0);

    auto residual = Aorig * x - b;
    double rel_residual = two_norm(residual) / two_norm(b);
    REQUIRE(rel_residual < 1e-8);
}

TEST_CASE("LU on ill-conditioned Hilbert 6x6", "[operation][lu][generator]") {
    constexpr std::size_t n = 6;
    generators::hilbert<double> H_gen(n);
    mat::dense2D<double> A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A(i, j) = H_gen(i, j);

    mat::dense2D<double> Aorig(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            Aorig(i, j) = A(i, j);

    vec::dense_vector<double> x_true(n);
    for (std::size_t i = 0; i < n; ++i)
        x_true(i) = 1.0;

    auto b = Aorig * x_true;

    vec::dense_vector<double> x(n);
    int info = lu_apply(A, x, b);
    REQUIRE(info == 0);

    // Hilbert is ill-conditioned: check backward error, not forward error
    auto residual = Aorig * x - b;
    double rel_residual = two_norm(residual) / two_norm(b);
    REQUIRE(rel_residual < 1e-4);
}

TEST_CASE("LU inverse of Pascal matrix", "[operation][lu][generator]") {
    constexpr std::size_t n = 5;
    auto A = generators::pascal<double>(n);

    auto Ainv = inv(A);

    // Pascal has det=1, well-conditioned: A * A^{-1} should be I
    auto I_approx = A * Ainv;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            REQUIRE_THAT(I_approx(i, j), Catch::Matchers::WithinAbs(expected, 1e-10));
        }
}

// ---------------------------------------------------------------------------
// #394: lu_adjoint_solve -- solve A^H x = b from the LU of A.
//
// lu_factor gives P*A = L*U, so A = P^-1*L*U and A^H = U^H*L^H*P (P is a real
// permutation, so P^-H = P). The solve is therefore the mirror of lu_solve: the
// triangular factors conjugated and in the opposite order, and the permutation
// moved to the END and inverted -- the same interchanges in reverse.
//
// Needed by bicg and qmr, the only solvers that ask for M^-H, via the
// block_diagonal preconditioner.
// ---------------------------------------------------------------------------

TEST_CASE("lu_adjoint_solve solves A^H x = b (#394)", "[operation][lu][regression]") {
    // Sizes chosen to span the pivoting behaviour: n = 1 and 2 exercise the
    // degenerate paths, the larger ones actually permute rows.
    for (std::size_t n : {std::size_t{1}, std::size_t{2}, std::size_t{5},
                          std::size_t{17}, std::size_t{40}}) {
        INFO("n = " << n);
        mat::dense2D<double> A(n, n);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                A(i, j) = std::sin(static_cast<double>(3 * i + 7 * j + 1))
                        + (i == j ? 4.0 * static_cast<double>(n) : 0.0);

        mat::dense2D<double> LU(A);
        std::vector<std::size_t> pivot;
        REQUIRE(lu_factor(LU, pivot) == 0);

        vec::dense_vector<double> b(n), x(n);
        for (std::size_t i = 0; i < n; ++i)
            b[i] = 1.0 + 0.5 * std::cos(static_cast<double>(i));

        lu_adjoint_solve(LU, pivot, x, b);

        // Residual of A^H x - b, computed directly from the ORIGINAL A. For a
        // real A, A^H is A^T, so column i of A dotted with x.
        double bnorm = 0.0, resid = 0.0;
        for (std::size_t i = 0; i < n; ++i) bnorm = std::max(bnorm, std::abs(b(i)));
        for (std::size_t i = 0; i < n; ++i) {
            double s = 0.0;
            for (std::size_t j = 0; j < n; ++j) s += A(j, i) * x(j);
            resid = std::max(resid, std::abs(s - b(i)));
        }
        REQUIRE(resid / bnorm < 1e-10);
    }
}

TEST_CASE("lu_adjoint_solve is the adjoint of lu_solve (#394)",
          "[operation][lu][regression]") {
    // The defining property, which is sharper than a residual check and is what
    // the preconditioners are ultimately relied on for:
    //     <A^-1 b, c> == <b, A^-H c>
    const std::size_t n = 24;
    mat::dense2D<double> A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A(i, j) = std::sin(static_cast<double>(5 * i + 2 * j))
                    + (i == j ? 8.0 : 0.0);

    mat::dense2D<double> LU(A);
    std::vector<std::size_t> pivot;
    REQUIRE(lu_factor(LU, pivot) == 0);

    vec::dense_vector<double> b(n), c(n), Ab(n), AHc(n);
    for (std::size_t i = 0; i < n; ++i) {
        b[i] = std::cos(static_cast<double>(i) * 1.7);
        c[i] = std::sin(static_cast<double>(i) * 0.9);
    }
    lu_solve(LU, pivot, Ab, b);
    lu_adjoint_solve(LU, pivot, AHc, c);

    double lhs = 0.0, rhs = 0.0;
    for (std::size_t i = 0; i < n; ++i) { lhs += Ab(i) * c(i); rhs += b(i) * AHc(i); }
    REQUIRE(std::abs(lhs - rhs) / std::max(1.0, std::abs(lhs)) < 1e-12);
}
