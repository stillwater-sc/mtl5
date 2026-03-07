// MTL5 Phase 14 -- Tests for LAPACK dispatch paths
// These tests verify that the dispatch logic in LU, QR, Cholesky, SVD, and
// eigenvalue operations produces correct results. When MTL5_HAS_LAPACK is defined,
// the LAPACK-accelerated paths are exercised; otherwise the C++ fallback paths are tested.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/lu.hpp>
#include <mtl/operation/qr.hpp>
#include <mtl/operation/cholesky.hpp>
#include <mtl/operation/svd.hpp>
#include <mtl/operation/eigenvalue_symmetric.hpp>
#include <mtl/operation/mult.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/operation/operators.hpp>
#include <mtl/math/identity.hpp>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace mtl;
using Catch::Matchers::WithinAbs;

// -- LU dispatch -------------------------------------------------------------

TEST_CASE("lu_factor and lu_solve produce correct solution", "[interface][lapack]") {
    // A = [2 1; 5 3], b = [4; 7] => x = [5; -6]
    mat::dense2D<double> A(2, 2);
    A(0,0) = 2.0; A(0,1) = 1.0;
    A(1,0) = 5.0; A(1,1) = 3.0;

    vec::dense_vector<double> b = {4.0, 7.0};
    vec::dense_vector<double> x(2, 0.0);

    std::vector<std::size_t> pivot;
    int info = lu_factor(A, pivot);
    REQUIRE(info == 0);

    lu_solve(A, pivot, x, b);
    REQUIRE_THAT(x(0), WithinAbs(5.0, 1e-10));
    REQUIRE_THAT(x(1), WithinAbs(-6.0, 1e-10));
}

TEST_CASE("lu_apply convenience function", "[interface][lapack]") {
    mat::dense2D<double> A(3, 3);
    A(0,0) = 1; A(0,1) = 2; A(0,2) = 3;
    A(1,0) = 4; A(1,1) = 5; A(1,2) = 6;
    A(2,0) = 7; A(2,1) = 8; A(2,2) = 10;

    vec::dense_vector<double> b = {1.0, 2.0, 3.0};
    vec::dense_vector<double> x(3, 0.0);

    int info = lu_apply(A, x, b);
    REQUIRE(info == 0);

    // Verify: original A * x should equal b
    // Reconstruct original A
    mat::dense2D<double> A_orig(3, 3);
    A_orig(0,0) = 1; A_orig(0,1) = 2; A_orig(0,2) = 3;
    A_orig(1,0) = 4; A_orig(1,1) = 5; A_orig(1,2) = 6;
    A_orig(2,0) = 7; A_orig(2,1) = 8; A_orig(2,2) = 10;

    vec::dense_vector<double> Ax(3, 0.0);
    mult(A_orig, x, Ax);
    for (int i = 0; i < 3; ++i)
        REQUIRE_THAT(Ax(i), WithinAbs(b(i), 1e-10));
}

// -- QR dispatch -------------------------------------------------------------

TEST_CASE("qr_factor produces valid Q and R", "[interface][lapack]") {
    mat::dense2D<double> A(3, 3);
    A(0,0) = 12; A(0,1) = -51; A(0,2) = 4;
    A(1,0) = 6;  A(1,1) = 167; A(1,2) = -68;
    A(2,0) = -4; A(2,1) = 24;  A(2,2) = -41;

    // Save original
    mat::dense2D<double> A_orig(3, 3);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            A_orig(i, j) = A(i, j);

    vec::dense_vector<double> tau;
    int info = qr_factor(A, tau);
    REQUIRE(info == 0);

    auto Q = qr_extract_Q(A, tau);
    auto R = qr_extract_R(A);

    // Verify Q*R = A_orig
    mat::dense2D<double> QR(3, 3);
    mult(Q, R, QR);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            REQUIRE_THAT(QR(i, j), WithinAbs(A_orig(i, j), 1e-8));
}

// -- Cholesky dispatch -------------------------------------------------------

TEST_CASE("cholesky_factor and cholesky_solve on SPD matrix", "[interface][lapack]") {
    // A = [4 2; 2 3] (SPD), b = [1; 2] => x = [-1/8; 3/4]
    mat::dense2D<double> A(2, 2);
    A(0,0) = 4.0; A(0,1) = 2.0;
    A(1,0) = 2.0; A(1,1) = 3.0;

    int info = cholesky_factor(A);
    REQUIRE(info == 0);

    // Verify L(0,0) = sqrt(4) = 2
    REQUIRE_THAT(A(0,0), WithinAbs(2.0, 1e-10));

    vec::dense_vector<double> b = {1.0, 2.0};
    vec::dense_vector<double> x(2, 0.0);

    cholesky_solve(A, x, b);
    REQUIRE_THAT(x(0), WithinAbs(-0.125, 1e-10));
    REQUIRE_THAT(x(1), WithinAbs(0.75, 1e-10));
}

TEST_CASE("cholesky_factor detects non-SPD matrix", "[interface][lapack]") {
    mat::dense2D<double> A(2, 2);
    A(0,0) = 1.0; A(0,1) = 2.0;
    A(1,0) = 2.0; A(1,1) = 1.0;  // Not positive definite

    int info = cholesky_factor(A);
    REQUIRE(info != 0);
}

// -- SVD dispatch ------------------------------------------------------------

TEST_CASE("svd decomposes matrix correctly", "[interface][lapack]") {
    mat::dense2D<double> A(2, 2);
    A(0,0) = 3.0; A(0,1) = 0.0;
    A(1,0) = 0.0; A(1,1) = 4.0;

    mat::dense2D<double> U, S, V;
    svd(A, U, S, V, 1e-10);

    // Singular values should be 4 and 3 (sorted descending for LAPACK, may vary for C++)
    std::vector<double> sv = {S(0,0), S(1,1)};
    std::sort(sv.begin(), sv.end());
    REQUIRE_THAT(sv[0], WithinAbs(3.0, 1e-6));
    REQUIRE_THAT(sv[1], WithinAbs(4.0, 1e-6));
}

TEST_CASE("svd: U*S*V^T reconstructs A", "[interface][lapack]") {
    mat::dense2D<double> A(2, 2);
    A(0,0) = 1.0; A(0,1) = 2.0;
    A(1,0) = 3.0; A(1,1) = 4.0;

    mat::dense2D<double> U, S, V;
    svd(A, U, S, V, 1e-10);

    // Reconstruct: A_recon = U * S * V^T
    mat::dense2D<double> US(2, 2), A_recon(2, 2);
    mult(U, S, US);

    // V^T
    mat::dense2D<double> VT(2, 2);
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            VT(i, j) = V(j, i);

    mult(US, VT, A_recon);

    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            REQUIRE_THAT(A_recon(i, j), WithinAbs(A(i, j), 1e-6));
}

// -- Eigenvalue dispatch -----------------------------------------------------

TEST_CASE("eigenvalue_symmetric produces correct eigenvalues", "[interface][lapack]") {
    // Diagonal matrix: eigenvalues = {1, 2, 3}
    mat::dense2D<double> A(3, 3);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            A(i, j) = (i == j) ? static_cast<double>(i + 1) : 0.0;

    auto eigs = eigenvalue_symmetric(A);
    REQUIRE(eigs.size() == 3);
    REQUIRE_THAT(eigs(0), WithinAbs(1.0, 1e-8));
    REQUIRE_THAT(eigs(1), WithinAbs(2.0, 1e-8));
    REQUIRE_THAT(eigs(2), WithinAbs(3.0, 1e-8));
}

TEST_CASE("eigenvalue_symmetric on 2x2 SPD", "[interface][lapack]") {
    // A = [2 1; 1 2], eigenvalues = 1 and 3
    mat::dense2D<double> A(2, 2);
    A(0,0) = 2.0; A(0,1) = 1.0;
    A(1,0) = 1.0; A(1,1) = 2.0;

    auto eigs = eigenvalue_symmetric(A);
    REQUIRE(eigs.size() == 2);
    REQUIRE_THAT(eigs(0), WithinAbs(1.0, 1e-8));
    REQUIRE_THAT(eigs(1), WithinAbs(3.0, 1e-8));
}
