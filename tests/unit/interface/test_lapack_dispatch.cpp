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
#include <mtl/operation/eigenvalue.hpp>
#include <mtl/operation/mult.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/operation/operators.hpp>
#include <mtl/math/identity.hpp>
#include <vector>
#include <cmath>
#include <complex>
#include <algorithm>

using namespace mtl;
using Catch::Matchers::WithinAbs;
using cplxd = std::complex<double>;

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

// -- Complex Hermitian eigenvalue dispatch (cheev/zheev) ---------------------
// eigenvalue_symmetric routes a complex Hermitian matrix to LAPACK heev when
// MTL5_HAS_LAPACK is defined AND the matrix is column-major (LAPACK is
// column-major; the dispatch guards on !is_row_major_v<M>, exactly as the real
// syev / geev / BLAS-L3 dispatch tests do). dense2D defaults to row-major, so
// these tests use col-major explicitly -- otherwise the call silently takes the
// generic path and the cross-check proves nothing. eigenvalue_symmetric_generic
// is ALWAYS the in-house path, so comparing the two is the genuine native-vs-
// LAPACK cross-check the self-consistent #414 tests could not do. (In a build
// without LAPACK, eigenvalue_symmetric also runs generic, so the comparison
// degenerates to a still-valid native self-check.)
using col_params = mat::parameters<tag::col_major>;
using cmat       = mat::dense2D<cplxd, col_params>;

// Hermitian A = 2*I + v*v^H has eigenvalues {2 (mult n-1), 2+||v||^2}. With
// v = {1, i, 1+i, 2}, ||v||^2 = 8, so the spectrum is {2, 2, 2, 10} -- a known,
// degenerate spectrum that also stresses the repeated-eigenvalue subspace.
static cmat hermitian_rank1_update() {
    const std::size_t n = 4;
    cplxd v[4] = { cplxd(1,0), cplxd(0,1), cplxd(1,1), cplxd(2,0) };
    cmat A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A(i, j) = (i == j ? cplxd(2,0) : cplxd(0,0)) + v[i] * std::conj(v[j]);
    return A;
}

TEST_CASE("heev dispatch: complex Hermitian 2x2 known spectrum",
          "[interface][lapack][complex]") {
    // [[2, i], [-i, 2]] has eigenvalues 1 and 3.
    cmat A(2, 2);
    A(0,0) = cplxd(2,0);  A(0,1) = cplxd(0,1);
    A(1,0) = cplxd(0,-1); A(1,1) = cplxd(2,0);

    auto eigs = eigenvalue_symmetric(A);   // zheev under LAPACK, generic otherwise
    REQUIRE(eigs.size() == 2);
    REQUIRE_THAT(eigs(0), WithinAbs(1.0, 1e-9));
    REQUIRE_THAT(eigs(1), WithinAbs(3.0, 1e-9));
}

TEST_CASE("heev dispatch: known degenerate spectrum {2,2,2,10}",
          "[interface][lapack][complex]") {
    auto A = hermitian_rank1_update();
    auto eigs = eigenvalue_symmetric(A);
    REQUIRE(eigs.size() == 4);
    REQUIRE_THAT(eigs(0), WithinAbs(2.0,  1e-9));
    REQUIRE_THAT(eigs(1), WithinAbs(2.0,  1e-9));
    REQUIRE_THAT(eigs(2), WithinAbs(2.0,  1e-9));
    REQUIRE_THAT(eigs(3), WithinAbs(10.0, 1e-9));
}

TEST_CASE("heev vs native: eigenvalues agree on a dense 5x5 Hermitian",
          "[interface][lapack][complex]") {
    constexpr std::size_t n = 5;
    cmat A(n, n);
    for (std::size_t i = 0; i < n; ++i) {
        A(i, i) = cplxd(double(2 * i + 1), 0.0);
        for (std::size_t j = i + 1; j < n; ++j) {
            cplxd z(0.5 * double(i + 1) - 0.25 * double(j),
                    0.3 * double(j + 1) - 0.2 * double(i));
            A(i, j) = z;
            A(j, i) = std::conj(z);
        }
    }

    auto native = eigenvalue_symmetric_generic(A);  // in-house reduction + QR
    auto disp   = eigenvalue_symmetric(A);           // LAPACK zheev when available

    REQUIRE(native.size() == n);
    REQUIRE(disp.size()   == n);
    // Both ascending; compare elementwise. Under LAPACK this is native vs zheev.
    for (std::size_t i = 0; i < n; ++i)
        REQUIRE_THAT(disp(i), WithinAbs(native(i), 1e-9));
}

// -- #417: the DEFAULT (row-major) dense2D now dispatches to LAPACK too -------
// The eigen/SVD/geev dispatches build a column-major copy through the (i,j)
// accessor, so the old !is_row_major_v guard was redundant and only kept the
// default matrix type on the generic path. These tests use the DEFAULT
// (row-major) dense2D -- no col_params -- and check the LAPACK result matches the
// in-house path. (Verified out-of-band that a row-major-only eigen translation
// unit references dsyev_/zheev_ after the guard drop; before it, none did, so the
// dispatch really was dead for row-major.)

TEST_CASE("row-major dispatch: real symmetric eigenvalues match generic",
          "[interface][lapack][rowmajor]") {
    constexpr std::size_t n = 4;
    mat::dense2D<double> A(n, n);   // default row-major
    for (std::size_t i = 0; i < n; ++i) {
        A(i, i) = double(i + 1);
        for (std::size_t j = i + 1; j < n; ++j) {
            double v = 0.3 * double(i + 1) - 0.1 * double(j);
            A(i, j) = v; A(j, i) = v;
        }
    }
    auto disp = eigenvalue_symmetric(A);            // dsyev under LAPACK (row-major now)
    auto gen  = eigenvalue_symmetric_generic(A);    // in-house
    REQUIRE(disp.size() == n);
    for (std::size_t i = 0; i < n; ++i)
        REQUIRE_THAT(disp(i), WithinAbs(gen(i), 1e-9));
}

TEST_CASE("row-major dispatch: complex Hermitian eigenvalues match generic",
          "[interface][lapack][rowmajor][complex]") {
    constexpr std::size_t n = 4;
    mat::dense2D<cplxd> A(n, n);   // default row-major -- the case #416 could not cover
    for (std::size_t i = 0; i < n; ++i) {
        A(i, i) = cplxd(double(i + 2), 0.0);
        for (std::size_t j = i + 1; j < n; ++j) {
            cplxd z(0.4 * double(i + 1) - 0.1 * double(j),
                    0.25 * double(j) - 0.15 * double(i + 1));
            A(i, j) = z; A(j, i) = std::conj(z);
        }
    }
    auto disp = eigenvalue_symmetric(A);            // zheev under LAPACK (row-major now)
    auto gen  = eigenvalue_symmetric_generic(A);    // in-house reduction + QR
    REQUIRE(disp.size() == n);
    for (std::size_t i = 0; i < n; ++i)
        REQUIRE_THAT(disp(i), WithinAbs(gen(i), 1e-9));
}

TEST_CASE("row-major dispatch: general eigenvalue (geev) known real spectrum",
          "[interface][lapack][rowmajor]") {
    // Upper-triangular, so the eigenvalues are the diagonal {2, 3, 5}. Not
    // symmetric -> exercises the general geev dispatch, not syev.
    mat::dense2D<double> A(3, 3);   // default row-major
    A(0,0)=2; A(0,1)=1; A(0,2)=0;
    A(1,0)=0; A(1,1)=3; A(1,2)=1;
    A(2,0)=0; A(2,1)=0; A(2,2)=5;

    auto eigs = eigenvalue(A);      // geev under LAPACK (row-major now), else in-house QR
    REQUIRE(eigs.size() == 3);
    std::vector<double> re{ eigs(0).real(), eigs(1).real(), eigs(2).real() };
    std::sort(re.begin(), re.end());
    REQUIRE_THAT(re[0], WithinAbs(2.0, 1e-9));
    REQUIRE_THAT(re[1], WithinAbs(3.0, 1e-9));
    REQUIRE_THAT(re[2], WithinAbs(5.0, 1e-9));
    for (std::size_t i = 0; i < 3; ++i)
        REQUIRE_THAT(eigs(i).imag(), WithinAbs(0.0, 1e-9));
}
