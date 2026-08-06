// Complex Hermitian eigenproblem (#362).
//
// eigen_symmetric / eigenvalue_symmetric now reduce a complex Hermitian matrix
// by the UNITARY similarity A -> H*A*H^H and fold the subdiagonal phase into the
// eigenvectors. These tests pin the invariants the issue called out:
//   - eigenvalues are REAL and match a reference,
//   - A*v == lambda*v per eigenpair (catches a missing phase accumulation,
//     which leaves the eigenvalues looking right),
//   - Q^H*Q == I (necessary, NOT sufficient), and
//   - the reconstruction A == V*diag(lambda)*V^H (the check that actually
//     catches a non-similar reduction),
//   - the Hessenberg/tridiagonal reduction itself is a similarity (trace
//     preserved) and produces the right structure.
//
// Everything is computed with explicit loops over std::complex<double> so the
// tests do not depend on complex-capable matrix operators.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <complex>
#include <cstddef>
#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/eigenvalue_symmetric.hpp>
#include <mtl/operation/hessenberg.hpp>

using namespace mtl;
using cplx = std::complex<double>;
using Mat  = mat::dense2D<cplx>;

// ||A*v_k - lambda_k*v_k|| / (||A||_F * ||v_k||) maximised over the eigenpairs.
static double max_eigenpair_residual(const Mat& A,
                                     const vec::dense_vector<double>& eigs,
                                     const Mat& V) {
    const std::size_t n = A.num_rows();
    double Anorm = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            Anorm += std::norm(A(i, j));   // |A(i,j)|^2
    Anorm = std::sqrt(Anorm);

    double max_res = 0.0;
    for (std::size_t k = 0; k < n; ++k) {
        double res_sq = 0.0, v_sq = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            cplx Av_i = 0.0;
            for (std::size_t j = 0; j < n; ++j)
                Av_i += A(i, j) * V(j, k);
            cplx ri = Av_i - eigs(k) * V(i, k);
            res_sq += std::norm(ri);
            v_sq   += std::norm(V(i, k));
        }
        double res = std::sqrt(res_sq) / (Anorm * std::sqrt(v_sq));
        if (res > max_res) max_res = res;
    }
    return max_res;
}

// ||V^H*V - I||_F : the columns of V must be orthonormal in the Hermitian sense.
static double orthonormality_error(const Mat& V) {
    const std::size_t n = V.num_rows();
    double err = 0.0;
    for (std::size_t a = 0; a < n; ++a)
        for (std::size_t b = 0; b < n; ++b) {
            cplx dot = 0.0;
            for (std::size_t i = 0; i < n; ++i)
                dot += std::conj(V(i, a)) * V(i, b);
            cplx expected = (a == b) ? cplx(1.0) : cplx(0.0);
            err += std::norm(dot - expected);
        }
    return std::sqrt(err);
}

// ||V*diag(eigs)*V^H - A||_F : the reconstruction that catches a NON-similar
// reduction (a merely-unitary Q is not enough -- see #361).
static double reconstruction_error(const Mat& A,
                                   const vec::dense_vector<double>& eigs,
                                   const Mat& V) {
    const std::size_t n = A.num_rows();
    double err = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            cplx rec = 0.0;
            for (std::size_t k = 0; k < n; ++k)
                rec += V(i, k) * eigs(k) * std::conj(V(j, k));
            err += std::norm(rec - A(i, j));
        }
    return std::sqrt(err);
}

TEST_CASE("hermitian eigen: 2x2 with known spectrum", "[operation][eigen][complex]") {
    // A = [[2, i], [-i, 2]] has eigenvalues 1 and 3.
    Mat A(2, 2);
    A(0,0) = 2.0;            A(0,1) = cplx(0, 1);
    A(1,0) = cplx(0, -1);    A(1,1) = 2.0;

    auto [eigs, V] = eigen_symmetric(A);

    REQUIRE(eigs.size() == 2);
    REQUIRE_THAT(eigs(0), Catch::Matchers::WithinAbs(1.0, 1e-12));
    REQUIRE_THAT(eigs(1), Catch::Matchers::WithinAbs(3.0, 1e-12));

    REQUIRE(orthonormality_error(V)          < 1e-12);
    REQUIRE(max_eigenpair_residual(A, eigs, V) < 1e-12);
    REQUIRE(reconstruction_error(A, eigs, V)   < 1e-12);
}

TEST_CASE("hermitian eigen: 2x2 with an off-axis phase", "[operation][eigen][complex]") {
    // A = [[1, 1+i], [1-i, 2]] has det 0 and trace 3, so eigenvalues 0 and 3.
    // The subdiagonal 1-i carries a genuine (non +/-i) phase; n < 3 skips the
    // reduction, so this exercises the phase fold directly. Drop it and the
    // eigenvectors come back real and fail A*v == lambda*v.
    Mat A(2, 2);
    A(0,0) = 1.0;            A(0,1) = cplx(1, 1);
    A(1,0) = cplx(1, -1);    A(1,1) = 2.0;

    auto [eigs, V] = eigen_symmetric(A);

    REQUIRE_THAT(eigs(0), Catch::Matchers::WithinAbs(0.0, 1e-12));
    REQUIRE_THAT(eigs(1), Catch::Matchers::WithinAbs(3.0, 1e-12));

    REQUIRE(orthonormality_error(V)            < 1e-12);
    REQUIRE(max_eigenpair_residual(A, eigs, V) < 1e-12);
    REQUIRE(reconstruction_error(A, eigs, V)   < 1e-12);
}

TEST_CASE("hermitian eigen: 3x3 residual, orthonormality, reconstruction",
          "[operation][eigen][complex]") {
    Mat A(3, 3);
    A(0,0) = 2.0;          A(0,1) = cplx(1, -1);  A(0,2) = cplx(0, 0);
    A(1,0) = cplx(1, 1);   A(1,1) = 3.0;          A(1,2) = cplx(0, 1);
    A(2,0) = cplx(0, 0);   A(2,1) = cplx(0, -1);  A(2,2) = 4.0;

    auto [eigs, V] = eigen_symmetric(A);

    REQUIRE(eigs.size() == 3);
    // Sorted ascending.
    for (std::size_t i = 0; i + 1 < 3; ++i)
        REQUIRE(eigs(i) <= eigs(i + 1));
    // Trace is the sum of the eigenvalues (real for Hermitian).
    double trace = 2.0 + 3.0 + 4.0;
    REQUIRE_THAT(eigs(0) + eigs(1) + eigs(2),
                 Catch::Matchers::WithinAbs(trace, 1e-10));

    REQUIRE(orthonormality_error(V)            < 1e-12);
    REQUIRE(max_eigenpair_residual(A, eigs, V) < 1e-12);
    REQUIRE(reconstruction_error(A, eigs, V)   < 1e-12);
}

TEST_CASE("hermitian eigen: 5x5 dense Hermitian", "[operation][eigen][complex]") {
    constexpr std::size_t n = 5;
    // Deterministic Hermitian A: real diagonal, A(i,j) = conj(A(j,i)).
    Mat A(n, n);
    for (std::size_t i = 0; i < n; ++i) {
        A(i, i) = double(2 * i + 1);
        for (std::size_t j = i + 1; j < n; ++j) {
            cplx z(0.5 * double(i + 1) - 0.25 * double(j),
                   0.3 * double(j + 1) - 0.2 * double(i));
            A(i, j) = z;
            A(j, i) = std::conj(z);
        }
    }

    auto [eigs, V] = eigen_symmetric(A);

    // Trace invariant.
    double trace = 0.0;
    for (std::size_t i = 0; i < n; ++i) trace += A(i, i).real();
    double eig_sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) eig_sum += eigs(i);
    REQUIRE_THAT(eig_sum, Catch::Matchers::WithinAbs(trace, 1e-9));

    // Orthonormality is machine-precision (Givens rotations keep V unitary
    // regardless of convergence); the eigenpair residual and reconstruction are
    // bounded by the QR iteration's default tol (1e-10), so allow 1e-8.
    REQUIRE(orthonormality_error(V)            < 1e-11);
    REQUIRE(max_eigenpair_residual(A, eigs, V) < 1e-8);
    REQUIRE(reconstruction_error(A, eigs, V)   < 1e-8);
}

TEST_CASE("hermitian eigen: eigenvalue_symmetric matches eigen_symmetric values",
          "[operation][eigen][complex]") {
    Mat A(3, 3);
    A(0,0) = 5.0;          A(0,1) = cplx(2, -3);  A(0,2) = cplx(1, 1);
    A(1,0) = cplx(2, 3);   A(1,1) = 6.0;          A(1,2) = cplx(0, -2);
    A(2,0) = cplx(1, -1);  A(2,1) = cplx(0, 2);   A(2,2) = 7.0;

    auto values      = eigenvalue_symmetric(A);   // eigenvalues-only path
    auto [eigs, V]   = eigen_symmetric(A);         // eigenvalues + vectors

    REQUIRE(values.size() == 3);
    for (std::size_t i = 0; i < 3; ++i)
        REQUIRE_THAT(values(i), Catch::Matchers::WithinAbs(eigs(i), 1e-10));

    // And the eigenpairs must actually solve A*v = lambda*v.
    REQUIRE(max_eigenpair_residual(A, eigs, V) < 1e-11);
}

TEST_CASE("hermitian eigen: 0x0 complex input returns empty", "[operation][eigen][complex]") {
    // eigenvalue_symmetric routes complex to eigenvalue_symmetric_generic, whose
    // subdiagonal sentinel write e(n-1) would underflow the unsigned index at
    // n == 0. Both the dispatcher and the generic path must return empty.
    Mat A(0, 0);

    auto values = eigenvalue_symmetric(A);
    REQUIRE(values.size() == 0);

    auto values_generic = eigenvalue_symmetric_generic(A);
    REQUIRE(values_generic.size() == 0);

    auto [eigs, V] = eigen_symmetric(A);
    REQUIRE(eigs.size() == 0);
    REQUIRE(V.num_rows() == 0);
    REQUIRE(V.num_cols() == 0);
}

TEST_CASE("hessenberg: complex reduction preserves the trace (similarity)",
          "[operation][hessenberg][complex]") {
    // General (non-Hermitian) complex matrix. A -> H*A*H^H is a unitary
    // similarity, so the trace must be preserved and the result Hessenberg.
    constexpr std::size_t n = 4;
    Mat A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A(i, j) = cplx(1.0 + 0.5 * double(i) - 0.3 * double(j),
                           0.2 * double(i + 1) + 0.4 * double(j));

    cplx trace_before = 0.0;
    for (std::size_t i = 0; i < n; ++i) trace_before += A(i, i);

    auto H = hessenberg(A);

    cplx trace_after = 0.0;
    for (std::size_t i = 0; i < n; ++i) trace_after += H(i, i);
    REQUIRE(std::abs(trace_after - trace_before) < 1e-12);

    // Hessenberg structure: entries below the first subdiagonal are zero.
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j + 1 < i; ++j)
            REQUIRE(std::abs(H(i, j)) < 1e-12);
}

TEST_CASE("tridiagonalize: complex Hermitian reduces to Hermitian tridiagonal",
          "[operation][hessenberg][complex]") {
    constexpr std::size_t n = 5;
    Mat A(n, n);
    for (std::size_t i = 0; i < n; ++i) {
        A(i, i) = double(i + 2);
        for (std::size_t j = i + 1; j < n; ++j) {
            cplx z(0.4 * double(i + 1) - 0.1 * double(j),
                   0.25 * double(j) - 0.15 * double(i + 1));
            A(i, j) = z;
            A(j, i) = std::conj(z);
        }
    }

    Mat T(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            T(i, j) = A(i, j);

    vec::dense_vector<cplx> tau;
    tridiagonalize(T, tau);

    // Structure of the *observable* tridiagonal. hessenberg_factor PACKS the
    // Householder vectors into the strict-lower region below the subdiagonal, so
    // that region is intentionally non-zero -- only the upper triangle above the
    // superdiagonal is cleared. Check that (j > i+1), not the lower band.
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = i + 2; j < n; ++j)
            REQUIRE(std::abs(T(i, j)) < 1e-12);

    // Hermitian tridiagonal: real diagonal, and T(i,i+1) == conj(T(i+1,i)).
    // MTL5's householder() reflects to a real beta, so the subdiagonal comes out
    // real for the reduced (n>=3) columns -- but the assertion below only relies
    // on the Hermitian relation, which holds either way.
    for (std::size_t i = 0; i < n; ++i)
        REQUIRE(std::abs(T(i, i).imag()) < 1e-12);
    for (std::size_t i = 0; i + 1 < n; ++i)
        REQUIRE(std::abs(T(i, i + 1) - std::conj(T(i + 1, i))) < 1e-12);

    // Trace preserved (similarity invariant).
    cplx tr_before = 0.0, tr_after = 0.0;
    for (std::size_t i = 0; i < n; ++i) { tr_before += A(i, i); tr_after += T(i, i); }
    REQUIRE(std::abs(tr_after - tr_before) < 1e-12);
}
