#pragma once
// MTL5 -- General eigenvalue computation via QR algorithm on Hessenberg form
// Reduce to upper Hessenberg, then apply implicit QR with single/double shifts.
#include <cmath>
#include <complex>
#include <algorithm>
#include <cassert>
#include <limits>
#include <vector>
#include <mtl/concepts/matrix.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/operation/hessenberg.hpp>
#include <mtl/math/identity.hpp>

namespace mtl {

namespace detail {

/// In-place LU factorization of an n x n complex matrix (row-major, `M[i*n+j]`)
/// with partial pivoting. Tiny pivots are floored to `pivot_floor` so that a
/// (near-)singular system stays solvable -- exactly the regime that inverse
/// iteration relies on, since A - lambda*I is singular at a true eigenvalue.
/// `piv[i]` holds the original row now living in row i after pivoting.
template <typename Complex, typename Real>
void eig_lu_factor(std::vector<Complex>& M, std::vector<std::size_t>& piv,
                   std::size_t n, Real pivot_floor) {
    using std::abs;
    for (std::size_t i = 0; i < n; ++i) piv[i] = i;
    for (std::size_t k = 0; k < n; ++k) {
        // Partial pivot: largest-magnitude entry in column k at or below k.
        std::size_t p = k;
        Real best = abs(M[k * n + k]);
        for (std::size_t i = k + 1; i < n; ++i) {
            Real v = abs(M[i * n + k]);
            if (v > best) { best = v; p = i; }
        }
        if (p != k) {
            for (std::size_t j = 0; j < n; ++j)
                std::swap(M[k * n + j], M[p * n + j]);
            std::swap(piv[k], piv[p]);
        }
        Complex akk = M[k * n + k];
        if (abs(akk) < pivot_floor) { akk = Complex(pivot_floor); M[k * n + k] = akk; }
        for (std::size_t i = k + 1; i < n; ++i) {
            Complex f = M[i * n + k] / akk;
            M[i * n + k] = f;
            for (std::size_t j = k + 1; j < n; ++j)
                M[i * n + j] -= f * M[k * n + j];
        }
    }
}

/// Solve M x = b in place given the LU factors from eig_lu_factor.
template <typename Complex>
void eig_lu_solve(const std::vector<Complex>& M, const std::vector<std::size_t>& piv,
                  std::size_t n, std::vector<Complex>& b) {
    std::vector<Complex> x(n);
    for (std::size_t i = 0; i < n; ++i) x[i] = b[piv[i]];
    // Forward substitution (unit lower triangle).
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < i; ++j)
            x[i] -= M[i * n + j] * x[j];
    // Back substitution (upper triangle).
    for (std::size_t i = n; i-- > 0; ) {
        for (std::size_t j = i + 1; j < n; ++j)
            x[i] -= M[i * n + j] * x[j];
        x[i] /= M[i * n + i];
    }
    b = std::move(x);
}

} // namespace detail

/// Compute eigenvalues of a general (non-symmetric) square matrix.
/// Returns eigenvalues as dense_vector of complex values.
template <Matrix M>
auto eigenvalue(const M& A, typename M::value_type tol = 1e-10,
                typename M::size_type max_iter = 0) {
    using value_type = typename M::value_type;
    using complex_type = std::complex<value_type>;
    using size_type  = typename M::size_type;
    using std::abs;
    using std::sqrt;
    const size_type n = A.num_rows();
    assert(n == A.num_cols());

    if (max_iter == 0) max_iter = 30 * n * n;

    // Reduce to upper Hessenberg form
    mat::dense2D<value_type> H = hessenberg(A);

    vec::dense_vector<complex_type> eigs(n);

    // Francis QR iteration with single/double shifts
    size_type nn = n;

    for (size_type iter = 0; iter < max_iter && nn > 0; ++iter) {
        // Check for convergence of last subdiagonal element
        if (nn == 1) {
            eigs(nn - 1) = complex_type(H(0, 0));
            nn = 0;
            break;
        }

        // Deflation: check if H(nn-1, nn-2) is small enough
        value_type threshold = tol * (abs(H(nn - 2, nn - 2)) + abs(H(nn - 1, nn - 1)));
        if (threshold == value_type(0)) threshold = tol;

        if (abs(H(nn - 1, nn - 2)) <= threshold) {
            // Single eigenvalue converged
            eigs(nn - 1) = complex_type(H(nn - 1, nn - 1));
            nn--;
            continue;
        }

        if (nn == 2) {
            // Extract eigenvalues of 2x2 block
            value_type a = H(0, 0), b = H(0, 1);
            value_type c = H(1, 0), d = H(1, 1);
            value_type tr = a + d;
            value_type det = a * d - b * c;
            value_type disc = tr * tr - value_type(4) * det;
            if (disc >= value_type(0)) {
                value_type sq = sqrt(disc);
                eigs(0) = complex_type((tr + sq) / value_type(2));
                eigs(1) = complex_type((tr - sq) / value_type(2));
            } else {
                value_type sq = sqrt(-disc);
                eigs(0) = complex_type(tr / value_type(2), sq / value_type(2));
                eigs(1) = complex_type(tr / value_type(2), -sq / value_type(2));
            }
            nn = 0;
            break;
        }

        // Also check for 2x2 block deflation at bottom
        if (nn >= 3) {
            value_type thr2 = tol * (abs(H(nn - 3, nn - 3)) + abs(H(nn - 2, nn - 2)));
            if (thr2 == value_type(0)) thr2 = tol;
            if (abs(H(nn - 2, nn - 3)) <= thr2) {
                // 2x2 block at bottom has decoupled
                value_type a = H(nn - 2, nn - 2), b = H(nn - 2, nn - 1);
                value_type c = H(nn - 1, nn - 2), d = H(nn - 1, nn - 1);
                value_type tr = a + d;
                value_type det = a * d - b * c;
                value_type disc = tr * tr - value_type(4) * det;
                if (disc >= value_type(0)) {
                    value_type sq = sqrt(disc);
                    eigs(nn - 2) = complex_type((tr + sq) / value_type(2));
                    eigs(nn - 1) = complex_type((tr - sq) / value_type(2));
                } else {
                    value_type sq = sqrt(-disc);
                    eigs(nn - 2) = complex_type(tr / value_type(2), sq / value_type(2));
                    eigs(nn - 1) = complex_type(tr / value_type(2), -sq / value_type(2));
                }
                nn -= 2;
                continue;
            }
        }

        // Single-shift QR step using Wilkinson shift
        value_type a = H(nn - 2, nn - 2), b = H(nn - 2, nn - 1);
        value_type c = H(nn - 1, nn - 2), d = H(nn - 1, nn - 1);
        value_type tr = a + d;
        value_type det = a * d - b * c;
        value_type disc = tr * tr - value_type(4) * det;

        value_type shift;
        if (disc >= value_type(0)) {
            // Real eigenvalues -- pick shift closest to H(nn-1,nn-1)
            value_type sq = sqrt(disc);
            value_type e1 = (tr + sq) / value_type(2);
            value_type e2 = (tr - sq) / value_type(2);
            shift = (abs(e1 - d) < abs(e2 - d)) ? e1 : e2;
        } else {
            // Complex eigenvalues -- use d as shift
            shift = d;
        }

        // Apply shifted QR step via Givens rotations
        value_type x = H(0, 0) - shift;
        value_type z = H(1, 0);

        for (size_type k = 0; k + 1 < nn; ++k) {
            // Compute Givens rotation
            value_type r = sqrt(x * x + z * z);
            if (r == value_type(0)) { r = tol; }
            value_type cs = x / r;
            value_type sn = z / r;

            // Apply rotation from left: rows k, k+1
            for (size_type j = 0; j < nn; ++j) {
                value_type t1 = H(k, j);
                value_type t2 = H(k + 1, j);
                H(k, j)     = cs * t1 + sn * t2;
                H(k + 1, j) = -sn * t1 + cs * t2;
            }

            // Apply rotation from right: cols k, k+1
            size_type limit = std::min(k + 3, nn);
            for (size_type i = 0; i < limit; ++i) {
                value_type t1 = H(i, k);
                value_type t2 = H(i, k + 1);
                H(i, k)     = cs * t1 + sn * t2;
                H(i, k + 1) = -sn * t1 + cs * t2;
            }

            // Update bulge chase
            if (k + 2 < nn) {
                x = H(k + 1, k);
                z = H(k + 2, k);
            }
        }
    }

    // Any remaining eigenvalues on the diagonal
    for (size_type i = 0; i < nn; ++i)
        eigs(i) = complex_type(H(i, i));

    return eigs;
}

/// Compute eigenvalues AND right eigenvectors of a general (non-symmetric)
/// square matrix. Returns {eigenvalues, eigenvectors} where:
///   eigenvalues:  dense_vector<complex> of size n
///   eigenvectors: dense2D<complex> of size n x n, column k is the right
///                 eigenvector for eigenvalues(k), i.e. A * v_k = lambda_k * v_k
///
/// Eigenvalues come from the tested general QR path (`eigenvalue`); each
/// eigenvector is then recovered by inverse iteration on A - lambda_k*I. This
/// handles real and complex-conjugate eigenpairs uniformly in complex
/// arithmetic. Eigenvectors are unit 2-norm with a canonical phase (largest
/// entry made real-positive). Eigenvalue/eigenvector pairing matches by column.
///
/// Accuracy is bounded by the eigenvalues from `eigenvalue`: inverse iteration
/// recovers the eigenvector of the true eigenvalue nearest each computed
/// lambda_k. Strongly non-normal matrices whose complex eigenvalues require a
/// double-shift (Francis) QR step are not yet resolved by the single-shift
/// `eigenvalue` path and are correspondingly inaccurate here (tracked in the
/// Francis double-shift issue #209 / general-eigenproblem LAPACK work #204).
template <Matrix M>
auto eigen(const M& A, typename M::value_type tol = 1e-10,
           typename M::size_type max_iter = 0) {
    using value_type   = typename M::value_type;
    using complex_type = std::complex<value_type>;
    using size_type    = typename M::size_type;
    using std::abs;
    using std::sqrt;
    const size_type n = A.num_rows();
    assert(n == A.num_cols());

    struct EigenResult {
        vec::dense_vector<complex_type> eigenvalues;
        mat::dense2D<complex_type> eigenvectors;
    };

    if (n == 0)
        return EigenResult{vec::dense_vector<complex_type>(0),
                           mat::dense2D<complex_type>(0, 0)};

    // Eigenvalues via the general QR iteration.
    vec::dense_vector<complex_type> eigs = eigenvalue(A, tol, max_iter);

    mat::dense2D<complex_type> V(n, n);
    if (n == 1) {
        V(0, 0) = complex_type(value_type(1));
        return EigenResult{eigs, V};
    }

    // Frobenius norm of A -- sets the scale for the pivot floor.
    value_type anorm = value_type(0);
    for (size_type i = 0; i < n; ++i)
        for (size_type j = 0; j < n; ++j)
            anorm += A(i, j) * A(i, j);
    anorm = sqrt(anorm);
    if (anorm == value_type(0)) anorm = value_type(1);
    const value_type eps = std::numeric_limits<value_type>::epsilon();
    const value_type pivot_floor = eps * anorm;

    const std::size_t nn = static_cast<std::size_t>(n);
    const int max_inv_iter = 5;

    for (size_type k = 0; k < n; ++k) {
        const complex_type lambda = eigs(k);

        // Build M = A - lambda*I (row-major complex) and factor once.
        std::vector<complex_type> Mc(nn * nn);
        for (size_type i = 0; i < n; ++i)
            for (size_type j = 0; j < n; ++j)
                Mc[static_cast<std::size_t>(i) * nn + static_cast<std::size_t>(j)] =
                    complex_type(A(i, j)) - (i == j ? lambda : complex_type(0));

        std::vector<std::size_t> piv(nn);
        detail::eig_lu_factor(Mc, piv, nn, pivot_floor);

        // Start vector: entries varied by index so that repeated/clustered
        // eigenvalues are seeded with non-parallel starts.
        std::vector<complex_type> x(nn);
        for (size_type i = 0; i < n; ++i)
            x[static_cast<std::size_t>(i)] =
                complex_type(value_type(1) + value_type(i) / value_type(n));

        for (int it = 0; it < max_inv_iter; ++it) {
            detail::eig_lu_solve(Mc, piv, nn, x);
            value_type nrm = value_type(0);
            for (std::size_t i = 0; i < nn; ++i) nrm += std::norm(x[i]);
            nrm = sqrt(nrm);
            if (nrm == value_type(0)) break;
            for (std::size_t i = 0; i < nn; ++i) x[i] /= nrm;

            // Residual ||A x - lambda x|| against the (unperturbed) eigenvalue.
            value_type res = value_type(0);
            for (size_type i = 0; i < n; ++i) {
                complex_type axi(0);
                for (size_type j = 0; j < n; ++j)
                    axi += complex_type(A(i, j)) * x[static_cast<std::size_t>(j)];
                axi -= lambda * x[static_cast<std::size_t>(i)];
                res += std::norm(axi);
            }
            if (sqrt(res) <= tol * anorm) break;
        }

        // Canonical phase: rotate so the largest-magnitude entry is real-positive.
        std::size_t imax = 0;
        value_type mmax = value_type(0);
        for (std::size_t i = 0; i < nn; ++i) {
            value_type m = abs(x[i]);
            if (m > mmax) { mmax = m; imax = i; }
        }
        if (mmax > value_type(0)) {
            complex_type phase = x[imax] / abs(x[imax]);
            for (std::size_t i = 0; i < nn; ++i) x[i] /= phase;
        }

        for (size_type i = 0; i < n; ++i)
            V(i, k) = x[static_cast<std::size_t>(i)];
    }

    return EigenResult{eigs, V};
}

} // namespace mtl
