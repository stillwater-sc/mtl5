#pragma once
// MTL5 -- Two-sided cyclic Jacobi eigensolver for small dense symmetric matrices
// Rotates every off-diagonal pair to zero, row by row, until each remaining entry
// is negligible relative to the diagonal entries it couples. Header-only and
// generic over the real scalar type; no LAPACK dispatch.
#include <cmath>
#include <algorithm>
#include <cassert>
#include <limits>
#include <utility>
#include <vector>
#include <mtl/concepts/matrix.hpp>
#include <mtl/concepts/scalar.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/math/identity.hpp>

namespace mtl {

/// Eigenvalues and eigenvectors of a real symmetric matrix by the cyclic-by-row
/// two-sided Jacobi method. Returns {eigenvalues, eigenvectors} exactly as
/// eigen_symmetric does: eigenvalues ascending, column k of eigenvectors the
/// unit eigenvector for eigenvalues(k), so that A = V * diag(eigenvalues) * V^T.
///
/// Why Jacobi: for a positive (semi)definite A it computes small eigenvalues,
/// and their eigenvectors, to high RELATIVE accuracy (Demmel & Veselic, 1992),
/// where tridiagonal QR (eigen_symmetric) is accurate only to about eps*||A||
/// in every eigenvalue. That matters for the null vector of a badly scaled
/// normal matrix A^T A. It costs O(n^3) per sweep and takes several sweeps, so
/// it is meant for small matrices: at n = 9..12 it is a few microseconds.
///
/// The rotation for pair (p,q) is the stable one, with
///   theta = (a_qq - a_pp) / (2 a_pq),  t = sign(theta) / (|theta| + sqrt(theta^2 + 1)),
/// computed so that theta^2 cannot overflow. A pair is skipped when
///   |a_pq| <= tol * sqrt(|a_pp|) * sqrt(|a_qq|),
/// the relative test of Demmel & Veselic, and iteration stops after a sweep that
/// rotates no pair (or after max_sweeps sweeps, default 60). This, rather than a
/// stop on off-diagonal mass <= tol^2 * ||A||_F^2, is what keeps small eigenvalues
/// relatively accurate: a Frobenius-norm stop can leave an entry far below
/// eps*||A|| that still couples two small eigenvalues and moves them both. An
/// exactly zero diagonal entry cannot stall the test, since the rotation that
/// zeroes a_pq also makes a_pp nonzero. No entry is ever squared, so matrices
/// scaled to near the overflow or underflow threshold converge too.
///
/// Real element types only (posits and other custom reals included); a complex
/// Hermitian A is rejected at compile time.
template <FieldMatrix M>
auto eigen_symmetric_jacobi(const M& A,
                            typename M::value_type tol =
                                std::numeric_limits<typename M::value_type>::epsilon(),
                            typename M::size_type max_sweeps = 0) {
    using value_type = typename M::value_type;
    using size_type  = typename M::size_type;
    static_assert(!is_complex_v<value_type>,
                  "eigen_symmetric_jacobi: real symmetric matrices only");
    using std::abs;
    using std::sqrt;

    const size_type n = A.num_rows();
    assert(n == A.num_cols());

    struct EigenResult {
        vec::dense_vector<value_type> eigenvalues;
        mat::dense2D<value_type> eigenvectors;
    };

    if (n == 0) return EigenResult{vec::dense_vector<value_type>(0), mat::dense2D<value_type>(0, 0)};
    if (max_sweeps == 0) max_sweeps = 60;

    const value_type zero = math::zero<value_type>();
    const value_type one  = math::one<value_type>();

    mat::dense2D<value_type> S(n, n), V(n, n);
    for (size_type i = 0; i < n; ++i)
        for (size_type j = 0; j < n; ++j) {
            S(i, j) = A(i, j);
            V(i, j) = (i == j) ? one : zero;
        }

    for (size_type sweep = 0; sweep < max_sweeps; ++sweep) {
        bool rotated = false;
        for (size_type p = 0; p + 1 < n; ++p) {
            for (size_type q = p + 1; q < n; ++q) {
                const value_type apq = S(p, q);
                const value_type mag = abs(apq);
                if (mag == zero) continue;
                const value_type app = S(p, p), aqq = S(q, q);
                if (mag <= tol * sqrt(abs(app)) * sqrt(abs(aqq))) continue;

                // theta = (aqq - app) / (2 apq), halved first so the difference
                // cannot overflow; t is the root of smaller magnitude of
                // t^2 + 2 theta t - 1 = 0, formed without squaring a large theta.
                const value_type theta = (aqq / value_type(2) - app / value_type(2)) / apq;
                const value_type r = abs(theta);
                value_type t;
                if (r > one) {
                    const value_type ir = one / r;
                    t = ir / (one + sqrt(one + ir * ir));
                } else {
                    t = one / (r + sqrt(one + r * r));
                }
                if (theta < zero) t = -t;
                const value_type c   = one / sqrt(one + t * t);
                const value_type s   = t * c;
                const value_type tau = s / (one + c);

                S(p, p) = app - t * apq;
                S(q, q) = aqq + t * apq;
                S(p, q) = zero;
                S(q, p) = zero;
                for (size_type k = 0; k < n; ++k) {
                    if (k == p || k == q) continue;
                    const value_type g = S(k, p), h = S(k, q);
                    const value_type gp = g - s * (h + g * tau);
                    const value_type hq = h + s * (g - h * tau);
                    S(k, p) = gp; S(p, k) = gp;
                    S(k, q) = hq; S(q, k) = hq;
                }
                for (size_type k = 0; k < n; ++k) {
                    const value_type g = V(k, p), h = V(k, q);
                    V(k, p) = g - s * (h + g * tau);
                    V(k, q) = h + s * (g - h * tau);
                }
                rotated = true;
            }
        }
        if (!rotated) break;
    }

    // Sort ascending and reorder the eigenvector columns to match.
    std::vector<std::pair<value_type, size_type>> eig_idx(n);
    for (size_type i = 0; i < n; ++i)
        eig_idx[i] = {S(i, i), i};
    std::sort(eig_idx.begin(), eig_idx.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    vec::dense_vector<value_type> eigenvalues(n);
    mat::dense2D<value_type> eigenvectors(n, n);
    for (size_type k = 0; k < n; ++k) {
        eigenvalues(k) = eig_idx[k].first;
        const size_type orig = eig_idx[k].second;
        for (size_type i = 0; i < n; ++i)
            eigenvectors(i, k) = V(i, orig);
    }
    return EigenResult{eigenvalues, eigenvectors};
}

} // namespace mtl
