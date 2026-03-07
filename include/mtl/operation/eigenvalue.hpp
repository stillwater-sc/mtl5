#pragma once
// MTL5 -- General eigenvalue computation via QR algorithm on Hessenberg form
// Reduce to upper Hessenberg, then apply implicit QR with single/double shifts.
#include <cmath>
#include <complex>
#include <algorithm>
#include <cassert>
#include <mtl/concepts/matrix.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/operation/hessenberg.hpp>
#include <mtl/math/identity.hpp>

namespace mtl {

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

} // namespace mtl
