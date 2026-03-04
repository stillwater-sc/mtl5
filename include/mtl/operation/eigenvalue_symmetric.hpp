#pragma once
// MTL5 — Symmetric eigenvalue solver via implicit QR on tridiagonal form
// Tridiagonalize via Householder, then apply Wilkinson-shifted QR iterations.
#include <cmath>
#include <algorithm>
#include <cassert>
#include <mtl/concepts/matrix.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/operation/hessenberg.hpp>
#include <mtl/math/identity.hpp>

namespace mtl {

/// Compute eigenvalues of a symmetric matrix via implicit QR on tridiagonal form.
/// Returns eigenvalues as a dense_vector sorted in ascending order.
template <Matrix M>
auto eigenvalue_symmetric(const M& A, typename M::value_type tol = 1e-10,
                          typename M::size_type max_iter = 0) {
    using value_type = typename M::value_type;
    using size_type  = typename M::size_type;
    using std::abs;
    using std::sqrt;
    const size_type n = A.num_rows();
    assert(n == A.num_cols());

    if (max_iter == 0) max_iter = 30 * n;

    // Copy diagonal and subdiagonal from tridiagonalized matrix
    mat::dense2D<value_type> T(n, n);
    for (size_type i = 0; i < n; ++i)
        for (size_type j = 0; j < n; ++j)
            T(i, j) = A(i, j);

    vec::dense_vector<value_type> tau;
    tridiagonalize(T, tau);

    // Extract diagonal (d) and subdiagonal (e) from tridiagonal T
    vec::dense_vector<value_type> d(n), e(n);
    for (size_type i = 0; i < n; ++i)
        d(i) = T(i, i);
    for (size_type i = 0; i + 1 < n; ++i)
        e(i) = T(i + 1, i);
    e(n - 1) = math::zero<value_type>();

    // Implicit QR iteration (symmetric tridiagonal QR with Wilkinson shift)
    for (size_type iter = 0; iter < max_iter; ++iter) {
        // Find the largest unreduced subdiagonal element
        size_type p = 0; // start of active block
        size_type q = 0; // number of converged trailing eigenvalues

        // Find trailing block of zeros (converged eigenvalues)
        for (size_type i = n; i > 1; --i) {
            if (abs(e(i - 2)) > tol * (abs(d(i - 2)) + abs(d(i - 1)))) break;
            e(i - 2) = math::zero<value_type>();
            q++;
        }
        if (q >= n - 1) break; // all converged

        size_type end = n - q; // active block is d[p..end-1]

        // Find start of active block
        for (size_type i = end - 1; i > 0; --i) {
            if (abs(e(i - 1)) <= tol * (abs(d(i - 1)) + abs(d(i)))) {
                e(i - 1) = math::zero<value_type>();
                p = i;
                break;
            }
        }

        if (end - p < 2) continue;

        // Wilkinson shift: eigenvalue of trailing 2x2 block closest to d[end-1]
        value_type a = d(end - 2);
        value_type b = e(end - 2);
        value_type c = d(end - 1);
        value_type delta = (a - c) / value_type(2);
        value_type sign_delta = (delta >= 0) ? value_type(1) : value_type(-1);
        value_type mu = c - b * b / (delta + sign_delta * sqrt(delta * delta + b * b));

        // Implicit QR step with Givens rotations (Golub-Van Loan Algorithm 8.3.2)
        value_type x = d(p) - mu;
        value_type z = e(p);

        for (size_type k = p; k + 1 < end; ++k) {
            // Compute Givens rotation to zero z
            value_type r = sqrt(x * x + z * z);
            value_type cs = x / r;
            value_type sn = z / r;

            if (k > p) e(k - 1) = r;

            value_type d0 = d(k);
            value_type d1 = d(k + 1);
            value_type ek = e(k);

            d(k)     = cs * cs * d0 + value_type(2) * cs * sn * ek + sn * sn * d1;
            d(k + 1) = sn * sn * d0 - value_type(2) * cs * sn * ek + cs * cs * d1;
            e(k)     = cs * sn * (d1 - d0) + (cs * cs - sn * sn) * ek;

            if (k + 2 < end) {
                x = e(k);
                z = sn * e(k + 1);
                e(k + 1) *= cs;
            }
        }
    }

    // Sort eigenvalues
    std::vector<value_type> eigs(n);
    for (size_type i = 0; i < n; ++i)
        eigs[i] = d(i);
    std::sort(eigs.begin(), eigs.end());

    vec::dense_vector<value_type> result(n);
    for (size_type i = 0; i < n; ++i)
        result(i) = eigs[i];
    return result;
}

} // namespace mtl
