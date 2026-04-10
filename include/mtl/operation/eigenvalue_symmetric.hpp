#pragma once
// MTL5 -- Symmetric eigenvalue solver via implicit QR on tridiagonal form
// Tridiagonalize via Householder, then apply Wilkinson-shifted QR iterations.
// Optional LAPACK dispatch when MTL5_HAS_LAPACK is defined and types qualify.
#include <cmath>
#include <algorithm>
#include <cassert>
#include <vector>
#include <mtl/concepts/matrix.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/operation/hessenberg.hpp>
#include <mtl/math/identity.hpp>
#include <mtl/interface/dispatch_traits.hpp>
#ifdef MTL5_HAS_LAPACK
#include <mtl/interface/lapack.hpp>
#endif

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

#ifdef MTL5_HAS_LAPACK
    if constexpr (interface::BlasDenseMatrix<M> && !interface::is_row_major_v<M>) {
        // LAPACK syev: eigenvalues only ('N'), lower triangle ('L')
        // Work on a column-major copy
        std::vector<value_type> A_copy(n * n);
        for (size_type i = 0; i < n; ++i)
            for (size_type j = 0; j < n; ++j)
                A_copy[j * n + i] = A(i, j);

        std::vector<value_type> W(n);
        // Workspace query
        value_type work_opt;
        interface::lapack::syev('N', 'L', static_cast<int>(n),
            A_copy.data(), static_cast<int>(n), W.data(), &work_opt, -1);
        int lwork = static_cast<int>(work_opt);
        std::vector<value_type> work(lwork);
        interface::lapack::syev('N', 'L', static_cast<int>(n),
            A_copy.data(), static_cast<int>(n), W.data(), work.data(), lwork);

        // LAPACK returns eigenvalues in ascending order
        vec::dense_vector<value_type> result(n);
        for (size_type i = 0; i < n; ++i)
            result(i) = W[i];
        return result;
    }
#endif

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

/// Compute eigenvalues and eigenvectors of a symmetric matrix.
/// Returns {eigenvalues, eigenvectors} where:
///   eigenvalues:  dense_vector of size n, sorted ascending
///   eigenvectors: dense2D of size n×n, column k = eigenvector for eigenvalues(k)
///
/// The algorithm accumulates similarity transforms (Householder reflectors
/// from tridiagonalization + Givens rotations from QR iteration) into Q,
/// so that A = Q * diag(eigenvalues) * Q^T.
template <Matrix M>
auto eigen_symmetric(const M& A, typename M::value_type tol = 1e-10,
                     typename M::size_type max_iter = 0) {
    using value_type = typename M::value_type;
    using size_type  = typename M::size_type;
    using std::abs;
    using std::sqrt;
    const size_type n = A.num_rows();
    assert(n == A.num_cols());

    struct EigenResult {
        vec::dense_vector<value_type> eigenvalues;
        mat::dense2D<value_type> eigenvectors;
    };

    if (n == 0) return EigenResult{vec::dense_vector<value_type>(0), mat::dense2D<value_type>(0, 0)};

    if (max_iter == 0) max_iter = 30 * n;

    // Copy A into T for tridiagonalization
    mat::dense2D<value_type> T(n, n);
    for (size_type i = 0; i < n; ++i)
        for (size_type j = 0; j < n; ++j)
            T(i, j) = A(i, j);

    // Initialize Q = I (will accumulate all transforms)
    mat::dense2D<value_type> Q(n, n);
    for (size_type i = 0; i < n; ++i)
        for (size_type j = 0; j < n; ++j)
            Q(i, j) = (i == j) ? math::one<value_type>() : math::zero<value_type>();

    // === Phase 1: Tridiagonalize via Householder, accumulating into Q ===
    if (n >= 3) {
        size_type k = n - 2;
        for (size_type j = 0; j < k; ++j) {
            // Extract column T(j+1:n-1, j)
            size_type len = n - j - 1;
            vec::dense_vector<value_type> col(len);
            for (size_type i = 0; i < len; ++i)
                col(i) = T(j + 1 + i, j);

            auto [v, beta] = householder(col);

            // Apply from left: T = (I - beta*v*v^T) * T
            apply_householder_left(T, v, beta, j + 1, j);
            // Apply from right: T = T * (I - beta*v*v^T)
            apply_householder_right(T, v, beta, 0, j + 1);

            // Accumulate into Q: Q = Q * (I - beta*v*v^T)
            apply_householder_right(Q, v, beta, 0, j + 1);
        }
    }

    // Extract diagonal (d) and subdiagonal (e) from tridiagonal T
    vec::dense_vector<value_type> d(n), e(n);
    for (size_type i = 0; i < n; ++i)
        d(i) = T(i, i);
    for (size_type i = 0; i + 1 < n; ++i)
        e(i) = T(i + 1, i);
    e(n - 1) = math::zero<value_type>();

    // === Phase 2: Implicit QR iteration, accumulating Givens rotations into Q ===
    for (size_type iter = 0; iter < max_iter; ++iter) {
        size_type p = 0;
        size_type q = 0;

        // Find trailing block of zeros (converged eigenvalues)
        for (size_type i = n; i > 1; --i) {
            if (abs(e(i - 2)) > tol * (abs(d(i - 2)) + abs(d(i - 1)))) break;
            e(i - 2) = math::zero<value_type>();
            q++;
        }
        if (q >= n - 1) break;

        size_type end = n - q;

        // Find start of active block
        for (size_type i = end - 1; i > 0; --i) {
            if (abs(e(i - 1)) <= tol * (abs(d(i - 1)) + abs(d(i)))) {
                e(i - 1) = math::zero<value_type>();
                p = i;
                break;
            }
        }

        if (end - p < 2) continue;

        // Wilkinson shift
        value_type a = d(end - 2);
        value_type b = e(end - 2);
        value_type c = d(end - 1);
        value_type delta = (a - c) / value_type(2);
        value_type sign_delta = (delta >= 0) ? value_type(1) : value_type(-1);
        value_type mu = c - b * b / (delta + sign_delta * sqrt(delta * delta + b * b));

        // Implicit QR step with Givens rotations
        value_type x = d(p) - mu;
        value_type z = e(p);

        for (size_type k = p; k + 1 < end; ++k) {
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

            // Accumulate Givens rotation into Q: Q = Q * G(k, k+1, theta)
            // Q(:,k) and Q(:,k+1) updated
            for (size_type i = 0; i < n; ++i) {
                value_type qik  = Q(i, k);
                value_type qik1 = Q(i, k + 1);
                Q(i, k)     = cs * qik + sn * qik1;
                Q(i, k + 1) = -sn * qik + cs * qik1;
            }
        }
    }

    // === Phase 3: Sort eigenvalues and reorder eigenvectors to match ===
    std::vector<std::pair<value_type, size_type>> eig_idx(n);
    for (size_type i = 0; i < n; ++i)
        eig_idx[i] = {d(i), i};
    std::sort(eig_idx.begin(), eig_idx.end());

    vec::dense_vector<value_type> eigenvalues(n);
    mat::dense2D<value_type> eigenvectors(n, n);
    for (size_type k = 0; k < n; ++k) {
        eigenvalues(k) = eig_idx[k].first;
        size_type orig = eig_idx[k].second;
        for (size_type i = 0; i < n; ++i)
            eigenvectors(i, k) = Q(i, orig);
    }

    return EigenResult{eigenvalues, eigenvectors};
}

} // namespace mtl
