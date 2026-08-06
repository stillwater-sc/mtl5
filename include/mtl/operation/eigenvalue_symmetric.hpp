#pragma once
// MTL5 -- Symmetric eigenvalue solver via implicit QR on tridiagonal form
// Tridiagonalize via Householder, then apply Wilkinson-shifted QR iterations.
// Optional LAPACK dispatch when MTL5_HAS_LAPACK is defined and types qualify.
#include <cmath>
#include <complex>
#include <algorithm>
#include <cassert>
#include <vector>
#include <mtl/concepts/matrix.hpp>
#include <mtl/concepts/scalar.hpp>
#include <mtl/concepts/magnitude.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/operation/hessenberg.hpp>
#include <mtl/functor/scalar/real.hpp>
#include <mtl/functor/scalar/conj.hpp>
#include <mtl/math/identity.hpp>
#include <mtl/interface/dispatch_traits.hpp>
#ifdef MTL5_HAS_LAPACK
#include <mtl/interface/lapack.hpp>
#endif

namespace mtl {

/// Generic (LAPACK-free) symmetric/Hermitian eigenvalue solver: tridiagonalize,
/// then run implicit QR with Wilkinson shifts on the tridiagonal form. Returns
/// the eigenvalues as a dense_vector sorted in ascending order. The eigenvalues
/// of a Hermitian matrix are real, so the result element type is the magnitude
/// type magnitude_t<value_type> (== value_type for a real matrix).
///
/// For a complex Hermitian A the reduction yields a Hermitian tridiagonal whose
/// subdiagonal carries an arbitrary phase. Replacing each subdiagonal entry by
/// its modulus is a diagonal unitary similarity D^H T D, which leaves the
/// spectrum unchanged -- so eigenVALUES need no phase bookkeeping. (Eigenvectors
/// do: see eigen_symmetric, which accumulates D.)
///
/// This is the C++ reference path. `eigenvalue_symmetric` dispatches to LAPACK
/// when available and otherwise calls this; benchmarks and tests can call this
/// directly to exercise the generic algorithm regardless of MTL5_HAS_LAPACK.
template <Matrix M>
auto eigenvalue_symmetric_generic(const M& A,
                                  magnitude_t<typename M::value_type> tol = 1e-10,
                                  typename M::size_type max_iter = 0) {

    using value_type = typename M::value_type;
    using size_type  = typename M::size_type;
    using real_t     = magnitude_t<value_type>;
    using std::abs;
    using std::sqrt;
    const size_type n = A.num_rows();
    assert(n == A.num_cols());

    // No eigenvalues for a 0x0 matrix. Return before the e(n-1) sentinel write
    // below, which would otherwise underflow the unsigned index to n == 0.
    if (n == 0) return vec::dense_vector<real_t>(0);

    if (max_iter == 0) max_iter = 30 * n;

    // Copy diagonal and subdiagonal from tridiagonalized matrix
    mat::dense2D<value_type> T(n, n);
    for (size_type i = 0; i < n; ++i)
        for (size_type j = 0; j < n; ++j)
            T(i, j) = A(i, j);

    vec::dense_vector<value_type> tau;
    tridiagonalize(T, tau);

    // Extract the real diagonal (d) and real subdiagonal (e) of the tridiagonal.
    // The diagonal of a Hermitian tridiagonal is real; real() is the identity
    // for a real value_type, so this stays bit-exact there. The subdiagonal of a
    // complex Hermitian tridiagonal has a phase -- take its modulus (the D^H T D
    // similarity above). For a real value_type abs(...) would drop the sign and
    // change the arithmetic, so the real branch keeps the signed entry verbatim.
    vec::dense_vector<real_t> d(n), e(n);
    for (size_type i = 0; i < n; ++i)
        d(i) = functor::scalar::real<value_type>::apply(T(i, i));
    for (size_type i = 0; i + 1 < n; ++i) {
        if constexpr (is_complex_v<value_type>)
            e(i) = abs(T(i + 1, i));
        else
            e(i) = T(i + 1, i);
    }
    e(n - 1) = math::zero<real_t>();

    // Implicit QR iteration (symmetric tridiagonal QR with Wilkinson shift)
    for (size_type iter = 0; iter < max_iter; ++iter) {
        // Find the largest unreduced subdiagonal element
        size_type p = 0; // start of active block
        size_type q = 0; // number of converged trailing eigenvalues

        // Find trailing block of zeros (converged eigenvalues)
        for (size_type i = n; i > 1; --i) {
            if (abs(e(i - 2)) > tol * (abs(d(i - 2)) + abs(d(i - 1)))) break;
            e(i - 2) = math::zero<real_t>();
            q++;
        }
        if (q >= n - 1) break; // all converged

        size_type end = n - q; // active block is d[p..end-1]

        // Find start of active block
        for (size_type i = end - 1; i > 0; --i) {
            if (abs(e(i - 1)) <= tol * (abs(d(i - 1)) + abs(d(i)))) {
                e(i - 1) = math::zero<real_t>();
                p = i;
                break;
            }
        }

        if (end - p < 2) continue;

        // Wilkinson shift: eigenvalue of trailing 2x2 block closest to d[end-1].
        // The tridiagonal is real by construction, so all iteration scalars are
        // real_t (== value_type for a real matrix, so the arithmetic is unchanged).
        real_t a = d(end - 2);
        real_t b = e(end - 2);
        real_t c = d(end - 1);
        real_t delta = (a - c) / real_t(2);
        real_t sign_delta = (delta >= 0) ? real_t(1) : real_t(-1);
        real_t mu = c - b * b / (delta + sign_delta * sqrt(delta * delta + b * b));

        // Implicit QR step with Givens rotations (Golub-Van Loan Algorithm 8.3.2)
        real_t x = d(p) - mu;
        real_t z = e(p);

        for (size_type k = p; k + 1 < end; ++k) {
            // Compute Givens rotation to zero z
            real_t r = sqrt(x * x + z * z);
            real_t cs = x / r;
            real_t sn = z / r;

            if (k > p) e(k - 1) = r;

            real_t d0 = d(k);
            real_t d1 = d(k + 1);
            real_t ek = e(k);

            d(k)     = cs * cs * d0 + real_t(2) * cs * sn * ek + sn * sn * d1;
            d(k + 1) = sn * sn * d0 - real_t(2) * cs * sn * ek + cs * cs * d1;
            e(k)     = cs * sn * (d1 - d0) + (cs * cs - sn * sn) * ek;

            if (k + 2 < end) {
                x = e(k);
                z = sn * e(k + 1);
                e(k + 1) *= cs;
            }
        }
    }

    // Sort eigenvalues
    std::vector<real_t> eigs(n);
    for (size_type i = 0; i < n; ++i)
        eigs[i] = d(i);
    std::sort(eigs.begin(), eigs.end());

    vec::dense_vector<real_t> result(n);
    for (size_type i = 0; i < n; ++i)
        result(i) = eigs[i];
    return result;
}

/// Compute eigenvalues of a symmetric/Hermitian matrix via implicit QR on
/// tridiagonal form. Returns eigenvalues (real: magnitude_t<value_type>) sorted
/// in ascending order. Dispatches to LAPACK syev when available and the type
/// qualifies (real float/double); complex Hermitian input falls through to
/// eigenvalue_symmetric_generic.
template <Matrix M>
auto eigenvalue_symmetric(const M& A,
                          magnitude_t<typename M::value_type> tol = 1e-10,
                          typename M::size_type max_iter = 0) {
#ifdef MTL5_HAS_LAPACK
    if constexpr (interface::BlasDenseMatrix<M> && !interface::is_row_major_v<M>) {
        using value_type = typename M::value_type;
        using size_type  = typename M::size_type;
        const size_type n = A.num_rows();
        assert(n == A.num_cols());
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

    return eigenvalue_symmetric_generic(A, tol, max_iter);
}

/// Compute eigenvalues and eigenvectors of a symmetric or Hermitian matrix.
/// Returns {eigenvalues, eigenvectors} where:
///   eigenvalues:  dense_vector of size n, sorted ascending. The eigenvalues of
///                 a Hermitian matrix are real, so the element type is the
///                 magnitude type magnitude_t<value_type> (== value_type real).
///   eigenvectors: dense2D of size n x n, column k = eigenvector for eigenvalues(k)
///
/// so that A = Q * diag(eigenvalues) * Q^H (Q^T for a real matrix).
///
/// For complex Hermitian A the reduction is the UNITARY similarity A -> H*A*H^H
/// (H^H = H^-1), and the Hermitian tridiagonal's phased subdiagonal is made real
/// by a diagonal unitary D so the real symmetric QR can run. D preserves the
/// eigenVALUES but rotates each eigenVECTOR by a per-row phase, so it must be
/// folded into Q -- skip it and eigenvalues still look right while A*v != lambda*v.
/// For a real value_type conj() is the identity, D is the identity, real_t is
/// value_type, and the arithmetic is bit-identical to the earlier H*A*H code.
template <Matrix M>
auto eigen_symmetric(const M& A,
                     magnitude_t<typename M::value_type> tol = 1e-10,
                     typename M::size_type max_iter = 0) {
    using value_type = typename M::value_type;
    using size_type  = typename M::size_type;
    using real_t     = magnitude_t<value_type>;
    using conj_t     = functor::scalar::conj<value_type>;
    using std::abs;
    using std::sqrt;
    const size_type n = A.num_rows();
    assert(n == A.num_cols());

    struct EigenResult {
        vec::dense_vector<real_t> eigenvalues;
        mat::dense2D<value_type> eigenvectors;
    };

    if (n == 0) return EigenResult{vec::dense_vector<real_t>(0), mat::dense2D<value_type>(0, 0)};

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

            // T -> H*T*H^H: left factor H (I - beta*v*v^H), right factor H^H
            // (conj(beta)); conj is the identity for a real value_type.
            apply_householder_left(T, v, beta, j + 1, j);
            apply_householder_right(T, v, conj_t::apply(beta), 0, j + 1);

            // Accumulate Q = H_0^H * H_1^H * ... so that A = Q*T*Q^H. Building it
            // by right-multiplication in forward order gives Q *= H_j^H, i.e.
            // conj(beta) again -- the qr_extract_Q / lq_extract_Q adjoint pattern.
            apply_householder_right(Q, v, conj_t::apply(beta), 0, j + 1);
        }
    }

    // Extract the real diagonal (d) and real subdiagonal (e) of the tridiagonal.
    // For a complex Hermitian T the subdiagonal T(i+1,i) carries a phase; scaling
    // it to its modulus is the diagonal unitary similarity D^H*T*D. D is folded
    // into Q column by column (Q(:,m) *= d_m) so the eigenvectors come out with
    // the right phase; without it A*v != lambda*v even though the eigenvalues are
    // correct. For a real value_type D is the identity, real() and abs-avoidance
    // keep the signed subdiagonal, and this is bit-identical to the old code.
    vec::dense_vector<real_t> d(n), e(n);
    for (size_type i = 0; i < n; ++i)
        d(i) = functor::scalar::real<value_type>::apply(T(i, i));
    if constexpr (is_complex_v<value_type>) {
        value_type phase = math::one<value_type>();   // d_0 = 1, Q(:,0) unchanged
        for (size_type m = 1; m < n; ++m) {
            const value_type sub = T(m, m - 1);        // e_{m-1}
            const real_t     mag = abs(sub);
            e(m - 1) = mag;
            if (mag > real_t(0))
                phase = phase * (sub / value_type(mag)); // d_m = d_{m-1}*e/|e|
            for (size_type i = 0; i < n; ++i)
                Q(i, m) = Q(i, m) * phase;
        }
    } else {
        for (size_type i = 0; i + 1 < n; ++i)
            e(i) = T(i + 1, i);
    }
    e(n - 1) = math::zero<real_t>();

    // === Phase 2: Implicit QR iteration, accumulating Givens rotations into Q ===
    for (size_type iter = 0; iter < max_iter; ++iter) {
        size_type p = 0;
        size_type q = 0;

        // Find trailing block of zeros (converged eigenvalues)
        for (size_type i = n; i > 1; --i) {
            if (abs(e(i - 2)) > tol * (abs(d(i - 2)) + abs(d(i - 1)))) break;
            e(i - 2) = math::zero<real_t>();
            q++;
        }
        if (q >= n - 1) break;

        size_type end = n - q;

        // Find start of active block
        for (size_type i = end - 1; i > 0; --i) {
            if (abs(e(i - 1)) <= tol * (abs(d(i - 1)) + abs(d(i)))) {
                e(i - 1) = math::zero<real_t>();
                p = i;
                break;
            }
        }

        if (end - p < 2) continue;

        // Wilkinson shift. The tridiagonal is real, so every iteration scalar is
        // real_t (== value_type for a real matrix, so the arithmetic is unchanged).
        real_t a = d(end - 2);
        real_t b = e(end - 2);
        real_t c = d(end - 1);
        real_t delta = (a - c) / real_t(2);
        real_t sign_delta = (delta >= 0) ? real_t(1) : real_t(-1);
        real_t mu = c - b * b / (delta + sign_delta * sqrt(delta * delta + b * b));

        // Implicit QR step with Givens rotations
        real_t x = d(p) - mu;
        real_t z = e(p);

        for (size_type k = p; k + 1 < end; ++k) {
            real_t r = sqrt(x * x + z * z);
            real_t cs = x / r;
            real_t sn = z / r;

            if (k > p) e(k - 1) = r;

            real_t d0 = d(k);
            real_t d1 = d(k + 1);
            real_t ek = e(k);

            d(k)     = cs * cs * d0 + real_t(2) * cs * sn * ek + sn * sn * d1;
            d(k + 1) = sn * sn * d0 - real_t(2) * cs * sn * ek + cs * cs * d1;
            e(k)     = cs * sn * (d1 - d0) + (cs * cs - sn * sn) * ek;

            if (k + 2 < end) {
                x = e(k);
                z = sn * e(k + 1);
                e(k + 1) *= cs;
            }

            // Accumulate the (real) Givens rotation into Q: Q = Q * G(k,k+1).
            // cs/sn are real_t; Q may be complex (it already carries the unitary
            // reduction and the phase D), and real*complex is well-formed.
            for (size_type i = 0; i < n; ++i) {
                value_type qik  = Q(i, k);
                value_type qik1 = Q(i, k + 1);
                Q(i, k)     = cs * qik + sn * qik1;
                Q(i, k + 1) = -sn * qik + cs * qik1;
            }
        }
    }

    // === Phase 3: Sort eigenvalues and reorder eigenvectors to match ===
    std::vector<std::pair<real_t, size_type>> eig_idx(n);
    for (size_type i = 0; i < n; ++i)
        eig_idx[i] = {d(i), i};
    std::sort(eig_idx.begin(), eig_idx.end());

    vec::dense_vector<real_t> eigenvalues(n);
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
