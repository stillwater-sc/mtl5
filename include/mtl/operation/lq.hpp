#pragma once
// MTL5 -- LQ factorization via Householder reflections on rows
// A = L*Q. A is overwritten with L (lower triangular) on+below diagonal,
// Householder vectors stored above diagonal. tau stores the beta scalars.
#include <algorithm>
#include <cassert>
#include <mtl/concepts/matrix.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/operation/householder.hpp>
#include <mtl/functor/scalar/conj.hpp>
#include <mtl/math/identity.hpp>

namespace mtl {

/// LQ factorization: A (m x n, n >= m) is overwritten with L on+below diagonal
/// and Householder vectors above diagonal. tau(k) stores beta for row k.
/// Returns 0 on success.
template <Matrix M>
int lq_factor(M& A, vec::dense_vector<typename M::value_type>& tau) {
    using value_type = typename M::value_type;
    using size_type  = typename M::size_type;
    const size_type m = A.num_rows();
    const size_type n = A.num_cols();
    const size_type k = std::min(m, n);
    tau.change_dim(k);

    for (size_type i = 0; i < k; ++i) {
        // Extract row A(i, i:n-1)
        size_type len = n - i;
        // CONJUGATE the row before forming the reflector (LAPACK zgelq2 does
        // the same with zlacgv). householder() guarantees H*c = beta*e_1 for a
        // COLUMN c. Here the thing to annihilate is a row r, and with c = r^H:
        //
        //     r * H^H = c^H * H^H = (H*c)^H = beta * e_1^T      (beta real)
        //
        // so the reflector must be built from r^H and applied on the right as
        // H^H, not H. Skipping either half leaves Q unitary but L*Q != A, which
        // is exactly what the first attempt at this produced.
        // Both conj() calls are the identity for a real T, so the real path is
        // unchanged.
        vec::dense_vector<value_type> row(len);
        for (size_type j = 0; j < len; ++j)
            row(j) = functor::scalar::conj<value_type>::apply(A(i, i + j));

        // Compute Householder reflection
        auto [v, beta] = householder(row);
        tau(i) = beta;

        // Apply reflection on the right to A(i:m-1, i:n-1) as A := A * H^H.
        apply_householder_right(A, v, functor::scalar::conj<value_type>::apply(beta), i, i);

        // Store Householder vector above diagonal (v(0) is implicit 1)
        for (size_type j = 1; j < len; ++j)
            A(i, i + j) = v(j);
    }
    return 0;
}

/// Extract explicit Q (n x n) from LQ factorization stored in A and tau.
template <Matrix M>
auto lq_extract_Q(const M& A, const vec::dense_vector<typename M::value_type>& tau) {
    using value_type = typename M::value_type;
    using size_type  = typename M::size_type;
    const size_type m = A.num_rows();
    const size_type n = A.num_cols();
    const size_type k = std::min(m, n);

    // Start with Q = I (n x n)
    mat::dense2D<value_type> Q(n, n);
    for (size_type i = 0; i < n; ++i)
        for (size_type j = 0; j < n; ++j)
            Q(i, j) = (i == j) ? math::one<value_type>() : math::zero<value_type>();

    // Apply Householder reflections in forward order: Q = H_{k-1} * ... * H_0
    for (size_type i = 0; i < k; ++i) {
        size_type len = n - i;
        vec::dense_vector<value_type> v(len);
        v(0) = math::one<value_type>();
        for (size_type j = 1; j < len; ++j)
            v(j) = A(i, i + j);

        // Apply H_i from the left, with tau -- NOT conj(tau).
        //
        // lq_factor forms L = A H_0^H H_1^H ... H_{k-1}^H, so
        // A = L H_{k-1} ... H_0 and Q = H_{k-1} ... H_0. The adjoint was already
        // taken on the factor side, so taking it again here would undo it. This
        // is the mirror image of qr_extract_Q, where the factor side applies H
        // and the extraction side applies H^H.
        apply_householder_left(Q, v, tau(i), i, 0);
    }
    return Q;
}

/// Extract explicit L (m x n) from LQ factorization stored in A.
template <Matrix M>
auto lq_extract_L(const M& A) {
    using value_type = typename M::value_type;
    using size_type  = typename M::size_type;
    const size_type m = A.num_rows();
    const size_type n = A.num_cols();

    mat::dense2D<value_type> L(m, n);
    for (size_type i = 0; i < m; ++i)
        for (size_type j = 0; j < n; ++j)
            L(i, j) = (j <= i) ? A(i, j) : math::zero<value_type>();
    return L;
}

} // namespace mtl
