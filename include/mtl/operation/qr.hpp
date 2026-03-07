#pragma once
// MTL5 -- QR factorization via Householder reflections
// A = Q*R. A is overwritten with R (upper triangular) on+above diagonal,
// Householder vectors stored below diagonal. tau stores the beta scalars.
// Optional LAPACK dispatch when MTL5_HAS_LAPACK is defined and types qualify.
#include <algorithm>
#include <cassert>
#include <vector>
#include <mtl/concepts/matrix.hpp>
#include <mtl/concepts/vector.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/operation/householder.hpp>
#include <mtl/math/identity.hpp>
#include <mtl/interface/dispatch_traits.hpp>
#ifdef MTL5_HAS_LAPACK
#include <mtl/interface/lapack.hpp>
#endif

namespace mtl {

/// QR factorization: A (m x n, m >= n) is overwritten with R on+above diagonal
/// and Householder vectors below diagonal. tau(k) stores beta for column k.
/// Returns 0 on success.
template <Matrix M>
int qr_factor(M& A, vec::dense_vector<typename M::value_type>& tau) {
    using value_type = typename M::value_type;
    using size_type  = typename M::size_type;
    const size_type m = A.num_rows();
    const size_type n = A.num_cols();
    const size_type k = std::min(m, n);
    tau.change_dim(k);

#ifdef MTL5_HAS_LAPACK
    if constexpr (interface::BlasDenseMatrix<M> && !interface::is_row_major_v<M>) {
        int m_int = static_cast<int>(m);
        int n_int = static_cast<int>(n);
        // Workspace query
        value_type work_opt;
        interface::lapack::geqrf(m_int, n_int,
            const_cast<value_type*>(A.data()), m_int,
            tau.data(), &work_opt, -1);
        int lwork = static_cast<int>(work_opt);
        std::vector<value_type> work(lwork);
        int info = interface::lapack::geqrf(m_int, n_int,
            const_cast<value_type*>(A.data()), m_int,
            tau.data(), work.data(), lwork);
        return info;
    }
#endif

    for (size_type j = 0; j < k; ++j) {
        // Extract column A(j:m-1, j)
        size_type len = m - j;
        vec::dense_vector<value_type> col(len);
        for (size_type i = 0; i < len; ++i)
            col(i) = A(j + i, j);

        // Compute Householder reflection
        auto [v, beta] = householder(col);
        tau(j) = beta;

        // Apply reflection to A(j:m-1, j:n-1)
        apply_householder_left(A, v, beta, j, j);

        // Store Householder vector below diagonal (v(0) is implicit 1)
        for (size_type i = 1; i < len; ++i)
            A(j + i, j) = v(i);
    }
    return 0;
}

/// Extract explicit Q (m x m) from QR factorization stored in A and tau.
/// Q = H_0 * H_1 * ... * H_{k-1} where H_j = I - tau(j) * v_j * v_j^T
template <Matrix M>
auto qr_extract_Q(const M& A, const vec::dense_vector<typename M::value_type>& tau) {
    using value_type = typename M::value_type;
    using size_type  = typename M::size_type;
    const size_type m = A.num_rows();
    const size_type n = A.num_cols();
    const size_type k = std::min(m, n);

    // Start with Q = I
    mat::dense2D<value_type> Q(m, m);
    for (size_type i = 0; i < m; ++i)
        for (size_type j = 0; j < m; ++j)
            Q(i, j) = (i == j) ? math::one<value_type>() : math::zero<value_type>();

    // Apply Householder reflections in reverse: Q = H_0 * H_1 * ... * H_{k-1}
    // which means Q *= H_{k-1}, ..., Q *= H_0
    for (size_type jj = 0; jj < k; ++jj) {
        size_type j = k - 1 - jj;
        size_type len = m - j;
        vec::dense_vector<value_type> v(len);
        v(0) = math::one<value_type>();
        for (size_type i = 1; i < len; ++i)
            v(i) = A(j + i, j);

        // Apply H_j to Q from the left: Q = (I - tau*v*v^T) * Q
        apply_householder_left(Q, v, tau(j), j, 0);
    }
    return Q;
}

/// Extract explicit R (m x n) from QR factorization stored in A.
template <Matrix M>
auto qr_extract_R(const M& A) {
    using value_type = typename M::value_type;
    using size_type  = typename M::size_type;
    const size_type m = A.num_rows();
    const size_type n = A.num_cols();

    mat::dense2D<value_type> R(m, n);
    for (size_type i = 0; i < m; ++i)
        for (size_type j = 0; j < n; ++j)
            R(i, j) = (j >= i) ? A(i, j) : math::zero<value_type>();
    return R;
}

/// Solve A*x = b via QR factorization (least-squares for m > n).
/// A must already be factored via qr_factor. x has size n, b has size m.
template <Matrix M, Vector VecX, Vector VecB>
void qr_solve(const M& QR, const vec::dense_vector<typename M::value_type>& tau,
              VecX& x, const VecB& b) {
    using value_type = typename M::value_type;
    using size_type  = typename M::size_type;
    const size_type m = QR.num_rows();
    const size_type n = QR.num_cols();
    const size_type k = std::min(m, n);
    assert(x.size() == n && b.size() == m);

    // y = Q^T * b: apply Householder reflections to b
    vec::dense_vector<value_type> y(m);
    for (size_type i = 0; i < m; ++i)
        y(i) = b(i);

    for (size_type j = 0; j < k; ++j) {
        size_type len = m - j;
        vec::dense_vector<value_type> v(len);
        v(0) = math::one<value_type>();
        for (size_type i = 1; i < len; ++i)
            v(i) = QR(j + i, j);

        // Apply H_j to y(j:m-1): y -= beta * v * (v^T * y)
        value_type dot = math::zero<value_type>();
        for (size_type i = 0; i < len; ++i)
            dot += v(i) * y(j + i);
        for (size_type i = 0; i < len; ++i)
            y(j + i) -= tau(j) * v(i) * dot;
    }

    // Back substitution: R*x = y(0:n-1) where R is upper triangular in QR
    for (size_type ii = 0; ii < n; ++ii) {
        size_type i = n - 1 - ii;
        auto sum = math::zero<value_type>();
        for (size_type j = i + 1; j < n; ++j)
            sum += QR(i, j) * x(j);
        x(i) = (y(i) - sum) / QR(i, i);
    }
}

} // namespace mtl
