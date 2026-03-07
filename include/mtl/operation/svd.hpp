#pragma once
// MTL5 -- Singular Value Decomposition via one-sided Jacobi / iterative QR
// Decomposes A = U * S * V^T where S is diagonal (singular values).
// Optional LAPACK dispatch when MTL5_HAS_LAPACK is defined and types qualify.
#include <cmath>
#include <algorithm>
#include <cassert>
#include <vector>
#include <mtl/concepts/matrix.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/operation/qr.hpp>
#include <mtl/operation/trans.hpp>
#include <mtl/math/identity.hpp>
#include <mtl/interface/dispatch_traits.hpp>
#ifdef MTL5_HAS_LAPACK
#include <mtl/interface/lapack.hpp>
#endif

namespace mtl {

/// SVD: decompose A (m x n) = U * S * V^T
/// U is m x m orthogonal, S is m x n diagonal (stored as m x n matrix), V is n x n orthogonal.
/// Uses iterative QR approach (Golub-Van Loan style).
template <Matrix M>
void svd(const M& A,
         mat::dense2D<typename M::value_type>& U,
         mat::dense2D<typename M::value_type>& S,
         mat::dense2D<typename M::value_type>& V,
         typename M::value_type tol = 1e-10) {
    using value_type = typename M::value_type;
    using size_type  = typename M::size_type;
    using std::abs;
    using std::sqrt;

    const size_type m = A.num_rows();
    const size_type n = A.num_cols();
    const size_type mn = std::min(m, n);

#ifdef MTL5_HAS_LAPACK
    if constexpr (interface::BlasDenseMatrix<M> && !interface::is_row_major_v<M>) {
        // LAPACK gesdd: A_copy is overwritten, returns U, S_vec, VT
        U.change_dim(m, m);
        S.change_dim(m, n);
        V.change_dim(n, n);

        // Work on a copy (LAPACK overwrites input)
        std::vector<value_type> A_copy(m * n);
        for (size_type i = 0; i < m; ++i)
            for (size_type j = 0; j < n; ++j)
                A_copy[j * m + i] = A(i, j);  // column-major copy

        std::vector<value_type> S_vec(mn);
        std::vector<value_type> U_data(m * m);
        std::vector<value_type> VT_data(n * n);
        std::vector<int> iwork(8 * mn);

        // Workspace query
        value_type work_opt;
        interface::lapack::gesdd('A', static_cast<int>(m), static_cast<int>(n),
            A_copy.data(), static_cast<int>(m), S_vec.data(),
            U_data.data(), static_cast<int>(m),
            VT_data.data(), static_cast<int>(n),
            &work_opt, -1, iwork.data());
        int lwork = static_cast<int>(work_opt);
        std::vector<value_type> work(lwork);
        interface::lapack::gesdd('A', static_cast<int>(m), static_cast<int>(n),
            A_copy.data(), static_cast<int>(m), S_vec.data(),
            U_data.data(), static_cast<int>(m),
            VT_data.data(), static_cast<int>(n),
            work.data(), lwork, iwork.data());

        // Copy results: U from column-major U_data
        for (size_type i = 0; i < m; ++i)
            for (size_type j = 0; j < m; ++j)
                U(i, j) = U_data[j * m + i];

        // S as diagonal m x n matrix
        for (size_type i = 0; i < m; ++i)
            for (size_type j = 0; j < n; ++j)
                S(i, j) = (i == j && i < mn) ? S_vec[i] : math::zero<value_type>();

        // V from VT (transpose): V(i,j) = VT(j,i) = VT_data[i * n + j]
        for (size_type i = 0; i < n; ++i)
            for (size_type j = 0; j < n; ++j)
                V(i, j) = VT_data[i * n + j];

        return;
    }
#endif

    const size_type max_iter = 100 * std::max(m, n);

    // Initialize: U = I(m), V = I(n), W = A
    U.change_dim(m, m);
    V.change_dim(n, n);
    S.change_dim(m, n);

    for (size_type i = 0; i < m; ++i)
        for (size_type j = 0; j < m; ++j)
            U(i, j) = (i == j) ? math::one<value_type>() : math::zero<value_type>();

    for (size_type i = 0; i < n; ++i)
        for (size_type j = 0; j < n; ++j)
            V(i, j) = (i == j) ? math::one<value_type>() : math::zero<value_type>();

    // Working matrix W starts as A
    mat::dense2D<value_type> W(m, n);
    for (size_type i = 0; i < m; ++i)
        for (size_type j = 0; j < n; ++j)
            W(i, j) = A(i, j);

    // Iterative QR sweeps: alternating QR on W and W^T
    for (size_type iter = 0; iter < max_iter; ++iter) {
        // QR decompose W = Q1 * R1
        mat::dense2D<value_type> W1(m, n);
        for (size_type i = 0; i < m; ++i)
            for (size_type j = 0; j < n; ++j)
                W1(i, j) = W(i, j);

        vec::dense_vector<value_type> tau1;
        qr_factor(W1, tau1);
        auto Q1 = qr_extract_Q(W1, tau1);
        auto R1 = qr_extract_R(W1);

        // U = U * Q1
        auto Unew = U * Q1;
        for (size_type i = 0; i < m; ++i)
            for (size_type j = 0; j < m; ++j)
                U(i, j) = Unew(i, j);

        // Transpose R1 for next QR
        mat::dense2D<value_type> R1t(n, m);
        for (size_type i = 0; i < m; ++i)
            for (size_type j = 0; j < n; ++j)
                R1t(j, i) = R1(i, j);

        // QR decompose R1^T = Q2 * R2
        vec::dense_vector<value_type> tau2;
        qr_factor(R1t, tau2);
        auto Q2 = qr_extract_Q(R1t, tau2);
        auto R2 = qr_extract_R(R1t);

        // V = V * Q2
        auto Vnew = V * Q2;
        for (size_type i = 0; i < n; ++i)
            for (size_type j = 0; j < n; ++j)
                V(i, j) = Vnew(i, j);

        // W = R2^T for next iteration
        for (size_type i = 0; i < m; ++i)
            for (size_type j = 0; j < n; ++j)
                W(i, j) = R2(j, i);

        // Check convergence: off-diagonal elements of R2 should be small
        value_type off_norm = math::zero<value_type>();
        value_type diag_norm = math::zero<value_type>();
        for (size_type i = 0; i < std::min(n, m); ++i) {
            diag_norm += abs(R2(i, i));
            for (size_type j = i + 1; j < m; ++j)
                off_norm += abs(R2(i, j));
        }
        if (diag_norm > math::zero<value_type>() && off_norm / diag_norm < tol)
            break;
    }

    // Extract singular values from W (should be approximately diagonal)
    for (size_type i = 0; i < m; ++i)
        for (size_type j = 0; j < n; ++j)
            S(i, j) = math::zero<value_type>();

    for (size_type i = 0; i < mn; ++i) {
        value_type sv = W(i, i);
        if (sv < math::zero<value_type>()) {
            // Flip sign: negate corresponding column of U
            S(i, i) = -sv;
            for (size_type k = 0; k < m; ++k)
                U(k, i) = -U(k, i);
        } else {
            S(i, i) = sv;
        }
    }
}

/// Convenience overload returning a tuple of (U, S, V).
template <Matrix M>
auto svd(const M& A, typename M::value_type tol = 1e-10) {
    using value_type = typename M::value_type;
    mat::dense2D<value_type> U, S, V;
    svd(A, U, S, V, tol);
    return std::make_tuple(std::move(U), std::move(S), std::move(V));
}

} // namespace mtl
