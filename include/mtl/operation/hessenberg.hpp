#pragma once
// MTL5 -- Hessenberg reduction via Householder reflections
// Reduces A to upper Hessenberg form H = Q^H * A * Q via a unitary similarity.
#include <algorithm>
#include <mtl/concepts/matrix.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/operation/householder.hpp>
#include <mtl/concepts/scalar.hpp>
#include <mtl/functor/scalar/conj.hpp>
#include <mtl/math/identity.hpp>

namespace mtl {

/// Reduce A to upper Hessenberg form in-place.
/// On output, A contains H (upper Hessenberg) and the Householder vectors
/// are stored in the strict lower triangle below the subdiagonal.
/// tau stores the Householder scalars.
template <Matrix M>
void hessenberg_factor(M& A, vec::dense_vector<typename M::value_type>& tau) {
    using value_type = typename M::value_type;
    using size_type  = typename M::size_type;
    const size_type n = A.num_rows();
    if (n < 3) { tau.change_dim(0); return; }
    // A similarity transform needs A -> H*A*H^-1. A Householder reflector H is
    // UNITARY, so H^-1 == H^H and A -> H*A*H^H is a genuine similarity for any
    // element type. The left factor stays H (I - beta*v*v^H); the right factor
    // must be H^H == I - conj(beta)*v*v^H, so apply_householder_right is passed
    // conj(beta) below -- the same adjoint bookkeeping #361 worked out for
    // qr_extract_Q / lq_extract_Q.
    //
    // For a REAL element type conj is the identity, so this is bit-identical to
    // the earlier H*A*H formulation: a real reflector is Hermitian (H^H == H),
    // which is why the unconjugated code was correct there and only there.
    const size_type k = n - 2;
    tau.change_dim(k);

    for (size_type j = 0; j < k; ++j) {
        // Extract column A(j+1:n-1, j)
        size_type len = n - j - 1;
        vec::dense_vector<value_type> col(len);
        for (size_type i = 0; i < len; ++i)
            col(i) = A(j + 1 + i, j);

        auto [v, beta] = householder(col);
        tau(j) = beta;

        // Apply from left: A(j+1:n-1, j:n-1) = H * A(j+1:n-1, j:n-1)
        apply_householder_left(A, v, beta, j + 1, j);

        // Apply from right: A(0:n-1, j+1:n-1) = A(0:n-1, j+1:n-1) * H^H.
        // apply_householder_right computes A*(I - s*v*v^H); passing conj(beta)
        // makes s == conj(beta), i.e. A*H^H, completing the similarity.
        apply_householder_right(A, v, functor::scalar::conj<value_type>::apply(beta), 0, j + 1);

        // Store Householder vector below subdiagonal
        for (size_type i = 1; i < len; ++i)
            A(j + 1 + i, j) = v(i);
    }
}

/// Extract explicit upper Hessenberg matrix from factored A.
template <Matrix M>
auto hessenberg_extract_H(const M& A) {
    using value_type = typename M::value_type;
    using size_type  = typename M::size_type;
    const size_type n = A.num_rows();

    mat::dense2D<value_type> H(n, n);
    for (size_type i = 0; i < n; ++i)
        for (size_type j = 0; j < n; ++j)
            H(i, j) = (i <= j + 1) ? A(i, j) : math::zero<value_type>();
    return H;
}

/// Reduce A to upper Hessenberg form and return H directly.
template <Matrix M>
auto hessenberg(const M& A) {
    using value_type = typename M::value_type;
    using size_type  = typename M::size_type;
    const size_type n = A.num_rows();

    mat::dense2D<value_type> H(n, n);
    for (size_type i = 0; i < n; ++i)
        for (size_type j = 0; j < n; ++j)
            H(i, j) = A(i, j);

    vec::dense_vector<value_type> tau;
    hessenberg_factor(H, tau);
    return hessenberg_extract_H(H);
}

/// Reduce a symmetric (or Hermitian) matrix to tridiagonal form in-place.
/// After this call, the diagonal and subdiagonal of A contain the tridiagonal
/// matrix. Hessenberg reduction of a symmetric matrix is tridiagonal; of a
/// Hermitian matrix it is Hermitian tridiagonal (real diagonal, conjugate
/// off-diagonals), so the subdiagonal entries may carry a phase.
template <Matrix M>
void tridiagonalize(M& A, vec::dense_vector<typename M::value_type>& tau) {
    hessenberg_factor(A, tau);
}

} // namespace mtl
