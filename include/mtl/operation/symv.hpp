#pragma once
// MTL5 -- symv: y <- alpha*A*x + beta*y  for symmetric A  (BLAS level-2)
// The generic path reads the full matrix (correct for any symmetric A); the
// BLAS path reads a single triangle. BLAS dispatch for column-major dense
// float/double when available.
#include <cassert>
#include <cstddef>
#include <limits>
#include <type_traits>

#include <mtl/concepts/scalar.hpp>
#include <mtl/concepts/vector.hpp>
#include <mtl/concepts/matrix.hpp>
#include <mtl/math/identity.hpp>
#include <mtl/interface/dispatch_traits.hpp>
#ifdef MTL5_HAS_BLAS
#include <mtl/interface/blas.hpp>
#endif

namespace mtl {

/// Symmetric matrix-vector product: y = alpha*A*x + beta*y, A assumed symmetric.
template <Scalar S, Matrix M, Vector VX, Vector VY>
void symv(const S& alpha, const M& A, const VX& x, const S& beta, VY& y) {
    using value_type = typename VY::value_type;
    using size_type  = typename M::size_type;
    const size_type n = A.num_rows();
    assert(A.num_cols() == n && x.size() == n && y.size() == n);

#ifdef MTL5_HAS_BLAS
    if constexpr (interface::BlasDenseMatrix<M> && interface::BlasDenseVector<VX> &&
                  interface::BlasDenseVector<VY> && !interface::is_row_major_v<M> &&
                  std::is_same_v<value_type, typename M::value_type> &&
                  std::is_same_v<value_type, typename VX::value_type>) {
        using T = value_type;
        if (n <= static_cast<size_type>(std::numeric_limits<int>::max())) {
            // Symmetric: 'L' reads the lower triangle (== upper for symmetric A).
            interface::blas::symv('L', static_cast<int>(n), static_cast<T>(alpha),
                                  A.data(), static_cast<int>(n), x.data(), 1,
                                  static_cast<T>(beta), y.data(), 1);
            return;
        }
    }
#endif
    for (size_type i = 0; i < n; ++i) {
        auto acc = math::zero<value_type>();
        for (size_type j = 0; j < n; ++j)
            acc += A(i, j) * x(j);
        y(i) = alpha * acc + beta * y(i);
    }
}

} // namespace mtl
