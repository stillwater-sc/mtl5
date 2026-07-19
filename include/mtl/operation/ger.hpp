#pragma once
// MTL5 -- ger: A <- alpha*x*y^T + A  (BLAS level-2 general rank-1 update)
// Generic path for any Matrix/Vector; BLAS dispatch for column-major dense
// float/double when available.
#include <cassert>
#include <cstddef>
#include <limits>
#include <type_traits>

#include <mtl/concepts/scalar.hpp>
#include <mtl/concepts/vector.hpp>
#include <mtl/concepts/matrix.hpp>
#include <mtl/interface/dispatch_traits.hpp>
#ifdef MTL5_HAS_BLAS
#include <mtl/interface/blas.hpp>
#endif

namespace mtl {

/// Rank-1 update: A(i,j) += alpha * x(i) * y(j).
template <Scalar S, Vector VX, Vector VY, Matrix M>
void ger(const S& alpha, const VX& x, const VY& y, M& A) {
    using size_type = typename M::size_type;
    const size_type m = A.num_rows();
    const size_type n = A.num_cols();
    assert(x.size() == m && y.size() == n);

#ifdef MTL5_HAS_BLAS
    if constexpr (interface::BlasDenseMatrix<M> && interface::BlasDenseVector<VX> &&
                  interface::BlasDenseVector<VY> && !interface::is_row_major_v<M> &&
                  std::is_same_v<typename M::value_type, typename VX::value_type> &&
                  std::is_same_v<typename M::value_type, typename VY::value_type>) {
        using T = typename M::value_type;
        const auto imax = static_cast<size_type>(std::numeric_limits<int>::max());
        if (m <= imax && n <= imax) {
            // Column-major storage: leading dimension is the row count.
            interface::blas::ger(static_cast<int>(m), static_cast<int>(n),
                                 static_cast<T>(alpha), x.data(), 1, y.data(), 1,
                                 A.data(), static_cast<int>(m));
            return;
        }
    }
#endif
    for (size_type i = 0; i < m; ++i)
        for (size_type j = 0; j < n; ++j)
            A(i, j) += alpha * x(i) * y(j);
}

} // namespace mtl
