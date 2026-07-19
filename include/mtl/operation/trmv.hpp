#pragma once
// MTL5 -- trmv: x <- A*x  for triangular A  (BLAS level-2)
// A is upper (upper=true) or lower triangular; the opposite triangle is ignored.
// unit_diag treats the diagonal as all ones. No transpose in this variant.
// BLAS dispatch for column-major dense float/double when available.
#include <cassert>
#include <cstddef>
#include <limits>
#include <type_traits>

#include <mtl/concepts/vector.hpp>
#include <mtl/concepts/matrix.hpp>
#include <mtl/math/identity.hpp>
#include <mtl/interface/dispatch_traits.hpp>
#ifdef MTL5_HAS_BLAS
#include <mtl/interface/blas.hpp>
#endif

namespace mtl {

/// Triangular matrix-vector product: x = A*x, A upper/lower triangular.
template <Matrix M, Vector VX>
void trmv(const M& A, VX& x, bool upper, bool unit_diag = false) {
    using value_type = typename VX::value_type;
    using size_type  = typename M::size_type;
    const size_type n = A.num_rows();
    assert(A.num_cols() == n && x.size() == n);

#ifdef MTL5_HAS_BLAS
    if constexpr (interface::BlasDenseMatrix<M> && interface::BlasDenseVector<VX> &&
                  !interface::is_row_major_v<M> &&
                  std::is_same_v<value_type, typename M::value_type>) {
        using T = value_type;
        if (n <= static_cast<size_type>(std::numeric_limits<int>::max())) {
            interface::blas::trmv(upper ? 'U' : 'L', 'N', unit_diag ? 'U' : 'N',
                                  static_cast<int>(n), A.data(), static_cast<int>(n),
                                  x.data(), 1);
            return;
        }
    }
#endif
    if (upper) {
        // x_i depends on x_j for j >= i; forward order is safe.
        for (size_type i = 0; i < n; ++i) {
            auto acc = unit_diag ? x(i) : A(i, i) * x(i);
            for (size_type j = i + 1; j < n; ++j)
                acc += A(i, j) * x(j);
            x(i) = acc;
        }
    } else {
        // x_i depends on x_j for j <= i; reverse order is safe.
        for (size_type ii = n; ii-- > 0; ) {
            auto acc = unit_diag ? x(ii) : A(ii, ii) * x(ii);
            for (size_type j = 0; j < ii; ++j)
                acc += A(ii, j) * x(j);
            x(ii) = acc;
        }
    }
}

} // namespace mtl
