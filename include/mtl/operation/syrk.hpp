#pragma once
// MTL5 -- syrk: C <- alpha*A*A^T + beta*C  (BLAS level-3 symmetric rank-k)
// C is m x m and treated as symmetric; the full symmetric result is produced
// (both triangles set). A is m x k. BLAS dispatch for column-major dense
// float/double when available (writes the lower triangle, then mirrored).
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <limits>
#include <type_traits>

#include <mtl/concepts/scalar.hpp>
#include <mtl/concepts/matrix.hpp>
#include <mtl/math/identity.hpp>
#include <mtl/interface/dispatch_traits.hpp>
#ifdef MTL5_HAS_BLAS
#include <mtl/interface/blas.hpp>
#endif

namespace mtl {

/// Symmetric rank-k update: C = alpha*A*A^T + beta*C, C symmetric (full result).
template <Scalar S, Matrix MA, Matrix MC>
void syrk(const S& alpha, const MA& A, const S& beta, MC& C) {
    using value_type = typename MC::value_type;
    using size_type  = typename MA::size_type;
    const size_type m = A.num_rows();
    const size_type k = A.num_cols();
    assert(C.num_rows() == m && C.num_cols() == m);

#ifdef MTL5_HAS_BLAS
    if constexpr (interface::BlasDenseMatrix<MA> && interface::BlasDenseMatrix<MC> &&
                  !interface::is_row_major_v<MA> && !interface::is_row_major_v<MC> &&
                  std::is_same_v<value_type, typename MA::value_type>) {
        using T = value_type;
        const auto imax = static_cast<size_type>(std::numeric_limits<int>::max());
        if (m <= imax && k <= imax) {
            const int lda = std::max(1, static_cast<int>(m));
            interface::blas::syrk('L', 'N', static_cast<int>(m), static_cast<int>(k),
                                  static_cast<T>(alpha), A.data(), lda,
                                  static_cast<T>(beta), C.data(), lda);
            // BLAS wrote the lower triangle; mirror to the upper for a full result.
            for (size_type i = 0; i < m; ++i)
                for (size_type j = 0; j < i; ++j)
                    C(j, i) = C(i, j);
            return;
        }
    }
#endif
    // Compute the lower triangle, then mirror.
    for (size_type i = 0; i < m; ++i)
        for (size_type j = 0; j <= i; ++j) {
            auto acc = math::zero<value_type>();
            for (size_type l = 0; l < k; ++l)
                acc += A(i, l) * A(j, l);
            C(i, j) = alpha * acc + beta * C(i, j);
        }
    for (size_type i = 0; i < m; ++i)
        for (size_type j = 0; j < i; ++j)
            C(j, i) = C(i, j);
}

} // namespace mtl
