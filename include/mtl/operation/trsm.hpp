#pragma once
// MTL5 -- trsm: solve A * X = alpha * B for triangular A  (BLAS level-3, left side)
// A is m x m upper (upper=true) or lower triangular; B is m x n and is
// overwritten with the solution X. unit_diag treats A's diagonal as all ones.
// Left side, no transpose in this variant. BLAS dispatch for column-major dense
// float/double when available.
//
// Mixed precision: pass an explicit `Accumulator` to form each row's
// already-solved-columns reduction in a precision distinct from the operand
// type (#261, Part C). The subtraction from alpha*B(i,c) and the division by
// the diagonal happen after that reduction is rounded out, outside the
// accumulator -- same split as trsv (#516): the reduction sum_{j solved}
// A(i,j)*X(j,c) is a genuine data dependency, not just a rounding choice.
// Default `Accumulator = void` keeps the BLAS / generic dispatch byte for
// byte. Gated on `interface::accumulator_allows_blas_v`, same as trsv/trmm: a
// custom accumulator cannot be honored by external BLAS.
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <limits>
#include <type_traits>

#include <mtl/concepts/scalar.hpp>
#include <mtl/concepts/matrix.hpp>
#include <mtl/math/accumulator_traits.hpp>
#include <mtl/interface/dispatch_traits.hpp>
#ifdef MTL5_HAS_BLAS
#include <mtl/interface/blas.hpp>
#endif

namespace mtl {

/// Triangular solve with multiple RHS: solve A * X = alpha * B, B := X.
template <typename Accumulator = void, Scalar S, Matrix MA, Matrix MB>
void trsm(const S& alpha, const MA& A, MB& B, bool upper, bool unit_diag = false) {
    using value_type = typename MB::value_type;
    using size_type  = typename MA::size_type;
    const size_type m = A.num_rows();
    const size_type n = B.num_cols();
    assert(A.num_cols() == m && B.num_rows() == m);

    if constexpr (!interface::accumulator_allows_blas_v<Accumulator>) {
        using Value = std::common_type_t<value_type, typename MA::value_type>;
        using AT = math::accumulator_traits<Accumulator, Value>;
        for (size_type c = 0; c < n; ++c) {
            if (upper) {
                // Back substitution.
                for (size_type ii = m; ii-- > 0; ) {
                    Accumulator acc{};
                    AT::clear(acc);
                    for (size_type j = ii + 1; j < m; ++j)
                        AT::add_product(acc, static_cast<Value>(A(ii, j)), static_cast<Value>(B(j, c)));
                    const value_type sum = AT::template value<value_type>(acc);
                    auto s = alpha * B(ii, c) - sum;
                    B(ii, c) = unit_diag ? s : s / A(ii, ii);
                }
            } else {
                // Forward substitution.
                for (size_type i = 0; i < m; ++i) {
                    Accumulator acc{};
                    AT::clear(acc);
                    for (size_type j = 0; j < i; ++j)
                        AT::add_product(acc, static_cast<Value>(A(i, j)), static_cast<Value>(B(j, c)));
                    const value_type sum = AT::template value<value_type>(acc);
                    auto s = alpha * B(i, c) - sum;
                    B(i, c) = unit_diag ? s : s / A(i, i);
                }
            }
        }
        return;
    }

#ifdef MTL5_HAS_BLAS
    if constexpr (interface::BlasDenseMatrix<MA> && interface::BlasDenseMatrix<MB> &&
                  !interface::is_row_major_v<MA> && !interface::is_row_major_v<MB> &&
                  std::is_same_v<value_type, typename MA::value_type>) {
        using T = value_type;
        const auto imax = static_cast<size_type>(std::numeric_limits<int>::max());
        if (m <= imax && n <= imax) {
            // BLAS requires lda,ldb >= max(1,m) even for empty (m==0) matrices.
            const int ld = std::max(1, static_cast<int>(m));
            interface::blas::trsm('L', upper ? 'U' : 'L', 'N', unit_diag ? 'U' : 'N',
                                  static_cast<int>(m), static_cast<int>(n),
                                  static_cast<T>(alpha), A.data(), ld, B.data(), ld);
            return;
        }
    }
#endif
    for (size_type c = 0; c < n; ++c) {
        if (upper) {
            // Back substitution.
            for (size_type ii = m; ii-- > 0; ) {
                auto s = alpha * B(ii, c);
                for (size_type j = ii + 1; j < m; ++j)
                    s -= A(ii, j) * B(j, c);
                B(ii, c) = unit_diag ? s : s / A(ii, ii);
            }
        } else {
            // Forward substitution.
            for (size_type i = 0; i < m; ++i) {
                auto s = alpha * B(i, c);
                for (size_type j = 0; j < i; ++j)
                    s -= A(i, j) * B(j, c);
                B(i, c) = unit_diag ? s : s / A(i, i);
            }
        }
    }
}

} // namespace mtl
