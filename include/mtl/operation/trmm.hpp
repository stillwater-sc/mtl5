#pragma once
// MTL5 -- trmm: B <- alpha * A * B  for triangular A  (BLAS level-3, left side)
// A is m x m upper (upper=true) or lower triangular; B is m x n. unit_diag treats
// A's diagonal as all ones. Left side, no transpose in this variant. BLAS
// dispatch for column-major dense float/double when available.
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <limits>
#include <type_traits>

#include <mtl/concepts/scalar.hpp>
#include <mtl/concepts/matrix.hpp>
#include <mtl/math/identity.hpp>
#include <mtl/math/accumulator_traits.hpp>
#include <mtl/interface/dispatch_traits.hpp>
#ifdef MTL5_HAS_BLAS
#include <mtl/interface/blas.hpp>
#endif

namespace mtl {

/// Triangular matrix-matrix product: B = alpha * A * B, A upper/lower triangular.
///
/// Mixed precision: pass an explicit `Accumulator` to form each (row, col)
/// reduction in a precision distinct from the operand type (#261, Part C).
/// Default `Accumulator = void` keeps the BLAS / generic dispatch byte for
/// byte.
///
/// Gated on `interface::accumulator_allows_blas_v`, same as trmv/symv: a
/// custom accumulator cannot be honored by external BLAS.
///
/// The diagonal term (1*B(i,c) for unit_diag, A(i,i)*B(i,c) otherwise) is fed
/// through `add_product` as the first term of a `clear`-seeded reduction per
/// (row, col) pair -- same shape as trmv's row loop extended over B's columns
/// -- not special-cased as a seed, so the diagonal gets the same rounding
/// treatment as every off-diagonal term.
///
/// The traversal order (forward for upper, reverse for lower) is an
/// in-place-overwrite hazard, not a precision concern: B(j,c) for the
/// off-diagonal terms must still hold its original value when read, which
/// this order guarantees regardless of Accumulator. Unchanged by this patch.
template <typename Accumulator = void, Scalar S, Matrix MA, Matrix MB>
void trmm(const S& alpha, const MA& A, MB& B, bool upper, bool unit_diag = false) {
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
                for (size_type i = 0; i < m; ++i) {
                    Accumulator acc{};
                    AT::clear(acc);
                    const Value diag = unit_diag ? Value(1) : static_cast<Value>(A(i, i));
                    AT::add_product(acc, diag, static_cast<Value>(B(i, c)));
                    for (size_type j = i + 1; j < m; ++j)
                        AT::add_product(acc, static_cast<Value>(A(i, j)), static_cast<Value>(B(j, c)));
                    B(i, c) = alpha * AT::template value<value_type>(acc);
                }
            } else {
                for (size_type ii = m; ii-- > 0; ) {
                    Accumulator acc{};
                    AT::clear(acc);
                    const Value diag = unit_diag ? Value(1) : static_cast<Value>(A(ii, ii));
                    AT::add_product(acc, diag, static_cast<Value>(B(ii, c)));
                    for (size_type j = 0; j < ii; ++j)
                        AT::add_product(acc, static_cast<Value>(A(ii, j)), static_cast<Value>(B(j, c)));
                    B(ii, c) = alpha * AT::template value<value_type>(acc);
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
            const int ld = std::max(1, static_cast<int>(m));
            interface::blas::trmm('L', upper ? 'U' : 'L', 'N', unit_diag ? 'U' : 'N',
                                  static_cast<int>(m), static_cast<int>(n),
                                  static_cast<T>(alpha), A.data(), ld, B.data(), ld);
            return;
        }
    }
#endif
    for (size_type c = 0; c < n; ++c) {
        if (upper) {
            for (size_type i = 0; i < m; ++i) {
                auto acc = unit_diag ? B(i, c) : A(i, i) * B(i, c);
                for (size_type j = i + 1; j < m; ++j)
                    acc += A(i, j) * B(j, c);
                B(i, c) = alpha * acc;
            }
        } else {
            for (size_type ii = m; ii-- > 0; ) {
                auto acc = unit_diag ? B(ii, c) : A(ii, ii) * B(ii, c);
                for (size_type j = 0; j < ii; ++j)
                    acc += A(ii, j) * B(j, c);
                B(ii, c) = alpha * acc;
            }
        }
    }
}

} // namespace mtl
