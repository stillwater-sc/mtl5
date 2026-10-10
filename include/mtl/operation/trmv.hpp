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
#include <mtl/math/accumulator_traits.hpp>
#include <mtl/interface/dispatch_traits.hpp>
#ifdef MTL5_HAS_BLAS
#include <mtl/interface/blas.hpp>
#endif

namespace mtl {

/// Triangular matrix-vector product: x = A*x, A upper/lower triangular.
///
/// Mixed precision: pass an explicit `Accumulator` to form each row's
/// reduction in a precision distinct from the operand type (#261, Part C).
/// Default `Accumulator = void` keeps the BLAS / generic dispatch byte for
/// byte.
///
/// Gated on `interface::accumulator_allows_blas_v`, same as symv/gemv: a
/// custom accumulator cannot be honored by external BLAS.
///
/// The diagonal term (1*x(i) for unit_diag, A(i,i)*x(i) otherwise) is fed
/// through `add_product` as the first term of a `clear`-seeded reduction,
/// same shape as symv's row loop -- not special-cased as a seed -- so the
/// diagonal gets the same rounding treatment as every off-diagonal term.
///
/// The traversal order (forward for upper, reverse for lower) is an
/// in-place-overwrite hazard, not a precision concern: x(j) for the
/// off-diagonal terms must still hold its original value when read, which
/// this order guarantees regardless of Accumulator. Unchanged by this patch.
template <typename Accumulator = void, Matrix M, Vector VX>
void trmv(const M& A, VX& x, bool upper, bool unit_diag = false) {
    using value_type = typename VX::value_type;
    using size_type  = typename M::size_type;
    const size_type n = A.num_rows();
    assert(A.num_cols() == n && x.size() == n);

    if constexpr (!interface::accumulator_allows_blas_v<Accumulator>) {
        using Value = std::common_type_t<value_type, typename M::value_type>;
        using AT = math::accumulator_traits<Accumulator, Value>;
        if (upper) {
            for (size_type i = 0; i < n; ++i) {
                Accumulator acc{};
                AT::clear(acc);
                const Value diag = unit_diag ? Value(1) : static_cast<Value>(A(i, i));
                AT::add_product(acc, diag, static_cast<Value>(x(i)));
                for (size_type j = i + 1; j < n; ++j)
                    AT::add_product(acc, static_cast<Value>(A(i, j)), static_cast<Value>(x(j)));
                x(i) = AT::template value<value_type>(acc);
            }
        } else {
            for (size_type ii = n; ii-- > 0; ) {
                Accumulator acc{};
                AT::clear(acc);
                const Value diag = unit_diag ? Value(1) : static_cast<Value>(A(ii, ii));
                AT::add_product(acc, diag, static_cast<Value>(x(ii)));
                for (size_type j = 0; j < ii; ++j)
                    AT::add_product(acc, static_cast<Value>(A(ii, j)), static_cast<Value>(x(j)));
                x(ii) = AT::template value<value_type>(acc);
            }
        }
        return;
    }

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
