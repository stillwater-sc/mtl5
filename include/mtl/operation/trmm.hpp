#pragma once
// MTL5 -- trmm: B <- alpha * A * B  for triangular A  (BLAS level-3, left side)
// A is m x m upper (upper=true) or lower triangular; B is m x n. unit_diag treats
// A's diagonal as all ones. Left side, no transpose in this variant. BLAS
// dispatch for dense float/double when available, in either orientation (#724):
// see interface::blas_triangular_describe.
//
// alpha == 0 follows the BLAS convention (#692): A and B are not read and B is
// set to zero, so a NaN or uninitialized B does not propagate.
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <vector>

#include <mtl/concepts/scalar.hpp>
#include <mtl/concepts/matrix.hpp>
#include <mtl/math/identity.hpp>
#include <mtl/math/accumulator_traits.hpp>
#include <mtl/interface/dispatch_traits.hpp>
#ifdef MTL5_HAS_BLAS
#include <mtl/interface/blas.hpp>
#endif

namespace mtl {

namespace detail {

/// Whether trmm<Accumulator>(alpha, A, B, ...) calls ?trmm (for sizes that fit
/// an int and a nonzero alpha). Every orientation of A and B qualifies.
template <typename Accumulator, typename MA, typename MB>
inline constexpr bool trmm_dispatches_blas_v = [] {
#ifdef MTL5_HAS_BLAS
    if constexpr (interface::accumulator_allows_blas_v<Accumulator> &&
                  interface::BlasDenseMatrix<MA> && interface::BlasDenseMatrix<MB>)
        return std::is_same_v<typename MB::value_type, typename MA::value_type>;
    else
        return false;
#else
    return false;
#endif
}();

/// Whether the generic trmm<void> forms a row-major B row by row (#724).
template <typename MB>
inline constexpr bool trmm_uses_row_loop_v = row_major_contiguous_v<MB>;

/// The generic product on a row-major B, a whole row of B at a time (#724).
///
/// The column-by-column loop reads B(j, c) down a column. This form builds row
/// i of the result as (A(i, i) or 1) * B.row(i) plus A(i, j) * B.row(j) for each
/// j in the triangle, in ascending j: contiguous inner loops. Rows are formed
/// in the order that leaves every B.row(j) still unwritten when it is read
/// (ascending i for upper, descending for lower), as the column loop does, and
/// each entry is the same expression in the same type, so the result is
/// bit-identical (contraction aside: see trsm_rows).
template <typename S, typename MA, typename MB>
void trmm_rows(const S& alpha, const MA& A, MB& B, bool upper, bool unit_diag) {
    using size_type  = typename MA::size_type;
    using value_type = typename MB::value_type;
    using Acc = std::remove_cvref_t<decltype(unit_diag ? std::declval<value_type&>()
                                                       : A(0, 0) * std::declval<value_type&>())>;
    const size_type m = A.num_rows();
    const auto n = B.num_cols();
    if (m == 0 || n == 0) return;
    std::vector<Acc> row(static_cast<std::size_t>(n));
    const auto form_row = [&](size_type i, size_type jb, size_type je) {
        value_type* bi = &B(i, 0);
        if (unit_diag) {
            for (decltype(+n) c = 0; c < n; ++c) row[c] = bi[c];
        } else {
            const auto& d = A(i, i);
            for (decltype(+n) c = 0; c < n; ++c) row[c] = d * bi[c];
        }
        for (size_type j = jb; j < je; ++j) {
            const auto& a = A(i, j);
            const value_type* bj = &B(j, 0);
            for (decltype(+n) c = 0; c < n; ++c) row[c] += a * bj[c];
        }
        for (decltype(+n) c = 0; c < n; ++c) bi[c] = alpha * row[c];
    };
    if (upper)
        for (size_type i = 0; i < m; ++i) form_row(i, i + 1, m);
    else
        for (size_type ii = m; ii-- > 0; ) form_row(ii, 0, ii);
}

} // namespace detail

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
    if (alpha == math::zero<S>()) {
        for (size_type r = 0; r < m; ++r)
            for (size_type c = 0; c < n; ++c)
                B(r, c) = math::zero<value_type>();
        return;
    }

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
    if constexpr (detail::trmm_dispatches_blas_v<Accumulator, MA, MB>) {
        using T = value_type;
        const auto imax = static_cast<size_type>(std::numeric_limits<int>::max());
        if (m <= imax && n <= imax) {
            const auto d = interface::blas_triangular_describe<MA, MB>(upper, m, n);
            interface::blas::trmm(d.side, d.uplo, d.trans, unit_diag ? 'U' : 'N', d.m, d.n,
                                  static_cast<T>(alpha), A.data(), std::max(1, static_cast<int>(m)),
                                  B.data(), d.ldb);
            return;
        }
    }
#endif
    if constexpr (detail::trmm_uses_row_loop_v<MB>) {
        detail::trmm_rows(alpha, A, B, upper, unit_diag);
        return;
    }
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
