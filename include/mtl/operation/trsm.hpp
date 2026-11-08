#pragma once
// MTL5 -- trsm: solve A * X = alpha * B for triangular A  (BLAS level-3, left side)
// A is m x m upper (upper=true) or lower triangular; B is m x n and is
// overwritten with the solution X. unit_diag treats A's diagonal as all ones.
// Left side, no transpose in this variant. BLAS dispatch for dense float/double
// when available, in either orientation (#724): a row-major buffer is the
// column-major transpose, so a row-major B is solved as X^T op(A)^T = alpha B^T
// (side 'R'), and a row-major A is passed as its transpose with uplo flipped.
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
#include <vector>

#include <mtl/concepts/scalar.hpp>
#include <mtl/concepts/matrix.hpp>
#include <mtl/math/accumulator_traits.hpp>
#include <mtl/interface/dispatch_traits.hpp>
#ifdef MTL5_HAS_BLAS
#include <mtl/interface/blas.hpp>
#endif

namespace mtl {

namespace detail {

/// Whether trsm<Accumulator>(alpha, A, B, ...) calls ?trsm (for sizes that fit
/// an int). Every orientation of A and B qualifies.
template <typename Accumulator, typename MA, typename MB>
inline constexpr bool trsm_dispatches_blas_v = [] {
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

/// Whether the generic trsm<void> solves a row-major B row by row (#724).
template <typename MB>
inline constexpr bool trsm_uses_row_loop_v = row_major_contiguous_v<MB>;

/// The generic solve on a row-major B, a whole row of B at a time (#724).
///
/// The column-by-column loop reads B(j, c) down a column, a stride of n per
/// access. This form subtracts A(i, j) * B.row(j) from row i for each solved
/// j: a contiguous, independent inner loop the compiler vectorizes. Every
/// X(i, c) is still alpha*B(i, c), minus A(i, j)*X(j, c) in ascending j, then
/// divided by A(i, i), with the same expressions in the same types (`row`
/// holds what the column loop's `s` held), so the result is bit-identical --
/// unless the compiler contracts across statements (GCC's -ffp-contract=fast
/// with an FMA), which may fuse a different product in each form: same order,
/// an ulp apart. That caveat applies to every row loop here, as to mult's.
template <typename S, typename MA, typename MB>
void trsm_rows(const S& alpha, const MA& A, MB& B, bool upper, bool unit_diag) {
    using size_type  = typename MA::size_type;
    using value_type = typename MB::value_type;
    using Acc = std::remove_cvref_t<decltype(alpha * std::declval<value_type&>())>;
    const size_type m = A.num_rows();
    const auto n = B.num_cols();
    if (m == 0 || n == 0) return;
    std::vector<Acc> row(static_cast<std::size_t>(n));
    const auto solve_row = [&](size_type i, size_type jb, size_type je) {
        value_type* bi = &B(i, 0);
        for (decltype(+n) c = 0; c < n; ++c) row[c] = alpha * bi[c];
        for (size_type j = jb; j < je; ++j) {
            const auto& a = A(i, j);
            const value_type* bj = &B(j, 0);
            for (decltype(+n) c = 0; c < n; ++c) row[c] -= a * bj[c];
        }
        if (unit_diag) {
            for (decltype(+n) c = 0; c < n; ++c) bi[c] = row[c];
        } else {
            const auto& d = A(i, i);
            for (decltype(+n) c = 0; c < n; ++c) bi[c] = row[c] / d;
        }
    };
    if (upper)
        for (size_type ii = m; ii-- > 0; ) solve_row(ii, ii + 1, m);   // back substitution
    else
        for (size_type i = 0; i < m; ++i) solve_row(i, 0, i);           // forward substitution
}

} // namespace detail

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
    if constexpr (detail::trsm_dispatches_blas_v<Accumulator, MA, MB>) {
        using T = value_type;
        const auto imax = static_cast<size_type>(std::numeric_limits<int>::max());
        if (m <= imax && n <= imax) {
            // BLAS requires lda,ldb >= max(1, rows) even for empty matrices.
            const auto d = interface::blas_triangular_describe<MA, MB>(upper, m, n);
            interface::blas::trsm(d.side, d.uplo, d.trans, unit_diag ? 'U' : 'N', d.m, d.n,
                                  static_cast<T>(alpha), A.data(), std::max(1, static_cast<int>(m)),
                                  B.data(), d.ldb);
            return;
        }
    }
#endif
    if constexpr (detail::trsm_uses_row_loop_v<MB>) {
        detail::trsm_rows(alpha, A, B, upper, unit_diag);
        return;
    }
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
