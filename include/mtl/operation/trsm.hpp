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
#ifdef MTL5_NATIVE_FAST_GEMM
#include <mtl/detail/gemm_blocked.hpp>
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
///
/// [b0, b1) restricts the solve to one diagonal block: rows b0..b1-1, using
/// only A's columns in the same range. The default is the whole matrix; the
/// blocked solve below calls it per block, after GEMM has removed the rest.
template <typename S, typename MA, typename MB>
void trsm_rows(const S& alpha, const MA& A, MB& B, bool upper, bool unit_diag,
               typename MA::size_type b0, typename MA::size_type b1) {
    using size_type  = typename MA::size_type;
    using value_type = typename MB::value_type;
    using Acc = std::remove_cvref_t<decltype(alpha * std::declval<value_type&>())>;
    const auto n = B.num_cols();
    if (b0 >= b1 || n == 0) return;
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
        for (size_type ii = b1; ii-- > b0; ) solve_row(ii, ii + 1, b1);  // back substitution
    else
        for (size_type i = b0; i < b1; ++i) solve_row(i, b0, i);         // forward substitution
}

template <typename S, typename MA, typename MB>
void trsm_rows(const S& alpha, const MA& A, MB& B, bool upper, bool unit_diag) {
    trsm_rows(alpha, A, B, upper, unit_diag, typename MA::size_type{0}, A.num_rows());
}

/// Whether trsm<void> on this build takes the blocked solve over the native
/// GEMM (#724): MTL5_NATIVE_FAST_GEMM, float or double, dense A, row-major B.
template <typename MA, typename MB>
inline constexpr bool trsm_uses_blocked_v = [] {
#ifdef MTL5_NATIVE_FAST_GEMM
    if constexpr (interface::SimdDenseMatrix<MA> && interface::SimdDenseMatrix<MB> &&
                  requires { typename MA::orientation; })
        return interface::is_blas_scalar_v<typename MB::value_type> &&
               std::is_same_v<typename MA::value_type, typename MB::value_type> &&
               row_major_contiguous_v<MB>;
    else
        return false;
#else
    return false;
#endif
}();

#ifdef MTL5_NATIVE_FAST_GEMM
/// Blocked triangular solve (#724), the LAPACK/BLIS structure: walk A's
/// diagonal in blocks of `triangular_block` rows; for each, GEMM subtracts the
/// already-solved rows' contribution, B(I) := alpha B(I) - A(I, done) X(done),
/// and the row loop solves the small diagonal block. All but O(m nb n) of the
/// m^2 n flops run in the blocked GEMM. The rounding differs from the row loop
/// (the GEMM sums in its own order), as native-fast GEMM's does from mult's.
template <typename S, typename MA, typename MB>
void trsm_blocked(const S& alpha, const MA& A, MB& B, bool upper, bool unit_diag) {
    using T = typename MB::value_type;
    const std::size_t m = A.num_rows();
    const std::size_t n = B.num_cols();
    const T a = static_cast<T>(alpha);
    constexpr std::size_t nb = triangular_block;
    const std::ptrdiff_t a_rs = interface::is_row_major_v<MA> ? static_cast<std::ptrdiff_t>(m) : 1;
    const std::ptrdiff_t a_cs = interface::is_row_major_v<MA> ? 1 : static_cast<std::ptrdiff_t>(m);
    const T* ap = A.data();
    T* bp = B.data();
    const auto at = [&](std::size_t i, std::size_t j) { return ap + static_cast<std::ptrdiff_t>(i) * a_rs +
                                                               static_cast<std::ptrdiff_t>(j) * a_cs; };
    const std::ptrdiff_t ldb = static_cast<std::ptrdiff_t>(n);
    const unsigned nt = gemm_default_threads();
    using sz = typename MA::size_type;
    if (!upper) {
        for (std::size_t i0 = 0; i0 < m; i0 += nb) {
            const std::size_t i1 = std::min(m, i0 + nb);
            if (i0 > 0)   // B(I) := alpha B(I) - A(I, 0:i0) X(0:i0)
                gemm_blocked<T>(i1 - i0, n, i0, T(-1), at(i0, 0), a_rs, a_cs, bp, ldb, 1,
                                a, bp + i0 * n, n, nt);
            trsm_rows(i0 > 0 ? T(1) : a, A, B, false, unit_diag, static_cast<sz>(i0), static_cast<sz>(i1));
        }
    } else {
        for (std::size_t i1 = m; i1 > 0; ) {
            const std::size_t i0 = i1 > nb ? i1 - nb : 0;
            if (i1 < m)   // B(I) := alpha B(I) - A(I, i1:m) X(i1:m)
                gemm_blocked<T>(i1 - i0, n, m - i1, T(-1), at(i0, i1), a_rs, a_cs, bp + i1 * n, ldb, 1,
                                a, bp + i0 * n, n, nt);
            trsm_rows(i1 < m ? T(1) : a, A, B, true, unit_diag, static_cast<sz>(i0), static_cast<sz>(i1));
            i1 = i0;
        }
    }
}
#endif

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
#ifdef MTL5_NATIVE_FAST_GEMM
    if constexpr (detail::trsm_uses_blocked_v<MA, MB>) {
        if (m > detail::triangular_block && n > 0) {
            detail::trsm_blocked(alpha, A, B, upper, unit_diag);
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
