#pragma once
// MTL5 -- symv: y <- alpha*A*x + beta*y  for symmetric A  (BLAS level-2)
// Only A's lower triangle is read, on every path, as BLAS symv with uplo 'L'
// reads it (#692): A(i,j) for j > i is taken as A(j,i), so the strict upper
// triangle may hold anything. Before #692 the generic path read the full
// matrix and so disagreed with the BLAS path for a non-symmetric A. BLAS
// dispatch for dense float/double when available, in either orientation of A
// (#724): a row-major A's lower triangle is its buffer's upper one.
//
// alpha/beta follow the BLAS convention, as gemmt does (#692): beta == 0 does
// not read y, so a NaN or uninitialized y does not propagate, and alpha == 0
// does not read A or x, leaving y = beta*y.
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <vector>

#include <mtl/concepts/scalar.hpp>
#include <mtl/concepts/vector.hpp>
#include <mtl/concepts/matrix.hpp>
#include <mtl/math/identity.hpp>
#include <mtl/math/accumulator_traits.hpp>
#include <mtl/detail/same_object.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/interface/dispatch_traits.hpp>
#include <mtl/detail/column_sweep.hpp>
#ifdef MTL5_HAS_BLAS
#include <mtl/interface/blas.hpp>
#endif

namespace mtl {

namespace detail {

/// Whether symv<Accumulator>(alpha, A, x, beta, y) calls ?symv (for sizes that
/// fit an int). Either orientation of A: a symmetric matrix's row-major buffer
/// holds its lower triangle where the column-major view has the upper one (#724).
template <typename Accumulator, typename M, typename VX, typename VY>
inline constexpr bool symv_dispatches_blas_v = [] {
#ifdef MTL5_HAS_BLAS
    if constexpr (interface::accumulator_allows_blas_v<Accumulator> &&
                  interface::BlasDenseMatrix<M> && interface::BlasDenseVector<VX> && interface::BlasDenseVector<VY>)
        return std::is_same_v<typename VY::value_type, typename M::value_type> &&
               std::is_same_v<typename VY::value_type, typename VX::value_type>;
    else
        return false;
#else
    return false;
#endif
}();

/// Whether the generic symv<void> makes one row-wise pass over a row-major A's
/// lower triangle (#724).
template <typename M>
inline constexpr bool symv_uses_row_loop_v = row_major_contiguous_v<M>;

/// Rows per strip of the row-major symv (#236), and columns per block of a
/// strip's pass over the columns left of it: 8 rows of 256 doubles are 16 KB,
/// in L1 for the second of the two loops below. Below symv_strip_min_n the
/// row loop is as fast.
inline constexpr std::size_t symv_strip_rows = 8;
inline constexpr std::size_t symv_strip_block = 256;
inline constexpr std::size_t symv_strip_min_n = 128;

/// acc[j] += p[0][j] x[0] + p[1][j] x[1] + ... in row order, for j < len: R
/// rows' contributions to the entries left of the strip. Each acc[j] gets its
/// terms one at a time, in the row loop's order; the j's are independent, so
/// the loop vectorizes over j.
template <std::size_t R, typename Acc, typename TA, typename TX>
inline void symv_strip_scatter(std::size_t len, const TA* const* p, const TX* xr, Acc* acc) {
    for (std::size_t j = 0; j < len; ++j) {
        Acc a = acc[j];
        for (std::size_t r = 0; r < R; ++r)
            a += p[r][j] * xr[r];
        acc[j] = a;
    }
}

/// s[r] += p[r][j] x[j] in ascending j, for j < len: each row's own sum, R
/// independent chains sharing each x[j] (see MTL5_NO_LOOP_VECTORIZE).
template <std::size_t R, typename Acc, typename TA, typename X>
MTL5_NO_LOOP_VECTORIZE inline void symv_strip_gather(std::size_t len, const TA* const* p, Acc* s,
                                                     const X& x, std::size_t j0) {
    for (std::size_t j = 0; j < len; ++j) {
        const auto& xj = x(j0 + j);
        for (std::size_t r = 0; r < R; ++r)
            s[r] += p[r][j] * xj;
    }
}

} // namespace detail

/// Symmetric matrix-vector product: y = alpha*A*x + beta*y, A assumed symmetric.
///
/// Mixed precision: pass an explicit `Accumulator` to form each row's A*x
/// reduction in a precision distinct from the operand type (#261, Part C); the
/// result is rounded out to y's element type before the alpha/beta combine.
/// Default `Accumulator = void` keeps the BLAS / generic dispatch byte for byte.
///
/// Gated on `interface::accumulator_allows_blas_v`, exactly as `gemv` (via
/// `mult`) and `dot` are: a custom accumulator cannot be honored by external
/// BLAS, whose symv accumulates in a hardware-fixed precision.
///
/// The accumulator sums only the row's A(i,j)*x(j) products, seeded with
/// `clear` -- the alpha/beta combine happens once, outside the reduction, same
/// as gemv's generic row loop and unlike axpy's `assign`-seeded one-term sum.
template <typename Accumulator = void, Scalar S, Matrix M, Vector VX, Vector VY>
void symv(const S& alpha, const M& A, const VX& x, const S& beta, VY& y) {
    using value_type = typename VY::value_type;
    using size_type  = typename M::size_type;
    const size_type n = A.num_rows();
    assert(A.num_cols() == n && x.size() == n && y.size() == n);
    const bool alpha_zero = (alpha == math::zero<S>());
    const bool beta_zero  = (beta == math::zero<S>());
    // Every path writes y(i) while later rows still read x (and A), so a y
    // that shares storage with x or A is computed into a copy of y (beta*y
    // reads it; with beta == 0 y is not read, as on every other path) and
    // copied back, in every build (#855). It was a debug-only assert, on one
    // path, that saw only the same object.
    if (detail::storage_overlaps(y, x) || detail::storage_overlaps(y, A)) {
        vec::dense_vector<std::remove_cvref_t<value_type>> t(y.size());
        if (!beta_zero)
            for (size_type i = 0; i < n; ++i) t(i) = y(i);
        symv<Accumulator>(alpha, A, x, beta, t);
        for (size_type i = 0; i < n; ++i) y(i) = t(i);
        return;
    }
    // The lower triangle, read as a symmetric matrix.
    auto a = [&A](size_type i, size_type j) -> decltype(auto) { return j <= i ? A(i, j) : A(j, i); };

    if constexpr (!interface::accumulator_allows_blas_v<Accumulator>) {
        using Value = std::common_type_t<S, typename M::value_type, typename VX::value_type>;
        using AT = math::accumulator_traits<Accumulator, Value>;
        for (size_type i = 0; i < n; ++i) {
            if (alpha_zero) {
                y(i) = beta_zero ? math::zero<value_type>() : value_type(beta * y(i));
                continue;
            }
            Accumulator acc{};
            AT::clear(acc);
            for (size_type j = 0; j < n; ++j)
                AT::add_product(acc, static_cast<Value>(a(i, j)), static_cast<Value>(x(j)));
            const value_type reduced = AT::template value<value_type>(acc);
            y(i) = beta_zero ? value_type(alpha * reduced) : value_type(alpha * reduced + beta * y(i));
        }
        return;
    }

#ifdef MTL5_HAS_BLAS
    if constexpr (detail::symv_dispatches_blas_v<Accumulator, M, VX, VY>) {
        using T = value_type;
        if (n == 0) return;   // y is empty; skip BLAS (#693)
        if (n <= static_cast<size_type>(std::numeric_limits<int>::max())) {
            // The lower triangle, as the generic paths read it: 'L' in a
            // column-major buffer, 'U' in a row-major one. BLAS requires
            // lda >= 1 even for an empty A (#693).
            interface::blas::symv(interface::blas_lower_uplo<M>(), static_cast<int>(n), static_cast<T>(alpha),
                                  A.data(), std::max(1, static_cast<int>(n)), x.data(), 1,
                                  static_cast<T>(beta), y.data(), 1);
            return;
        }
    }
#endif
    if constexpr (detail::symv_uses_row_loop_v<M>) {
        // Row-major A: one pass over the stored lower triangle, row by row
        // (#724). The row loop below reads A(j, i) down a column for every
        // j > i. Here each stored A(i, j), j < i, is read once, from a
        // contiguous row, and feeds both y(i) (as a(i,j) x(j)) and y(j) (as
        // a(j,i) x(i)). Every y(i) still receives its terms in ascending j: j <= i
        // from row i itself, then j > i from rows j = i+1, i+2, ... in turn. Each
        // term is the same product and acc[] holds what the row loop's `acc`
        // held, so the result is bit-identical
        // (contraction aside: see detail::trsm_rows).
        if (n == 0) return;
        if (alpha_zero) {
            for (size_type i = 0; i < n; ++i)
                y(i) = beta_zero ? math::zero<value_type>() : value_type(beta * y(i));
            return;
        }
        std::vector<value_type> acc(static_cast<std::size_t>(n), math::zero<value_type>());
        size_type i = 0;
        if (static_cast<std::size_t>(n) >= detail::symv_strip_min_n) {
            // Strips of R rows (#236). The row loop's s is a serial chain, one
            // term per FMA latency; R rows at once are R independent chains, and
            // the j < i0 updates of acc[] read each acc[j] once for R terms.
            // Every term, and the order in which each acc[] and s receives
            // them, is the row loop's: row i0 + r's own sum runs over j < i0,
            // then j in [i0, i0 + r]; acc[j], j < i0, receives rows i0, i0 + 1,
            // ... in turn; an in-strip acc[i0 + q] holds its row's sum, then the
            // later rows of the strip, then those of later strips. So the
            // result is bit-identical (contraction aside).
            using TA = typename M::value_type;
            using TX = std::remove_cvref_t<decltype(x(size_type{0}))>;   // x(i)'s type, as the row loop's xi
            constexpr std::size_t R  = detail::symv_strip_rows;
            constexpr std::size_t Bk = detail::symv_strip_block;
            const std::size_t ns = static_cast<std::size_t>(n);
            for (; static_cast<std::size_t>(i) + R <= ns; i += static_cast<size_type>(R)) {
                const std::size_t i0 = static_cast<std::size_t>(i);
                const TA* p[R];
                TX xr[R];
                value_type s[R];
                for (std::size_t r = 0; r < R; ++r) {
                    p[r]  = &A(static_cast<size_type>(i0 + r), 0);
                    xr[r] = x(static_cast<size_type>(i0 + r));
                    s[r]  = math::zero<value_type>();
                }
                for (std::size_t j0 = 0; j0 < i0; j0 += Bk) {
                    const std::size_t len = std::min(Bk, i0 - j0);
                    const TA* q[R];
                    for (std::size_t r = 0; r < R; ++r) q[r] = p[r] + j0;
                    detail::symv_strip_scatter<R>(len, q, xr, acc.data() + j0);
                    detail::symv_strip_gather<R>(len, q, s, x, j0);
                }
                for (std::size_t r = 0; r < R; ++r)
                    for (std::size_t j = i0; j <= i0 + r; ++j)
                        s[r] += p[r][j] * x(static_cast<size_type>(j));
                for (std::size_t r = 0; r < R; ++r) acc[i0 + r] = s[r];
                for (std::size_t r = 1; r < R; ++r)
                    for (std::size_t q = 0; q < r; ++q)
                        acc[i0 + q] += p[r][i0 + q] * xr[r];
            }
        }
        for (; i < n; ++i) {
            const auto* ai = &A(i, 0);
            const auto& xi = x(i);
            for (size_type j = 0; j < i; ++j)       // y(j) += a(j,i) x(i), j < i
                acc[j] += ai[j] * xi;
            value_type s = acc[i];                  // untouched so far: zero
            for (size_type j = 0; j <= i; ++j)      // y(i) += a(i,j) x(j), j <= i
                s += ai[j] * x(j);
            acc[i] = s;
        }
        for (size_type i = 0; i < n; ++i)
            y(i) = beta_zero ? value_type(alpha * acc[i]) : value_type(alpha * acc[i] + beta * y(i));
        return;
    }
    for (size_type i = 0; i < n; ++i) {
        if (alpha_zero) {
            y(i) = beta_zero ? math::zero<value_type>() : value_type(beta * y(i));
            continue;
        }
        auto acc = math::zero<value_type>();
        for (size_type j = 0; j < n; ++j)
            acc += a(i, j) * x(j);
        y(i) = beta_zero ? value_type(alpha * acc) : value_type(alpha * acc + beta * y(i));
    }
}

} // namespace mtl
