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
#include <mtl/interface/dispatch_traits.hpp>
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
        assert(!detail::same_object(x, y) && "symv: y must not alias x");
        if (n == 0) return;
        if (alpha_zero) {
            for (size_type i = 0; i < n; ++i)
                y(i) = beta_zero ? math::zero<value_type>() : value_type(beta * y(i));
            return;
        }
        std::vector<value_type> acc(static_cast<std::size_t>(n), math::zero<value_type>());
        for (size_type i = 0; i < n; ++i) {
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
