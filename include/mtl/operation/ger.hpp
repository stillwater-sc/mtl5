#pragma once
// MTL5 -- ger: A <- alpha*x*y^T + A  (BLAS level-2 general rank-1 update)
// Generic path for any Matrix/Vector; BLAS dispatch for dense float/double
// when available, in either orientation (#849).
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <limits>
#include <type_traits>

#include <mtl/concepts/scalar.hpp>
#include <mtl/concepts/vector.hpp>
#include <mtl/concepts/matrix.hpp>
#include <mtl/math/accumulator_traits.hpp>
#include <mtl/interface/dispatch_traits.hpp>
#ifdef MTL5_HAS_BLAS
#include <mtl/interface/blas.hpp>
#endif

namespace mtl {

namespace detail {

/// Whether ger<Accumulator>(alpha, x, y, A) calls ?ger (for sizes that fit an
/// int). Either orientation of A qualifies: a row-major A is the column-major
/// A^T, updated by ?ger with x and y swapped (#849).
///
/// ?ger takes alpha in A's element type T, so the scalar S must not be wider
/// than T: common_type_t<S, T> is T. An int or a float alpha with double A
/// converts as the generic expression would convert it; a double alpha with
/// float A would be rounded to float first (2^-150 becomes 0, and ?ger then
/// returns without an update), so it takes the generic path, which forms
/// alpha x(i) y(j) in double.
template <typename Accumulator, typename S, typename M, typename VX, typename VY>
inline constexpr bool ger_dispatches_blas_v = [] {
#ifdef MTL5_HAS_BLAS
    if constexpr (interface::accumulator_allows_blas_v<Accumulator> && interface::BlasDenseMatrix<M> &&
                  interface::BlasDenseVector<VX> && interface::BlasDenseVector<VY>)
        return std::is_same_v<typename M::value_type, typename VX::value_type> &&
               std::is_same_v<typename M::value_type, typename VY::value_type> &&
               std::is_same_v<std::common_type_t<S, typename M::value_type>, typename M::value_type>;
    else
        return false;
#else
    return false;
#endif
}();

} // namespace detail

/// Rank-1 update: A(i,j) += alpha * x(i) * y(j).
///
/// Mixed precision: pass an explicit `Accumulator` to fold the existing
/// A(i,j), and the product alpha*x(i)*y(j), into one rounding event (#261,
/// Part C) instead of two (product round, then add round). Default
/// `Accumulator = void` keeps the BLAS / generic dispatch byte for byte.
///
/// Gated on `interface::accumulator_allows_blas_v`, same as symv/gemv: a
/// custom accumulator cannot be honored by external BLAS.
///
/// Unlike symv's per-row reduction, each entry here is a single term, so the
/// accumulator is seeded with `assign(A(i,j))` -- not `clear` -- then given
/// one `add_product`, matching axpy's one-term seeded pattern (#511) rather
/// than gemv/symv's zero-seeded multi-term row loop.
template <typename Accumulator = void, Scalar S, Vector VX, Vector VY, Matrix M>
void ger(const S& alpha, const VX& x, const VY& y, M& A) {
    using size_type = typename M::size_type;
    const size_type m = A.num_rows();
    const size_type n = A.num_cols();
    assert(x.size() == m && y.size() == n);

    if constexpr (!interface::accumulator_allows_blas_v<Accumulator>) {
        using value_type = typename M::value_type;
        using Value = std::common_type_t<S, value_type, typename VX::value_type, typename VY::value_type>;
        using AT = math::accumulator_traits<Accumulator, Value>;
        for (size_type i = 0; i < m; ++i) {
            const Value axi = static_cast<Value>(alpha) * static_cast<Value>(x(i));
            for (size_type j = 0; j < n; ++j) {
                Accumulator acc{};
                AT::assign(acc, static_cast<Value>(A(i, j)));
                AT::add_product(acc, axi, static_cast<Value>(y(j)));
                A(i, j) = AT::template value<value_type>(acc);
            }
        }
        return;
    }

#ifdef MTL5_HAS_BLAS
    if constexpr (detail::ger_dispatches_blas_v<Accumulator, S, M, VX, VY>) {
        using T = typename M::value_type;
        const auto imax = static_cast<size_type>(std::numeric_limits<int>::max());
        if (m == 0 || n == 0) return;   // nothing to update; skip BLAS (#693)
        if (m <= imax && n <= imax) {
            const int mi = static_cast<int>(m), ni = static_cast<int>(n);
            if constexpr (interface::is_row_major_v<M>) {
                // A row-major m x n buffer is the column-major n x m A^T, and
                // A^T += alpha y x^T: ?ger with x and y swapped, lda = n.
                interface::blas::ger(ni, mi, static_cast<T>(alpha), y.data(), 1, x.data(), 1,
                                     A.data(), ni);
            } else {
                // Column-major storage: the leading dimension is the row count.
                interface::blas::ger(mi, ni, static_cast<T>(alpha), x.data(), 1, y.data(), 1,
                                     A.data(), mi);
            }
            return;
        }
    }
#endif
    for (size_type i = 0; i < m; ++i)
        for (size_type j = 0; j < n; ++j)
            A(i, j) += alpha * x(i) * y(j);
}

} // namespace mtl
