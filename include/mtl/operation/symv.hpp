#pragma once
// MTL5 -- symv: y <- alpha*A*x + beta*y  for symmetric A  (BLAS level-2)
// The generic path reads the full matrix (correct for any symmetric A); the
// BLAS path reads a single triangle. BLAS dispatch for column-major dense
// float/double when available.
#include <cassert>
#include <cstddef>
#include <limits>
#include <type_traits>

#include <mtl/concepts/scalar.hpp>
#include <mtl/concepts/vector.hpp>
#include <mtl/concepts/matrix.hpp>
#include <mtl/math/identity.hpp>
#include <mtl/math/accumulator_traits.hpp>
#include <mtl/interface/dispatch_traits.hpp>
#ifdef MTL5_HAS_BLAS
#include <mtl/interface/blas.hpp>
#endif

namespace mtl {

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

    if constexpr (!interface::accumulator_allows_blas_v<Accumulator>) {
        using Value = std::common_type_t<S, typename M::value_type, typename VX::value_type>;
        using AT = math::accumulator_traits<Accumulator, Value>;
        for (size_type i = 0; i < n; ++i) {
            Accumulator acc{};
            AT::clear(acc);
            for (size_type j = 0; j < n; ++j)
                AT::add_product(acc, static_cast<Value>(A(i, j)), static_cast<Value>(x(j)));
            const value_type reduced = AT::template value<value_type>(acc);
            y(i) = alpha * reduced + beta * y(i);
        }
        return;
    }

#ifdef MTL5_HAS_BLAS
    if constexpr (interface::BlasDenseMatrix<M> && interface::BlasDenseVector<VX> &&
                  interface::BlasDenseVector<VY> && !interface::is_row_major_v<M> &&
                  std::is_same_v<value_type, typename M::value_type> &&
                  std::is_same_v<value_type, typename VX::value_type>) {
        using T = value_type;
        if (n <= static_cast<size_type>(std::numeric_limits<int>::max())) {
            // Symmetric: 'L' reads the lower triangle (== upper for symmetric A).
            interface::blas::symv('L', static_cast<int>(n), static_cast<T>(alpha),
                                  A.data(), static_cast<int>(n), x.data(), 1,
                                  static_cast<T>(beta), y.data(), 1);
            return;
        }
    }
#endif
    for (size_type i = 0; i < n; ++i) {
        auto acc = math::zero<value_type>();
        for (size_type j = 0; j < n; ++j)
            acc += A(i, j) * x(j);
        y(i) = alpha * acc + beta * y(i);
    }
}

} // namespace mtl
