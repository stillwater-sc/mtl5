#pragma once
// MTL5 -- Lower triangular solve (forward substitution)
// Solves L*x = b where L is lower triangular.
//
// Mixed precision: pass an explicit `Accumulator` to form each row's
// Sigma L(i,j)*x(j) reduction in a precision distinct from the operand type
// (#261, Part C). The subtraction from b(i) and division by L(i,i) happen
// AFTER the reduction is rounded out, same principle as symv's alpha/beta
// combine outside the loop -- not inside the accumulator.
// Default Accumulator = void keeps the generic path byte for byte.
//
// The default path takes W rows at a time (detail::lower_trisolve_rows, #749,
// with W from interface::interleave_rows, #772), bit-identical to the one-row
// loop. upper_trisolve cannot do the same: its row i sums in ascending j, so
// its FIRST term needs x(i+1), and no two rows' reductions can overlap
// without reordering them.
//
// Traversal order (forward, j < i) is a genuine data dependency, not just an
// overwrite hazard: x(j) for j < i must already be SOLVED (not merely
// original) before row i can be computed. Unrelated to accumulator choice,
// unchanged by this patch.
#include <cassert>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <mtl/concepts/matrix.hpp>
#include <mtl/concepts/vector.hpp>
#include <mtl/math/identity.hpp>
#include <mtl/math/accumulator_traits.hpp>
#include <mtl/interface/dispatch_traits.hpp>
#include <mtl/interface/interleave_rows.hpp>
#include <mtl/detail/scalar_chain.hpp>

namespace mtl {

namespace detail {

/// The default-accumulator forward substitution on rows [i, n), W rows per
/// block (#749, #772).
///
/// Rows i0..i0+W-1 sum their shared columns j < i0, which are already solved,
/// together: W independent chains instead of one serial reduction, which hides
/// the add latency. Then the W x W diagonal block finishes in order: row i0
/// completes, row i0+1 adds its term in x(i0) and completes, and so on. Every
/// row still sums the same terms in ascending j with the same expressions, so
/// x is bit-identical to the one-row loop for every W. The in-place overload
/// passes x as b: x(i) is read before row i writes it, as before. The rows
/// left over go to W / 2, down to W = 1, which is the one-row loop.
template <std::size_t W, typename M, typename VecX, typename VecB>
void lower_trisolve_rows(const M& L, VecX& x, const VecB& b, bool unit_diag,
                         typename M::size_type i = 0) {
    static_assert(W >= 1);
    using value_type = typename VecX::value_type;
    using size_type  = typename M::size_type;
    const size_type n = L.num_rows();
    [&]<std::size_t... K>(std::index_sequence<K...>) {
        for (; n - i >= W; i += W) {
            const size_type i0 = i;
            value_type s[W] = {(static_cast<void>(K), math::zero<value_type>())...};
            for (size_type j = 0; j < i0; ++j) {
                const auto& xj = x(j);
                ((s[K] += L(i0 + K, j) * xj), ...);
                (keep_scalar<keep_scalar_v<value_type, M, W>>(s[K]), ...);   // #776
            }
            ([&] {
                constexpr size_type k = K;
                for (size_type j = i0; j < i0 + k; ++j) s[K] += L(i0 + k, j) * x(j);
                if (unit_diag) x(i0 + k) = b(i0 + k) - s[K];
                else           x(i0 + k) = (b(i0 + k) - s[K]) / L(i0 + k, i0 + k);
            }(), ...);
        }
    }(std::make_index_sequence<W>{});
    if constexpr (W > 1) {
        if (i < n) lower_trisolve_rows<W / 2>(L, x, b, unit_diag, i);
    }
}

} // namespace detail

/// Solve L*x = b by forward substitution where L is lower triangular.
/// x is overwritten with the solution.
/// If unit_diag is true, the diagonal of L is assumed to be 1.
template <typename Accumulator = void, Matrix M, Vector VecX, Vector VecB>
void lower_trisolve(const M& L, VecX& x, const VecB& b, bool unit_diag = false) {
    using value_type = typename VecX::value_type;
    using size_type  = typename M::size_type;
    const size_type n = L.num_rows();
    assert(L.num_cols() == n && x.size() == n && b.size() == n);

    if constexpr (!interface::accumulator_allows_blas_v<Accumulator>) {
        using Value = std::common_type_t<value_type, typename M::value_type>;
        using AT = math::accumulator_traits<Accumulator, Value>;
        for (size_type i = 0; i < n; ++i) {
            Accumulator acc{};
            AT::clear(acc);
            for (size_type j = 0; j < i; ++j)
                AT::add_product(acc, static_cast<Value>(L(i, j)), static_cast<Value>(x(j)));
            const value_type sum = AT::template value<value_type>(acc);
            if (unit_diag)
                x(i) = b(i) - sum;
            else
                x(i) = (b(i) - sum) / L(i, i);
        }
        return;
    }

    detail::lower_trisolve_rows<
        interface::interleave_rows_v<value_type, interface::kernel::lower_trisolve>>(L, x, b, unit_diag);
}

/// In-place variant: solve L*x = x (b is x on input)
template <typename Accumulator = void, Matrix M, Vector VecX>
void lower_trisolve(const M& L, VecX& x, bool unit_diag = false) {
    using value_type = typename VecX::value_type;
    using size_type  = typename M::size_type;
    const size_type n = L.num_rows();
    assert(L.num_cols() == n && x.size() == n);

    if constexpr (!interface::accumulator_allows_blas_v<Accumulator>) {
        using Value = std::common_type_t<value_type, typename M::value_type>;
        using AT = math::accumulator_traits<Accumulator, Value>;
        for (size_type i = 0; i < n; ++i) {
            Accumulator acc{};
            AT::clear(acc);
            for (size_type j = 0; j < i; ++j)
                AT::add_product(acc, static_cast<Value>(L(i, j)), static_cast<Value>(x(j)));
            const value_type sum = AT::template value<value_type>(acc);
            if (unit_diag)
                x(i) = x(i) - sum;
            else
                x(i) = (x(i) - sum) / L(i, i);
        }
        return;
    }

    detail::lower_trisolve_rows<
        interface::interleave_rows_v<value_type, interface::kernel::lower_trisolve>>(L, x, x, unit_diag);
}

} // namespace mtl
