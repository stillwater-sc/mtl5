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
// with W from interface::interleave_rows, #772), or on a contiguous
// column-major L the column sweep (#777), both bit-identical to the one-row
// loop when a product and its add are contracted the same way in both forms
// (-ffp-contract=off, as the tests build; with contraction on the last bit
// can differ, #849). upper_trisolve cannot do the same: its row i sums in ascending j, so
// its FIRST term needs x(i+1), and no two rows' reductions can overlap
// without reordering them.
//
// Traversal order (forward, j < i) is a genuine data dependency, not just an
// overwrite hazard: x(j) for j < i must already be SOLVED (not merely
// original) before row i can be computed. Unrelated to accumulator choice,
// unchanged by this patch.
#include <algorithm>
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
#include <mtl/detail/column_sweep.hpp>

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
                (keep_scalar<keep_scalar_triangular_v<value_type, M, W>>(s[K]), ...);   // #776, #795
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

/// The default-accumulator forward substitution on a contiguous column-major
/// L, S rows per strip, top-down (#777; see detail/column_sweep.hpp).
///
/// A strip of rows [r0, r1) first adds the columns j < r0, which are already
/// solved, one column at a time down the strip. Then its diagonal block goes
/// column by column: row j is solved, and its x(j) is added to the strip's
/// rows below it. Every row still sums L(i, j) x(j) in ascending j with the
/// same expressions as the one-row loop, so x is bit-identical to it. The
/// in-place overload passes x as b: b(j) is read when row j is solved, before
/// anything writes x(j).
///
/// The solved columns j < r0, and the diagonal block's columns below each
/// group of C, go C at a time (#884, see column_sweep_columns); C = 1 is the
/// sweep as #777 wrote it.
template <std::size_t S, std::size_t C = column_sweep_columns, typename M, typename VecX, typename VecB>
void lower_trisolve_column_sweep(const M& L, VecX& x, const VecB& b, bool unit_diag) {
    static_assert(S >= 1);
    using value_type = typename VecX::value_type;
    using size_type  = typename M::size_type;
    const std::size_t n = static_cast<std::size_t>(L.num_rows());
    const auto* a = L.data();
    value_type s[S];
    const auto x_at = [&](std::size_t j) -> decltype(auto) { return x(static_cast<size_type>(j)); };
    const auto add = [](const value_type& t, const auto& lij, const auto& xj) -> value_type { return t + lij * xj; };
    // Columns [cb, ce) into the strip rows [i0, i0 + ni), which s[i0 - r0 ..]
    // holds: C > 1 takes strip_columns, C = 1 one column at a time (#777).
    const auto cols = [&](std::size_t r0, std::size_t i0, std::size_t ni, std::size_t cb, std::size_t ce) {
        if constexpr (C == 1) detail::sweep_columns<1>(s + (i0 - r0), a, n, i0, ni, cb, ce, x_at, add);
        else                  detail::strip_columns<C>(s + (i0 - r0), a, n, i0, ni, cb, ce, x_at, add);
    };
    for (std::size_t r0 = 0; r0 < n; r0 += S) {
        const std::size_t r1 = std::min(n, r0 + S);
        for (std::size_t l = 0; l < r1 - r0; ++l) s[l] = math::zero<value_type>();
        cols(r0, r0, r1 - r0, 0, r0);
        // The diagonal block C columns at a time (#884): solve the group's rows
        // [j0, je) in turn, each adding its column to the group's rows below
        // it, then add the group's columns to the strip's rows i >= je.
        for (std::size_t j0 = r0; j0 < r1; j0 += C) {
            const std::size_t je = std::min(j0 + C, r1);
            for (std::size_t j = j0; j < je; ++j) {
                const auto jj = static_cast<size_type>(j);
                if (unit_diag) x(jj) = b(jj) - s[j - r0];
                else           x(jj) = (b(jj) - s[j - r0]) / L(jj, jj);
                const auto& xj = x(jj);
                const auto* col = a + j * n;
                for (std::size_t i = j + 1; i < je; ++i) s[i - r0] += col[i] * xj;
            }
            cols(r0, je, r1 - je, j0, je);
        }
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

    // A column-major L above column_sweep_trsv_row_kernel_n takes the column
    // sweep (#777, #884), anything else the row kernels. The same bits.
    if constexpr (detail::column_sweep_v<value_type, M>) {
        if (static_cast<std::size_t>(n) > detail::column_sweep_trsv_row_kernel_n) {
            detail::lower_trisolve_column_sweep<detail::column_strip_rows<value_type>>(L, x, b, unit_diag);   // #777
            return;
        }
    }
    interface::with_interleave_rows<value_type, interface::kernel::lower_trisolve>([&](auto w) {
        detail::lower_trisolve_rows<decltype(w)::value>(L, x, b, unit_diag);
    });
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

    if constexpr (detail::column_sweep_v<value_type, M>) {
        if (static_cast<std::size_t>(n) > detail::column_sweep_trsv_row_kernel_n) {
            detail::lower_trisolve_column_sweep<detail::column_strip_rows<value_type>>(L, x, x, unit_diag);   // #777
            return;
        }
    }
    interface::with_interleave_rows<value_type, interface::kernel::lower_trisolve>([&](auto w) {
        detail::lower_trisolve_rows<decltype(w)::value>(L, x, x, unit_diag);
    });
}

} // namespace mtl
