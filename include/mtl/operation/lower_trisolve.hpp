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
// The default path takes four rows at a time (detail::lower_trisolve_rows,
// #749), bit-identical to the one-row loop. upper_trisolve cannot do the
// same: its row i sums in ascending j, so its FIRST term needs x(i+1), and
// no two rows' reductions can overlap without reordering them.
//
// Traversal order (forward, j < i) is a genuine data dependency, not just an
// overwrite hazard: x(j) for j < i must already be SOLVED (not merely
// original) before row i can be computed. Unrelated to accumulator choice,
// unchanged by this patch.
#include <cassert>
#include <type_traits>
#include <mtl/concepts/matrix.hpp>
#include <mtl/concepts/vector.hpp>
#include <mtl/math/identity.hpp>
#include <mtl/math/accumulator_traits.hpp>
#include <mtl/interface/dispatch_traits.hpp>

namespace mtl {

namespace detail {

/// The default-accumulator forward substitution, four rows at a time (#749).
///
/// Rows i0..i0+3 sum their shared columns j < i0, which are already solved,
/// together: four independent chains instead of one serial reduction, which
/// hides the add latency. Then the 4x4 diagonal block finishes in order: row
/// i0 completes, row i0+1 adds its term in x(i0) and completes, and so on.
/// Every row still sums the same terms in ascending j with the same
/// expressions, so x is bit-identical to the one-row loop. The in-place
/// overload passes x as b: x(i) is read before row i writes it, as before.
template <typename M, typename VecX, typename VecB>
void lower_trisolve_rows(const M& L, VecX& x, const VecB& b, bool unit_diag) {
    using value_type = typename VecX::value_type;
    using size_type  = typename M::size_type;
    const size_type n = L.num_rows();
    size_type i0 = 0;
    for (; n - i0 >= 4; i0 += 4) {
        const size_type i1 = i0 + 1, i2 = i0 + 2, i3 = i0 + 3;
        auto s0 = math::zero<value_type>();
        auto s1 = math::zero<value_type>();
        auto s2 = math::zero<value_type>();
        auto s3 = math::zero<value_type>();
        for (size_type j = 0; j < i0; ++j) {
            s0 += L(i0, j) * x(j);
            s1 += L(i1, j) * x(j);
            s2 += L(i2, j) * x(j);
            s3 += L(i3, j) * x(j);
        }
        if (unit_diag) x(i0) = b(i0) - s0; else x(i0) = (b(i0) - s0) / L(i0, i0);
        s1 += L(i1, i0) * x(i0);
        if (unit_diag) x(i1) = b(i1) - s1; else x(i1) = (b(i1) - s1) / L(i1, i1);
        s2 += L(i2, i0) * x(i0);
        s2 += L(i2, i1) * x(i1);
        if (unit_diag) x(i2) = b(i2) - s2; else x(i2) = (b(i2) - s2) / L(i2, i2);
        s3 += L(i3, i0) * x(i0);
        s3 += L(i3, i1) * x(i1);
        s3 += L(i3, i2) * x(i2);
        if (unit_diag) x(i3) = b(i3) - s3; else x(i3) = (b(i3) - s3) / L(i3, i3);
    }
    for (size_type i = i0; i < n; ++i) {
        auto sum = math::zero<value_type>();
        for (size_type j = 0; j < i; ++j)
            sum += L(i, j) * x(j);
        if (unit_diag)
            x(i) = b(i) - sum;
        else
            x(i) = (b(i) - sum) / L(i, i);
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

    detail::lower_trisolve_rows(L, x, b, unit_diag);
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

    detail::lower_trisolve_rows(L, x, x, unit_diag);
}

} // namespace mtl
