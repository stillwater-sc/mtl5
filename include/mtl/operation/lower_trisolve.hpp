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

    for (size_type i = 0; i < n; ++i) {
        auto sum = math::zero<value_type>();
        for (size_type j = 0; j < i; ++j)
            sum += L(i, j) * x(j);
        if (unit_diag)
            x(i) = b(i) - sum;
        else
            x(i) = (b(i) - sum) / L(i, i);
    }
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

    for (size_type i = 0; i < n; ++i) {
        auto sum = math::zero<value_type>();
        for (size_type j = 0; j < i; ++j)
            sum += L(i, j) * x(j);
        if (unit_diag)
            x(i) = x(i) - sum;
        else
            x(i) = (x(i) - sum) / L(i, i);
    }
}

} // namespace mtl
