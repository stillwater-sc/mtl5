#pragma once
// MTL5 — Lower triangular solve (forward substitution)
// Solves L*x = b where L is lower triangular.
#include <cassert>
#include <mtl/concepts/matrix.hpp>
#include <mtl/concepts/vector.hpp>
#include <mtl/math/identity.hpp>

namespace mtl {

/// Solve L*x = b by forward substitution where L is lower triangular.
/// x is overwritten with the solution.
/// If unit_diag is true, the diagonal of L is assumed to be 1.
template <Matrix M, Vector VecX, Vector VecB>
void lower_trisolve(const M& L, VecX& x, const VecB& b, bool unit_diag = false) {
    using value_type = typename VecX::value_type;
    using size_type  = typename M::size_type;
    const size_type n = L.num_rows();
    assert(L.num_cols() == n && x.size() == n && b.size() == n);

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
template <Matrix M, Vector VecX>
void lower_trisolve(const M& L, VecX& x, bool unit_diag = false) {
    using value_type = typename VecX::value_type;
    using size_type  = typename M::size_type;
    const size_type n = L.num_rows();
    assert(L.num_cols() == n && x.size() == n);

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
