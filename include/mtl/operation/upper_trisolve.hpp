#pragma once
// MTL5 -- Upper triangular solve (back substitution)
// Solves U*x = b where U is upper triangular.
#include <cassert>
#include <mtl/concepts/matrix.hpp>
#include <mtl/concepts/vector.hpp>
#include <mtl/math/identity.hpp>

namespace mtl {

/// Solve U*x = b by back substitution where U is upper triangular.
/// x is overwritten with the solution.
/// If unit_diag is true, the diagonal of U is assumed to be 1.
template <Matrix M, Vector VecX, Vector VecB>
void upper_trisolve(const M& U, VecX& x, const VecB& b, bool unit_diag = false) {
    using value_type = typename VecX::value_type;
    using size_type  = typename M::size_type;
    const size_type n = U.num_rows();
    assert(U.num_cols() == n && x.size() == n && b.size() == n);

    for (size_type ii = 0; ii < n; ++ii) {
        size_type i = n - 1 - ii;
        auto sum = math::zero<value_type>();
        for (size_type j = i + 1; j < n; ++j)
            sum += U(i, j) * x(j);
        if (unit_diag)
            x(i) = b(i) - sum;
        else
            x(i) = (b(i) - sum) / U(i, i);
    }
}

/// In-place variant: solve U*x = x (b is x on input)
template <Matrix M, Vector VecX>
void upper_trisolve(const M& U, VecX& x, bool unit_diag = false) {
    using value_type = typename VecX::value_type;
    using size_type  = typename M::size_type;
    const size_type n = U.num_rows();
    assert(U.num_cols() == n && x.size() == n);

    for (size_type ii = 0; ii < n; ++ii) {
        size_type i = n - 1 - ii;
        auto sum = math::zero<value_type>();
        for (size_type j = i + 1; j < n; ++j)
            sum += U(i, j) * x(j);
        if (unit_diag)
            x(i) = x(i) - sum;
        else
            x(i) = (x(i) - sum) / U(i, i);
    }
}

} // namespace mtl
