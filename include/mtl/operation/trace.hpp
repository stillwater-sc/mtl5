#pragma once
// MTL5 — Matrix trace: sum of diagonal elements
#include <algorithm>
#include <mtl/concepts/matrix.hpp>
#include <mtl/math/identity.hpp>

namespace mtl {

/// trace(A) = sum(A(i,i)) for i = 0..min(nrows,ncols)-1
template <Matrix M>
auto trace(const M& A) {
    using value_type = typename M::value_type;
    auto n = std::min(A.num_rows(), A.num_cols());
    auto acc = math::zero<value_type>();
    for (typename M::size_type i = 0; i < n; ++i)
        acc += A(i, i);
    return acc;
}

} // namespace mtl
