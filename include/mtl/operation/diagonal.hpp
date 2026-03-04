#pragma once
// MTL5 — Extract diagonal of a matrix into a dense_vector
#include <algorithm>
#include <mtl/concepts/matrix.hpp>
#include <mtl/vec/dense_vector.hpp>

namespace mtl {

/// Returns a dense_vector containing the diagonal entries A(i,i)
template <Matrix M>
auto diagonal(const M& A) {
    auto n = std::min(A.num_rows(), A.num_cols());
    vec::dense_vector<typename M::value_type> v(n);
    for (typename M::size_type i = 0; i < n; ++i)
        v(i) = A(i, i);
    return v;
}

} // namespace mtl
