#pragma once
// MTL5 -- Extract diagonal of a matrix into a dense_vector,
//         and construct a diagonal sparse matrix from a vector.
#include <algorithm>
#include <mtl/concepts/matrix.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>

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

/// Construct an n x n diagonal sparse matrix from a vector (inverse of diagonal)
template <typename T>
auto diag(const vec::dense_vector<T>& v) {
    auto n = v.size();
    mat::compressed2D<T> D(n, n);
    {
        mat::inserter<mat::compressed2D<T>> ins(D);
        for (typename vec::dense_vector<T>::size_type i = 0; i < n; ++i)
            ins[i][i] << v(i);
    }
    return D;
}

} // namespace mtl
