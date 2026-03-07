#pragma once
// MTL5 -- Element-wise base-2 exponential
#include <cmath>
#include <mtl/concepts/vector.hpp>
#include <mtl/concepts/matrix.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/mat/dense2D.hpp>

namespace mtl {

/// Element-wise exp2 for vectors
template <Vector V>
auto exp2(const V& v) {
    using T = typename V::value_type;
    vec::dense_vector<T> result(v.size());
    for (typename V::size_type i = 0; i < v.size(); ++i) {
        using std::exp2;
        result(i) = exp2(v(i));
    }
    return result;
}

/// Element-wise exp2 for matrices
template <Matrix M>
auto exp2(const M& m) {
    using T = typename M::value_type;
    mat::dense2D<T> result(m.num_rows(), m.num_cols());
    for (typename M::size_type r = 0; r < m.num_rows(); ++r) {
        for (typename M::size_type c = 0; c < m.num_cols(); ++c) {
            using std::exp2;
            result(r, c) = exp2(m(r, c));
        }
    }
    return result;
}

} // namespace mtl
