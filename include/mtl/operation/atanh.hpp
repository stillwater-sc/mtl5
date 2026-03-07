#pragma once
// MTL5 — Element-wise inverse hyperbolic tangent
#include <cmath>
#include <mtl/concepts/vector.hpp>
#include <mtl/concepts/matrix.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/mat/dense2D.hpp>

namespace mtl {

/// Element-wise atanh for vectors
template <Vector V>
auto atanh(const V& v) {
    using T = typename V::value_type;
    vec::dense_vector<T> result(v.size());
    for (typename V::size_type i = 0; i < v.size(); ++i) {
        using std::atanh;
        result(i) = atanh(v(i));
    }
    return result;
}

/// Element-wise atanh for matrices
template <Matrix M>
auto atanh(const M& m) {
    using T = typename M::value_type;
    mat::dense2D<T> result(m.num_rows(), m.num_cols());
    for (typename M::size_type r = 0; r < m.num_rows(); ++r) {
        for (typename M::size_type c = 0; c < m.num_cols(); ++c) {
            using std::atanh;
            result(r, c) = atanh(m(r, c));
        }
    }
    return result;
}

} // namespace mtl
