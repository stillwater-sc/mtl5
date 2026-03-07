#pragma once
// MTL5 -- Element-wise absolute value
#include <cmath>
#include <cstdlib>
#include <mtl/concepts/vector.hpp>
#include <mtl/concepts/matrix.hpp>
#include <mtl/concepts/magnitude.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/mat/dense2D.hpp>

namespace mtl {

/// Element-wise abs for vectors -> dense_vector<magnitude_t<T>>
template <Vector V>
auto abs(const V& v) {
    using mag_t = magnitude_t<typename V::value_type>;
    vec::dense_vector<mag_t> result(v.size());
    for (typename V::size_type i = 0; i < v.size(); ++i) {
        using std::abs;
        result(i) = abs(v(i));
    }
    return result;
}

/// Element-wise abs for matrices -> dense2D<magnitude_t<T>>
template <Matrix M>
auto abs(const M& m) {
    using mag_t = magnitude_t<typename M::value_type>;
    mat::dense2D<mag_t> result(m.num_rows(), m.num_cols());
    for (typename M::size_type r = 0; r < m.num_rows(); ++r) {
        for (typename M::size_type c = 0; c < m.num_cols(); ++c) {
            using std::abs;
            result(r, c) = abs(m(r, c));
        }
    }
    return result;
}

} // namespace mtl
