#pragma once
// MTL5 -- Element-wise complementary error function
#include <cmath>
#include <mtl/concepts/vector.hpp>
#include <mtl/concepts/matrix.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/mat/dense2D.hpp>

namespace mtl {

/// Element-wise erfc for vectors
template <Vector V>
auto erfc(const V& v) {
    using T = typename V::value_type;
    vec::dense_vector<T> result(v.size());
    for (typename V::size_type i = 0; i < v.size(); ++i) {
        using std::erfc;
        result(i) = erfc(v(i));
    }
    return result;
}

/// Element-wise erfc for matrices
template <Matrix M>
auto erfc(const M& m) {
    using T = typename M::value_type;
    mat::dense2D<T> result(m.num_rows(), m.num_cols());
    for (typename M::size_type r = 0; r < m.num_rows(); ++r) {
        for (typename M::size_type c = 0; c < m.num_cols(); ++c) {
            using std::erfc;
            result(r, c) = erfc(m(r, c));
        }
    }
    return result;
}

} // namespace mtl
