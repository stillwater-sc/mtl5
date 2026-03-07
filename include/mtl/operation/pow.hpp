#pragma once
// MTL5 -- Element-wise power (binary: scalar exponent)
#include <cmath>
#include <mtl/concepts/vector.hpp>
#include <mtl/concepts/matrix.hpp>
#include <mtl/concepts/scalar.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/mat/dense2D.hpp>

namespace mtl {

/// Element-wise pow for vectors: pow(v[i], exponent)
template <Vector V, Scalar S>
auto pow(const V& v, const S& exponent) {
    using T = typename V::value_type;
    vec::dense_vector<T> result(v.size());
    for (typename V::size_type i = 0; i < v.size(); ++i) {
        using std::pow;
        result(i) = pow(v(i), exponent);
    }
    return result;
}

/// Element-wise pow for matrices: pow(m[r][c], exponent)
template <Matrix M, Scalar S>
auto pow(const M& m, const S& exponent) {
    using T = typename M::value_type;
    mat::dense2D<T> result(m.num_rows(), m.num_cols());
    for (typename M::size_type r = 0; r < m.num_rows(); ++r) {
        for (typename M::size_type c = 0; c < m.num_cols(); ++c) {
            using std::pow;
            result(r, c) = pow(m(r, c), exponent);
        }
    }
    return result;
}

} // namespace mtl
