#pragma once
// MTL5 -- Element-wise complex conjugation
#include <mtl/concepts/vector.hpp>
#include <mtl/concepts/matrix.hpp>
#include <mtl/concepts/scalar.hpp>
#include <mtl/functor/scalar/conj.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/mat/dense2D.hpp>

namespace mtl {

/// Element-wise conjugation for vectors
template <Vector V>
auto conj(const V& v) {
    using T = typename V::value_type;
    vec::dense_vector<T> result(v.size());
    for (typename V::size_type i = 0; i < v.size(); ++i) {
        result(i) = functor::scalar::conj<T>::apply(v(i));
    }
    return result;
}

/// Element-wise conjugation for matrices
template <Matrix M>
auto conj(const M& m) {
    using T = typename M::value_type;
    mat::dense2D<T> result(m.num_rows(), m.num_cols());
    for (typename M::size_type r = 0; r < m.num_rows(); ++r) {
        for (typename M::size_type c = 0; c < m.num_cols(); ++c) {
            result(r, c) = functor::scalar::conj<T>::apply(m(r, c));
        }
    }
    return result;
}

} // namespace mtl
