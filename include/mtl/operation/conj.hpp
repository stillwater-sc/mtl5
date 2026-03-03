#pragma once
// MTL5 — Element-wise complex conjugation
#include <mtl/concepts/vector.hpp>
#include <mtl/concepts/scalar.hpp>
#include <mtl/functor/scalar/conj.hpp>
#include <mtl/vec/dense_vector.hpp>

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

} // namespace mtl
