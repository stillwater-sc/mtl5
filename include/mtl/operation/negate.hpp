#pragma once
// MTL5 — Element-wise negation
#include <mtl/concepts/vector.hpp>
#include <mtl/vec/dense_vector.hpp>

namespace mtl {

/// Element-wise negation for vectors: -v[i]
template <Vector V>
auto negate(const V& v) {
    using T = typename V::value_type;
    vec::dense_vector<T> result(v.size());
    for (typename V::size_type i = 0; i < v.size(); ++i) {
        result(i) = -v(i);
    }
    return result;
}

} // namespace mtl
