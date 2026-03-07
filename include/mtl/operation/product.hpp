#pragma once
// MTL5 -- Product of all elements in a collection
#include <mtl/concepts/collection.hpp>
#include <mtl/math/identity.hpp>

namespace mtl {

/// Returns the product of all elements: product(c[i])
template <Collection C>
auto product(const C& c) {
    using T = typename C::value_type;
    auto acc = math::one<T>();
    for (auto it = c.begin(); it != c.end(); ++it) {
        acc *= *it;
    }
    return acc;
}

} // namespace mtl
