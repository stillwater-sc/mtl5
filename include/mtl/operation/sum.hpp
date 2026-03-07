#pragma once
// MTL5 -- Sum of all elements in a collection
#include <mtl/concepts/collection.hpp>
#include <mtl/math/identity.hpp>

namespace mtl {

/// Returns the sum of all elements: sum(c[i])
template <Collection C>
auto sum(const C& c) {
    using T = typename C::value_type;
    auto acc = math::zero<T>();
    for (auto it = c.begin(); it != c.end(); ++it) {
        acc += *it;
    }
    return acc;
}

} // namespace mtl
