#pragma once
// MTL5 -- Maximum element in a collection
#include <limits>
#include <mtl/concepts/collection.hpp>

namespace mtl {

/// Returns the maximum element value
template <Collection C>
auto max(const C& c) {
    using T = typename C::value_type;
    auto result = std::numeric_limits<T>::lowest();
    for (auto it = c.begin(); it != c.end(); ++it) {
        if (*it > result) result = *it;
    }
    return result;
}

} // namespace mtl
