#pragma once
// MTL5 -- Minimum element in a collection
#include <limits>
#include <mtl/concepts/collection.hpp>

namespace mtl {

/// Returns the minimum element value
template <Collection C>
auto min(const C& c) {
    using T = typename C::value_type;
    auto result = std::numeric_limits<T>::max();
    for (auto it = c.begin(); it != c.end(); ++it) {
        if (*it < result) result = *it;
    }
    return result;
}

} // namespace mtl
