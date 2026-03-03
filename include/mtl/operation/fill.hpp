#pragma once
// MTL5 — Fill collection with a constant value
#include <algorithm>
#include <mtl/concepts/collection.hpp>

namespace mtl {

/// Fill all elements of a collection with the given value
template <MutableCollection C>
void fill(C& c, const typename C::value_type& val) {
    std::fill(c.begin(), c.end(), val);
}

} // namespace mtl
