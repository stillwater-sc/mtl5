#pragma once
// MTL5 — Set all elements to zero using math::zero<T>()
#include <algorithm>
#include <mtl/concepts/collection.hpp>
#include <mtl/math/identity.hpp>

namespace mtl {

/// Set all elements of a collection to zero
template <MutableCollection C>
void set_to_zero(C& c) {
    using value_type = typename C::value_type;
    std::fill(c.begin(), c.end(), math::zero<value_type>());
}

} // namespace mtl
