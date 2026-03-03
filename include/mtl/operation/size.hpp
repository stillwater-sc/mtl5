#pragma once
// MTL5 — Free function: collection size
#include <mtl/concepts/collection.hpp>

namespace mtl {

/// Return the number of elements in a collection
template <Collection C>
auto size(const C& c) { return c.size(); }

} // namespace mtl
