#pragma once
// MTL5 -- Resource descriptor for workspace allocation
#include <mtl/concepts/collection.hpp>

namespace mtl {

/// Returns the resource descriptor of a collection (its size).
/// Named boundary for future distributed-vector extension.
template <Collection C>
auto resource(const C& c) { return c.size(); }

} // namespace mtl
