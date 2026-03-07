#pragma once
// MTL5 -- Scale collection by a scalar factor
#include <mtl/concepts/scalar.hpp>
#include <mtl/concepts/collection.hpp>

namespace mtl {

/// In-place scale: c[i] *= alpha
template <Scalar S, MutableCollection C>
void scale(const S& alpha, C& c) {
    for (auto it = c.begin(); it != c.end(); ++it) {
        *it *= alpha;
    }
}

/// Returns a scaled copy of a vector
template <Scalar S, Collection C>
auto scaled(const S& alpha, const C& c) {
    auto result = c;  // copy
    scale(alpha, result);
    return result;
}

} // namespace mtl
