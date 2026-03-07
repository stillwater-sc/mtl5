#pragma once
// MTL5 -- Unit vector factory: returns dense_vector with 1 at position k, 0 elsewhere
#include <cstddef>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/math/identity.hpp>

namespace mtl::vec {

/// Create a unit vector of size n with 1 at position k, 0 elsewhere.
template <typename Value = double>
dense_vector<Value> unit_vector(std::size_t n, std::size_t k) {
    assert(k < n && "unit_vector: k must be less than n");
    dense_vector<Value> v(n, math::zero<Value>());
    v(k) = math::one<Value>();
    return v;
}

} // namespace mtl::vec

namespace mtl { using vec::unit_vector; }
