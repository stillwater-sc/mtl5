#pragma once
// MTL5 — Operation tags for algebraic identities
// Replaces boost/numeric/linear_algebra/operators.hpp
#include <algorithm>
#include <functional>

namespace mtl::math {

/// Additive operation tag
template <typename T>
struct add {
    constexpr T operator()(const T& a, const T& b) const { return a + b; }
};

/// Multiplicative operation tag
template <typename T>
struct mult {
    constexpr T operator()(const T& a, const T& b) const { return a * b; }
};

/// Maximum operation tag
template <typename T>
struct max_op {
    constexpr T operator()(const T& a, const T& b) const { return std::max(a, b); }
};

/// Minimum operation tag
template <typename T>
struct min_op {
    constexpr T operator()(const T& a, const T& b) const { return std::min(a, b); }
};

} // namespace mtl::math
