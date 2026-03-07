#pragma once
// MTL5 -- LinearOperator concept for ITL (replaces duck-typing in MTL4 ITL)
#include <concepts>

namespace mtl {

/// Anything that supports A * x producing a vector-like result
/// Used by ITL Krylov solvers
template <typename A, typename X>
concept LinearOperator = requires(const A& a, const X& x) {
    { a * x };
};

} // namespace mtl
