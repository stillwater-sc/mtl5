#pragma once
// MTL5 -- Expression template storage helper
// Determines whether to store an operand by value or by const reference.
// - lvalue operand (T deduced as X&) -> store as const X& (no copy)
// - rvalue operand (T deduced as X)  -> store as X (moved from rvalue)
#include <type_traits>

namespace mtl::detail {

/// Given the deduced forwarding-reference type T:
/// - if T is an lvalue reference (X& or const X&), store as const X&
/// - if T is a non-reference (rvalue), store as X (by value)
template <typename T>
using expr_store_t = std::conditional_t<
    std::is_lvalue_reference_v<T>,
    const std::remove_reference_t<T>&,
    std::remove_cvref_t<T>
>;

/// Strip reference and cv to get the underlying type for member access
template <typename T>
using expr_raw_t = std::remove_cvref_t<T>;

} // namespace mtl::detail
