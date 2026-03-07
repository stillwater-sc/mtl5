#pragma once
// MTL5 -- Trait to distinguish expression templates from concrete types
#include <type_traits>

namespace mtl::traits {

/// Primary: concrete types are not expressions
template <typename T>
struct is_expression : std::false_type {};

template <typename T>
inline constexpr bool is_expression_v = is_expression<T>::value;

} // namespace mtl::traits
