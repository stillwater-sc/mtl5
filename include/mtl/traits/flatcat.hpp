#pragma once
// MTL5 -- Flat category trait
#include <mtl/traits/category.hpp>

namespace mtl::traits {

/// Maps T to its category -- trivial alias for now
template <typename T>
using flatcat_t = category_t<T>;

} // namespace mtl::traits
