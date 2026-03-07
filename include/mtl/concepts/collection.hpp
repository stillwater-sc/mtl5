#pragma once
// MTL5 -- Collection concepts (replaces MTL4 concept/collection.hpp)
#include <cstddef>
#include <concepts>

namespace mtl {

/// A read-only collection with value_type, size_type, and size()
template <typename T>
concept Collection = requires(const T& c) {
    typename T::value_type;
    typename T::size_type;
    { c.size() } -> std::convertible_to<typename T::size_type>;
};

/// A mutable collection allowing element modification
template <typename T>
concept MutableCollection = Collection<T> && requires(T& c) {
    typename T::reference;
};

} // namespace mtl
