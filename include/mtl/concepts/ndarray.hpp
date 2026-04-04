#pragma once
// MTL5 -- NdArray concept for N-dimensional array types
#include <concepts>
#include <cstddef>

#include <mtl/concepts/collection.hpp>

namespace mtl {

/// An N-dimensional array with rank, shape, and multi-dimensional element access.
template <typename T>
concept NdArray = Collection<T> && requires(const T& a) {
    { T::rank }       -> std::convertible_to<std::size_t>;
    { a.get_shape() };
    { a.get_strides() };
    { a.data() };
};

/// A mutable NdArray that allows element modification.
template <typename T>
concept MutableNdArray = NdArray<T> && MutableCollection<T> && requires(T& a) {
    { a.data() };
};

} // namespace mtl
