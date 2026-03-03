#pragma once
// MTL5 — Vector concepts (replaces MTL4 concept/vector.hpp)
#include <mtl/concepts/collection.hpp>
#include <cstddef>

namespace mtl {

/// A 1D collection with indexed access
template <typename T>
concept Vector = Collection<T> && requires(const T& v, std::size_t i) {
    { v(i) } -> std::convertible_to<typename T::value_type>;
};

/// A dense vector with contiguous storage
template <typename T>
concept DenseVector = Vector<T>;

/// A column vector
template <typename T>
concept ColumnVector = Vector<T>;

/// A row vector
template <typename T>
concept RowVector = Vector<T>;

} // namespace mtl
