#pragma once
// MTL5 — Matrix concepts (replaces MTL4 concept/matrix.hpp)
#include <mtl/concepts/collection.hpp>
#include <cstddef>

namespace mtl {

/// A 2D collection with num_rows(), num_cols(), and (r,c) access
template <typename T>
concept Matrix = Collection<T> && requires(const T& m, std::size_t r, std::size_t c) {
    { m.num_rows() } -> std::convertible_to<typename T::size_type>;
    { m.num_cols() } -> std::convertible_to<typename T::size_type>;
    { m(r, c) }      -> std::convertible_to<typename T::value_type>;
};

/// A dense matrix with contiguous or strided storage
template <typename T>
concept DenseMatrix = Matrix<T>;

/// A sparse matrix
template <typename T>
concept SparseMatrix = Matrix<T>;

} // namespace mtl
