#pragma once
// MTL5 -- Matrix concepts (replaces MTL4 concept/matrix.hpp)
#include <mtl/concepts/collection.hpp>
#include <mtl/traits/category.hpp>
#include <mtl/tag/sparsity.hpp>
#include <cstddef>
#include <type_traits>

namespace mtl {

/// A 2D collection with num_rows(), num_cols(), and (r,c) access
template <typename T>
concept Matrix = Collection<T> && requires(const T& m, std::size_t r, std::size_t c) {
    { m.num_rows() } -> std::convertible_to<typename T::size_type>;
    { m.num_cols() } -> std::convertible_to<typename T::size_type>;
    { m(r, c) }      -> std::convertible_to<typename T::value_type>;
};

/// A dense matrix: category tag is tag::dense
template <typename T>
concept DenseMatrix = Matrix<T> && std::is_same_v<traits::category_t<T>, tag::dense>;

/// A sparse matrix: category tag is tag::sparse
template <typename T>
concept SparseMatrix = Matrix<T> && std::is_same_v<traits::category_t<T>, tag::sparse>;

} // namespace mtl
