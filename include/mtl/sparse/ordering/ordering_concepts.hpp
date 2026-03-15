#pragma once
// MTL5 -- Concepts for fill-reducing orderings
// A fill-reducing ordering is a callable that takes a sparse matrix and
// returns a permutation vector that reduces fill-in during factorization.

#include <concepts>
#include <cstddef>
#include <vector>

#include <mtl/mat/compressed2D.hpp>

namespace mtl::sparse {

/// Concept for a fill-reducing ordering algorithm.
/// An ordering is a callable that takes a sparse matrix and returns
/// a permutation vector p where p[new_index] = old_index.
template <typename O, typename Matrix>
concept FillReducingOrdering = requires(const O& ord, const Matrix& A) {
    { ord(A) } -> std::convertible_to<std::vector<std::size_t>>;
};

/// Concept for sparse direct solver results.
/// A solver must support solving Ax = b and provide dimensions.
template <typename F, typename VecX, typename VecB>
concept SparseDirectSolver = requires(const F& f, VecX& x, const VecB& b) {
    { f.solve(x, b) };
    { f.num_rows() } -> std::convertible_to<std::size_t>;
    { f.num_cols() } -> std::convertible_to<std::size_t>;
};

} // namespace mtl::sparse
