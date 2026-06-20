#pragma once
// MTL5 -- Approximate Minimum Degree (AMD) ordering for fill reduction
//
// AMD computes a fill-reducing permutation for the Cholesky factorization
// of a symmetric sparse matrix. It approximates the minimum degree ordering
// using the quotient graph technique, which avoids explicitly forming the
// elimination graph and runs in O(nnz) space.
//
// The algorithm greedily selects the node with the smallest approximate
// degree at each step, then absorbs it into the quotient graph. Approximate
// degrees are maintained cheaply via upper bounds.
//
// This is a simplified implementation suitable for small-to-medium matrices.
// For production use on large problems, consider interfacing with SuiteSparse
// AMD (Phase 6 external interfaces).
//
// Reference: Amestoy, Davis, Duff, "An Approximate Minimum Degree Ordering
//            Algorithm", SIAM J. Matrix Anal. Appl., 17(4), 1996.
//            Davis, "Direct Methods for Sparse Linear Systems", SIAM, Ch. 7.

#include <cassert>
#include <cstddef>
#include <vector>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/sparse/ordering/minimum_degree.hpp>

namespace mtl::sparse::ordering {

/// Approximate Minimum Degree ordering.
/// Callable object: given a symmetric sparse matrix, returns a permutation
/// vector that reduces fill-in during Cholesky factorization.
struct amd {

    /// Compute the AMD ordering for a symmetric sparse matrix.
    /// Returns permutation p where p[new] = old.
    ///
    /// Runs the near-linear quotient-graph minimum-degree algorithm
    /// (CSparse cs_amd) on the pattern of A + A^T.
    template <typename Value, typename Parameters>
    std::vector<std::size_t> operator()(
        const mat::compressed2D<Value, Parameters>& A) const
    {
        assert(A.num_rows() == A.num_cols());
        return detail::minimum_degree(/*order=*/1, A);
    }
};

} // namespace mtl::sparse::ordering
