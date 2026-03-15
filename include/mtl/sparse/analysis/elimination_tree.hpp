#pragma once
// MTL5 -- Elimination tree construction for sparse direct solvers
// The elimination tree is the fundamental data structure for sparse
// factorization, encoding column dependencies during Gaussian elimination.
//
// For a symmetric matrix A (or A^T*A for unsymmetric), the parent of
// column j in the elimination tree is the row index of the first
// subdiagonal nonzero in column j of the Cholesky factor L.
//
// Reference: Liu, "The Role of Elimination Trees in Sparse Factorization",
//            SIAM J. Matrix Anal. Appl., 11(1), 1990.

#include <cassert>
#include <cstddef>
#include <limits>
#include <vector>

#include <mtl/sparse/util/csc.hpp>

namespace mtl::sparse::analysis {

/// Sentinel value indicating no parent (root of the elimination tree).
inline constexpr std::size_t no_parent = std::numeric_limits<std::size_t>::max();

/// Compute the elimination tree of a symmetric matrix given in CSC format.
/// Only the upper triangular part is accessed.
/// Returns parent vector where parent[j] = first subdiagonal nonzero row in
/// column j of L, or no_parent if j is a root.
///
/// Algorithm: Liu's O(nnz) algorithm using path compression (union-find).
/// Reference: CSparse cs_etree.
template <typename Value, typename SizeType>
std::vector<std::size_t> elimination_tree(
    const util::csc_matrix<Value, SizeType>& A)
{
    SizeType n = A.ncols;
    assert(A.nrows == A.ncols);

    std::vector<std::size_t> parent(n, no_parent);
    std::vector<std::size_t> ancestor(n, no_parent);  // for path compression

    for (SizeType k = 0; k < n; ++k) {
        // Process column k: for each row index i < k in column k
        for (SizeType p = A.col_ptr[k]; p < A.col_ptr[k + 1]; ++p) {
            SizeType i = A.row_ind[p];
            if (i >= k) continue;  // only upper triangular part

            // Follow path from i to root of its subtree, attaching to k
            SizeType node = i;
            SizeType next;
            while (node != no_parent && node != k) {
                next = ancestor[node];
                ancestor[node] = k;  // path compression
                if (next == no_parent) {
                    parent[node] = k;  // attach subtree rooted at node to k
                }
                node = next;
            }
        }
    }

    return parent;
}

/// Overload: compute elimination tree from a CRS compressed2D matrix.
/// Converts to CSC internally (symmetric matrices are their own transpose).
template <typename Value, typename Parameters>
std::vector<std::size_t> elimination_tree(
    const mat::compressed2D<Value, Parameters>& A)
{
    auto csc = util::crs_to_csc(A);
    return elimination_tree(csc);
}

/// Compute the column counts of the Cholesky factor L using the elimination tree.
/// col_counts[j] = number of nonzeros in column j of L (including the diagonal).
/// Uses the "skeleton matrix" approach with the etree.
///
/// Reference: Gilbert, Ng, Peyton, "An efficient algorithm to compute row and
///            column counts for sparse Cholesky factorization".
template <typename Value, typename SizeType>
std::vector<std::size_t> column_counts(
    const util::csc_matrix<Value, SizeType>& A,
    const std::vector<std::size_t>& parent)
{
    SizeType n = A.ncols;
    assert(parent.size() == n);

    // First-descendant array and level array via postorder
    // Simplified approach: count nonzeros in each column of L by
    // walking up the elimination tree from each nonzero.
    std::vector<std::size_t> counts(n, 0);
    std::vector<std::size_t> prev_leaf(n, no_parent);  // for leaf detection
    std::vector<std::size_t> first(n, no_parent);       // first descendant

    // Initialize: each column has at least the diagonal
    for (SizeType j = 0; j < n; ++j)
        counts[j] = 1;

    // For each column k, examine each row i < k in the upper triangle.
    // Walk from i up the etree to find the least common ancestor with
    // previous entries in the same subtree.
    for (SizeType k = 0; k < n; ++k) {
        for (SizeType p = A.col_ptr[k]; p < A.col_ptr[k + 1]; ++p) {
            SizeType i = A.row_ind[p];
            if (i >= k) continue;

            // Walk from i up to k in the etree, counting new entries
            SizeType node = i;
            while (node != no_parent && node < k) {
                SizeType old_first = first[node];
                first[node] = k;
                if (old_first == k) break;  // already counted this path
                counts[node]++;  // this path contributes a fill entry
                node = parent[node];
            }
        }
    }

    return counts;
}

/// Count the total predicted nonzeros in L given column counts.
inline std::size_t total_nnz(const std::vector<std::size_t>& col_counts) {
    std::size_t total = 0;
    for (auto c : col_counts) total += c;
    return total;
}

} // namespace mtl::sparse::analysis
