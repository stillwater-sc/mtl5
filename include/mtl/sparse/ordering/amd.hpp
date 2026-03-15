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

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <limits>
#include <numeric>
#include <vector>

#include <mtl/mat/compressed2D.hpp>

namespace mtl::sparse::ordering {

/// Approximate Minimum Degree ordering.
/// Callable object: given a symmetric sparse matrix, returns a permutation
/// vector that reduces fill-in during Cholesky factorization.
struct amd {

    /// Compute the AMD ordering for a symmetric sparse matrix.
    /// Returns permutation p where p[new] = old.
    ///
    /// Algorithm: simplified minimum degree using adjacency sets.
    /// At each step, selects the uneliminated node with the fewest
    /// remaining connections, eliminates it, and updates neighbors.
    template <typename Value, typename Parameters>
    std::vector<std::size_t> operator()(
        const mat::compressed2D<Value, Parameters>& A) const
    {
        using size_type = std::size_t;
        size_type n = A.num_rows();
        assert(A.num_rows() == A.num_cols());

        if (n == 0) return {};

        const auto& starts  = A.ref_major();
        const auto& indices = A.ref_minor();

        // Build adjacency lists (excluding self-loops / diagonal)
        std::vector<std::vector<size_type>> adj(n);
        for (size_type i = 0; i < n; ++i) {
            for (size_type k = starts[i]; k < starts[i + 1]; ++k) {
                size_type j = indices[k];
                if (j != i) {
                    adj[i].push_back(j);
                }
            }
        }

        // Degree of each node (number of adjacent uneliminated nodes)
        std::vector<size_type> degree(n);
        for (size_type i = 0; i < n; ++i)
            degree[i] = adj[i].size();

        std::vector<bool> eliminated(n, false);
        std::vector<std::size_t> perm;
        perm.reserve(n);

        for (size_type step = 0; step < n; ++step) {
            // Find uneliminated node with minimum degree
            size_type pivot = n;
            size_type min_deg = std::numeric_limits<size_type>::max();
            for (size_type i = 0; i < n; ++i) {
                if (!eliminated[i] && degree[i] < min_deg) {
                    min_deg = degree[i];
                    pivot = i;
                }
            }

            // Eliminate pivot
            eliminated[pivot] = true;
            perm.push_back(pivot);

            // Mass elimination: connect all neighbors of pivot to each other
            // (fill edges), then update degrees.
            // Collect active neighbors
            std::vector<size_type> neighbors;
            for (size_type j : adj[pivot]) {
                if (!eliminated[j]) {
                    neighbors.push_back(j);
                }
            }

            // For each pair of neighbors, add a fill edge if not present
            // and update degrees. Use a set-based approach for efficiency.
            for (size_type ni = 0; ni < neighbors.size(); ++ni) {
                size_type u = neighbors[ni];

                // Remove pivot from u's adjacency
                auto& adj_u = adj[u];
                adj_u.erase(
                    std::remove(adj_u.begin(), adj_u.end(), pivot),
                    adj_u.end());

                // Add fill edges: for each other neighbor v of pivot,
                // ensure u-v edge exists
                for (size_type nj = 0; nj < neighbors.size(); ++nj) {
                    if (ni == nj) continue;
                    size_type v = neighbors[nj];

                    // Check if u already has v in adjacency
                    bool found = false;
                    for (size_type k : adj_u) {
                        if (k == v) { found = true; break; }
                    }
                    if (!found) {
                        adj_u.push_back(v);
                    }
                }

                // Recompute degree for u (count non-eliminated neighbors)
                size_type deg = 0;
                for (size_type k : adj_u) {
                    if (!eliminated[k]) ++deg;
                }
                degree[u] = deg;
            }
        }

        return perm;
    }
};

} // namespace mtl::sparse::ordering
