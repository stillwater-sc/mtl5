#pragma once
// MTL5 -- Reverse Cuthill-McKee (RCM) ordering for bandwidth reduction
// A BFS-based heuristic that reduces the bandwidth of a sparse matrix.
// While not optimal for fill reduction (AMD/COLAMD are better), RCM
// is simple to implement and useful as a baseline.
//
// Reference: Cuthill & McKee, "Reducing the bandwidth of sparse symmetric
//            matrices", Proc. 24th Natl. Conf. ACM, 1969.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <queue>
#include <vector>

#include <mtl/mat/compressed2D.hpp>

namespace mtl::sparse::ordering {

/// Reverse Cuthill-McKee ordering.
/// Callable object: given a symmetric sparse matrix, returns a permutation
/// vector that reduces bandwidth.
struct rcm {

    /// Compute the RCM ordering for a symmetric sparse matrix.
    /// Returns permutation p where p[new] = old.
    template <typename Value, typename Parameters>
    std::vector<std::size_t> operator()(
        const mat::compressed2D<Value, Parameters>& A) const
    {
        using size_type = typename mat::compressed2D<Value, Parameters>::size_type;
        size_type n = A.num_rows();
        assert(A.num_rows() == A.num_cols());

        const auto& starts  = A.ref_major();
        const auto& indices = A.ref_minor();

        // Compute degree of each node
        std::vector<size_type> degree(n);
        for (size_type i = 0; i < n; ++i)
            degree[i] = starts[i + 1] - starts[i];

        std::vector<bool> visited(n, false);
        std::vector<std::size_t> order;
        order.reserve(n);

        // Process each connected component
        while (order.size() < n) {
            // Find unvisited node with minimum degree (pseudo-peripheral start)
            size_type start = 0;
            size_type min_deg = n + 1;
            for (size_type i = 0; i < n; ++i) {
                if (!visited[i] && degree[i] < min_deg) {
                    min_deg = degree[i];
                    start = i;
                }
            }

            // BFS from start, ordering neighbors by increasing degree
            std::queue<size_type> bfs;
            bfs.push(start);
            visited[start] = true;

            while (!bfs.empty()) {
                size_type node = bfs.front();
                bfs.pop();
                order.push_back(node);

                // Collect unvisited neighbors
                std::vector<size_type> neighbors;
                for (size_type k = starts[node]; k < starts[node + 1]; ++k) {
                    size_type nb = indices[k];
                    if (!visited[nb]) {
                        neighbors.push_back(nb);
                    }
                }

                // Sort by increasing degree
                std::sort(neighbors.begin(), neighbors.end(),
                    [&degree](size_type a, size_type b) {
                        return degree[a] < degree[b];
                    });

                for (size_type nb : neighbors) {
                    if (!visited[nb]) {
                        visited[nb] = true;
                        bfs.push(nb);
                    }
                }
            }
        }

        // Reverse the order (Cuthill-McKee -> Reverse Cuthill-McKee)
        std::reverse(order.begin(), order.end());

        return order;
    }
};

} // namespace mtl::sparse::ordering
