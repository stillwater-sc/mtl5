#pragma once
// MTL5 -- Postorder traversal of the elimination tree
// A postorder traversal is used to improve cache locality during
// numeric factorization: children are processed before their parent.
//
// Reference: Davis, "Direct Methods for Sparse Linear Systems", Ch. 4

#include <cassert>
#include <cstddef>
#include <vector>

#include <mtl/sparse/analysis/elimination_tree.hpp>

namespace mtl::sparse::analysis {

/// Compute a depth-first postorder of the elimination tree.
/// Returns a permutation vector post where post[k] = node visited at step k.
/// The postorder ensures that for every node j, all descendants of j
/// appear before j in the ordering.
inline std::vector<std::size_t> tree_postorder(
    const std::vector<std::size_t>& parent)
{
    std::size_t n = parent.size();

    // Build child lists (first-child / next-sibling representation)
    std::vector<std::size_t> first_child(n, no_parent);
    std::vector<std::size_t> next_sibling(n, no_parent);

    // Traverse in reverse to get children in natural order
    for (std::size_t j = n; j > 0; --j) {
        std::size_t node = j - 1;
        std::size_t p = parent[node];
        if (p != no_parent) {
            next_sibling[node] = first_child[p];
            first_child[p] = node;
        }
    }

    // Iterative DFS postorder using an explicit stack
    std::vector<std::size_t> post;
    post.reserve(n);

    std::vector<std::size_t> stack;
    stack.reserve(n);

    // Push all roots (nodes with no parent)
    for (std::size_t j = n; j > 0; --j) {
        if (parent[j - 1] == no_parent)
            stack.push_back(j - 1);
    }

    // Use a visited marker to distinguish first visit from post-visit
    std::vector<bool> expanded(n, false);

    while (!stack.empty()) {
        std::size_t node = stack.back();

        if (!expanded[node]) {
            expanded[node] = true;
            // Push children in reverse order so leftmost is processed first
            std::vector<std::size_t> children;
            std::size_t child = first_child[node];
            while (child != no_parent) {
                children.push_back(child);
                child = next_sibling[child];
            }
            if (!children.empty()) {
                for (auto it = children.rbegin(); it != children.rend(); ++it)
                    stack.push_back(*it);
                continue;  // process children first
            }
        }

        // Post-visit: all children have been processed
        stack.pop_back();
        post.push_back(node);
    }

    return post;
}

/// Given a postorder, return the inverse postorder mapping:
/// inv_post[node] = position of node in the postorder.
inline std::vector<std::size_t> inverse_postorder(
    const std::vector<std::size_t>& post)
{
    std::size_t n = post.size();
    std::vector<std::size_t> inv(n);
    for (std::size_t k = 0; k < n; ++k)
        inv[post[k]] = k;
    return inv;
}

} // namespace mtl::sparse::analysis
