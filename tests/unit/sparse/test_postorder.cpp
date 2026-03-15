#include <catch2/catch_test_macros.hpp>

#include <mtl/sparse/analysis/elimination_tree.hpp>
#include <mtl/sparse/analysis/postorder.hpp>

#include <algorithm>
#include <set>

using namespace mtl::sparse::analysis;

TEST_CASE("Postorder of path tree", "[sparse][postorder]") {
    // Tree: 0->1->2->3 (path)
    std::vector<std::size_t> parent = {1, 2, 3, no_parent};

    auto post = tree_postorder(parent);

    REQUIRE(post.size() == 4);
    // In a path, postorder visits leaf first: 0, 1, 2, 3
    REQUIRE(post[0] == 0);
    REQUIRE(post[1] == 1);
    REQUIRE(post[2] == 2);
    REQUIRE(post[3] == 3);
}

TEST_CASE("Postorder of binary tree", "[sparse][postorder]") {
    // Tree:     4
    //          / \
    //         2   3
    //        / \
    //       0   1
    // parent = [2, 2, 4, 4, no_parent]
    std::vector<std::size_t> parent = {2, 2, 4, 4, no_parent};

    auto post = tree_postorder(parent);

    REQUIRE(post.size() == 5);

    // Postorder property: all descendants come before their ancestor
    auto inv = inverse_postorder(post);

    // 0 and 1 must come before 2
    REQUIRE(inv[0] < inv[2]);
    REQUIRE(inv[1] < inv[2]);
    // 2 and 3 must come before 4
    REQUIRE(inv[2] < inv[4]);
    REQUIRE(inv[3] < inv[4]);
    // 4 must be last (root)
    REQUIRE(inv[4] == 4);
}

TEST_CASE("Postorder of forest (multiple roots)", "[sparse][postorder]") {
    // Two disconnected paths: 0->1, 2->3
    std::vector<std::size_t> parent = {1, no_parent, 3, no_parent};

    auto post = tree_postorder(parent);

    REQUIRE(post.size() == 4);

    auto inv = inverse_postorder(post);

    // 0 before 1, 2 before 3
    REQUIRE(inv[0] < inv[1]);
    REQUIRE(inv[2] < inv[3]);

    // All nodes appear exactly once
    std::set<std::size_t> seen(post.begin(), post.end());
    REQUIRE(seen.size() == 4);
}

TEST_CASE("Inverse postorder is inverse of postorder", "[sparse][postorder]") {
    std::vector<std::size_t> parent = {2, 2, 4, 4, no_parent};

    auto post = tree_postorder(parent);
    auto inv = inverse_postorder(post);

    for (std::size_t k = 0; k < post.size(); ++k)
        REQUIRE(inv[post[k]] == k);
}
