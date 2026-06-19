// MTL5 -- Tests for Block Triangular Form (Dulmage-Mendelsohn) decomposition.
// Covers maximum matching (step 1), Tarjan SCC + assembly (steps 2-3) of the
// native KLU plan (issue #114).
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/sparse/ordering/dulmage_mendelsohn.hpp>

using namespace mtl;
using mtl::sparse::ordering::block_triangular_form;
using mtl::sparse::ordering::maximum_matching;

namespace {

// Build a pattern matrix (values all 1.0) from a list of (row, col) entries.
mat::compressed2D<double> make_pattern(
    std::size_t n,
    const std::vector<std::pair<std::size_t, std::size_t>>& entries)
{
    mat::compressed2D<double> A(n, n);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        for (auto [r, c] : entries)
            ins[r][c] << 1.0;
    }
    return A;
}

bool is_permutation(const std::vector<std::size_t>& p, std::size_t n) {
    if (p.size() != n) return false;
    std::vector<char> seen(n, 0);
    for (auto v : p) {
        if (v >= n || seen[v]) return false;
        seen[v] = 1;
    }
    return true;
}

std::size_t matching_cardinality(const std::vector<std::ptrdiff_t>& m) {
    return static_cast<std::size_t>(
        std::count_if(m.begin(), m.end(), [](std::ptrdiff_t x) { return x >= 0; }));
}

// Returns true iff A(row_perm, col_perm) is block upper triangular w.r.t. the
// given block boundaries (every nonzero sits in a diagonal or upper block).
bool is_block_upper_triangular(const mat::compressed2D<double>& A,
                               const mtl::sparse::ordering::btf_result& btf)
{
    std::size_t n = A.num_rows();
    std::vector<std::size_t> inv_row(n), inv_col(n);
    for (std::size_t i = 0; i < n; ++i) {
        inv_row[btf.row_perm[i]] = i;
        inv_col[btf.col_perm[i]] = i;
    }
    // new index -> block id
    std::vector<std::size_t> block_of(n);
    for (std::size_t b = 0; b < btf.nblocks(); ++b)
        for (std::size_t k = btf.blocks[b]; k < btf.blocks[b + 1]; ++k)
            block_of[k] = b;

    const auto& rp = A.ref_major();
    const auto& ci = A.ref_minor();
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t k = rp[i]; k < rp[i + 1]; ++k) {
            std::size_t ni = inv_row[i];
            std::size_t nj = inv_col[ci[k]];
            if (block_of[ni] > block_of[nj]) return false;  // strictly lower block
        }
    }
    return true;
}

// Zero-free diagonal of the permuted matrix.
bool has_zero_free_diagonal(const mat::compressed2D<double>& A,
                            const mtl::sparse::ordering::btf_result& btf)
{
    for (std::size_t i = 0; i < A.num_rows(); ++i)
        if (A(btf.row_perm[i], btf.col_perm[i]) == 0.0) return false;
    return true;
}

} // namespace

TEST_CASE("maximum_matching on identity is the diagonal", "[sparse][btf][matching]") {
    auto A = make_pattern(4, {{0, 0}, {1, 1}, {2, 2}, {3, 3}});
    auto m = maximum_matching(A);
    REQUIRE(matching_cardinality(m) == 4);
    for (std::size_t i = 0; i < 4; ++i)
        REQUIRE(m[i] == static_cast<std::ptrdiff_t>(i));
}

TEST_CASE("maximum_matching finds a perfect matching on a permuted pattern",
          "[sparse][btf][matching]") {
    // A structurally nonsingular but with no nonzero on the natural diagonal.
    auto A = make_pattern(3, {{0, 1}, {0, 2}, {1, 0}, {2, 0}, {2, 2}});
    auto m = maximum_matching(A);
    REQUIRE(matching_cardinality(m) == 3);
    // Verify it is a valid matching: distinct columns, each an actual nonzero.
    std::vector<char> col_seen(3, 0);
    for (std::size_t i = 0; i < 3; ++i) {
        REQUIRE(m[i] >= 0);
        std::size_t c = static_cast<std::size_t>(m[i]);
        REQUIRE_FALSE(col_seen[c]);
        col_seen[c] = 1;
        REQUIRE(A(i, c) != 0.0);
    }
}

TEST_CASE("maximum_matching detects structural singularity",
          "[sparse][btf][matching]") {
    // Column 2 is empty -> no perfect matching, cardinality < n.
    auto A = make_pattern(3, {{0, 0}, {0, 1}, {1, 0}, {1, 1}, {2, 0}});
    auto m = maximum_matching(A);
    REQUIRE(matching_cardinality(m) < 3);
}

TEST_CASE("BTF splits a reducible matrix into blocks", "[sparse][btf]") {
    // Block upper triangular already: 2x2 diagonal blocks {0,1} and {2,3}
    // with an upper coupling A(0,3). Reducible -> two blocks.
    auto A = make_pattern(4, {
        {0, 0}, {0, 1}, {1, 0}, {1, 1},   // block {0,1}
        {2, 2}, {2, 3}, {3, 2}, {3, 3},   // block {2,3}
        {0, 3}                            // upper coupling
    });
    auto btf = block_triangular_form(A);

    REQUIRE_FALSE(btf.structurally_singular);
    REQUIRE(is_permutation(btf.row_perm, 4));
    REQUIRE(is_permutation(btf.col_perm, 4));
    REQUIRE(btf.nblocks() == 2);
    REQUIRE(has_zero_free_diagonal(A, btf));
    REQUIRE(is_block_upper_triangular(A, btf));
}

TEST_CASE("BTF yields a single block for an irreducible matrix", "[sparse][btf]") {
    // A directed cycle 0->1->2->3->0 (plus diagonal) is strongly connected.
    auto A = make_pattern(4, {
        {0, 0}, {1, 1}, {2, 2}, {3, 3},
        {0, 1}, {1, 2}, {2, 3}, {3, 0}
    });
    auto btf = block_triangular_form(A);

    REQUIRE_FALSE(btf.structurally_singular);
    REQUIRE(btf.nblocks() == 1);
    REQUIRE(is_permutation(btf.row_perm, 4));
    REQUIRE(is_block_upper_triangular(A, btf));  // trivially true for one block
}

TEST_CASE("BTF on a fully triangular matrix yields n singleton blocks",
          "[sparse][btf]") {
    // Lower-triangular pattern with full diagonal: each node its own SCC.
    auto A = make_pattern(4, {
        {0, 0},
        {1, 0}, {1, 1},
        {2, 1}, {2, 2},
        {3, 2}, {3, 3}
    });
    auto btf = block_triangular_form(A);

    REQUIRE_FALSE(btf.structurally_singular);
    REQUIRE(btf.nblocks() == 4);
    REQUIRE(has_zero_free_diagonal(A, btf));
    REQUIRE(is_block_upper_triangular(A, btf));
}

TEST_CASE("block_triangular_form rejects a non-square matrix",
          "[sparse][btf]") {
    mat::compressed2D<double> A(2, 3);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 1.0;
        ins[1][1] << 1.0;
    }
    REQUIRE_THROWS_AS(block_triangular_form(A), std::invalid_argument);
}

TEST_CASE("BTF flags a structurally singular matrix with valid permutations",
          "[sparse][btf]") {
    auto A = make_pattern(3, {{0, 0}, {0, 1}, {1, 0}, {1, 1}, {2, 0}});
    auto btf = block_triangular_form(A);

    REQUIRE(btf.structurally_singular);
    REQUIRE(is_permutation(btf.row_perm, 3));
    REQUIRE(is_permutation(btf.col_perm, 3));
}
