#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cstddef>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/sparse/util/csc.hpp>
#include <mtl/sparse/util/permutation.hpp>
#include <mtl/sparse/util/scatter.hpp>
#include <mtl/sparse/analysis/elimination_tree.hpp>
#include <mtl/sparse/analysis/postorder.hpp>
#include <mtl/sparse/factorization/triangular_solve.hpp>

using namespace mtl;
using namespace mtl::sparse;

// ---- CSC edge cases ----

TEST_CASE("CSC conversion of empty matrix", "[sparse][csc][edge]") {
    mat::compressed2D<double> A(0, 0);
    auto csc = util::crs_to_csc(A);
    REQUIRE(csc.nrows == 0);
    REQUIRE(csc.ncols == 0);
    REQUIRE(csc.nnz() == 0);
}

TEST_CASE("CSC conversion of 1x1 matrix", "[sparse][csc][edge]") {
    mat::compressed2D<double> A(1, 1);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 7.0;
    }
    auto csc = util::crs_to_csc(A);
    REQUIRE(csc.nrows == 1);
    REQUIRE(csc.ncols == 1);
    REQUIRE(csc.nnz() == 1);
    REQUIRE(csc(0, 0) == 7.0);
}

TEST_CASE("CSC conversion of empty sparse matrix (allocated but no entries)", "[sparse][csc][edge]") {
    mat::compressed2D<double> A(5, 5);
    // No entries inserted
    auto csc = util::crs_to_csc(A);
    REQUIRE(csc.nrows == 5);
    REQUIRE(csc.ncols == 5);
    REQUIRE(csc.nnz() == 0);
}

TEST_CASE("CSC round-trip on rectangular 5x3", "[sparse][csc][edge]") {
    mat::compressed2D<double> A(5, 3);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 1.0;
        ins[1][1] << 2.0;
        ins[2][2] << 3.0;
        ins[3][0] << 4.0; ins[3][2] << 5.0;
        ins[4][1] << 6.0;
    }
    auto csc = util::crs_to_csc(A);
    auto B = util::csc_to_crs(csc);
    REQUIRE(B.num_rows() == 5);
    REQUIRE(B.num_cols() == 3);
    for (std::size_t i = 0; i < 5; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            REQUIRE(B(i, j) == A(i, j));
}

TEST_CASE("CSC round-trip on rectangular 3x5", "[sparse][csc][edge]") {
    mat::compressed2D<double> A(3, 5);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 1.0; ins[0][4] << 2.0;
        ins[1][2] << 3.0;
        ins[2][1] << 4.0; ins[2][3] << 5.0;
    }
    auto csc = util::crs_to_csc(A);
    auto B = util::csc_to_crs(csc);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 5; ++j)
            REQUIRE(B(i, j) == A(i, j));
}

// ---- Elimination tree edge cases ----

TEST_CASE("Elimination tree of 1x1 matrix", "[sparse][etree][edge]") {
    mat::compressed2D<double> A(1, 1);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 1.0;
    }
    auto parent = analysis::elimination_tree(A);
    REQUIRE(parent.size() == 1);
    REQUIRE(parent[0] == analysis::no_parent);
}

TEST_CASE("Elimination tree of block diagonal", "[sparse][etree][edge]") {
    // Two disconnected 2x2 blocks
    mat::compressed2D<double> A(4, 4);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 2.0; ins[0][1] << 1.0;
        ins[1][0] << 1.0; ins[1][1] << 2.0;
        ins[2][2] << 3.0; ins[2][3] << 1.0;
        ins[3][2] << 1.0; ins[3][3] << 3.0;
    }
    auto csc = util::crs_to_csc(A);
    auto parent = analysis::elimination_tree(csc);

    // Two separate subtrees: 0->1 and 2->3
    REQUIRE(parent[0] == 1);
    REQUIRE(parent[1] == analysis::no_parent);
    REQUIRE(parent[2] == 3);
    REQUIRE(parent[3] == analysis::no_parent);
}

TEST_CASE("Column counts for arrow matrix", "[sparse][etree][edge]") {
    // Arrow: node 0 connects to all
    std::size_t n = 5;
    mat::compressed2D<double> A(n, n);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 10.0;
        for (std::size_t i = 1; i < n; ++i) {
            ins[0][i] << 1.0;
            ins[i][0] << 1.0;
            ins[i][i] << 5.0;
        }
    }
    auto csc = util::crs_to_csc(A);
    auto parent = analysis::elimination_tree(csc);
    auto counts = analysis::column_counts(csc, parent);

    // Total nnz should be >= n (at least the diagonal)
    std::size_t total = analysis::total_nnz(counts);
    REQUIRE(total >= n);
}

// ---- Postorder edge cases ----

TEST_CASE("Postorder of single node", "[sparse][postorder][edge]") {
    std::vector<std::size_t> parent = {analysis::no_parent};
    auto post = analysis::tree_postorder(parent);
    REQUIRE(post.size() == 1);
    REQUIRE(post[0] == 0);
}

TEST_CASE("Postorder of long chain (50 nodes)", "[sparse][postorder][edge]") {
    std::size_t n = 50;
    std::vector<std::size_t> parent(n);
    for (std::size_t i = 0; i + 1 < n; ++i) parent[i] = i + 1;
    parent[n - 1] = analysis::no_parent;

    auto post = analysis::tree_postorder(parent);
    REQUIRE(post.size() == n);

    // Postorder of a chain: 0, 1, 2, ..., n-1
    for (std::size_t i = 0; i < n; ++i)
        REQUIRE(post[i] == i);
}

// ---- Permutation edge cases ----

TEST_CASE("Permutation on rectangular 4x3 column permute", "[sparse][perm][edge]") {
    mat::compressed2D<double> A(4, 3);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 1.0; ins[0][2] << 2.0;
        ins[1][1] << 3.0;
        ins[2][0] << 4.0;
        ins[3][1] << 5.0; ins[3][2] << 6.0;
    }
    // Swap columns 0 and 2: perm = [2, 1, 0]
    std::vector<std::size_t> perm = {2, 1, 0};
    auto B = util::column_permute(A, perm);

    // B(:, pinv[old]) = A(:, old)
    // pinv = [2, 1, 0], so B(:,2)=A(:,0), B(:,1)=A(:,1), B(:,0)=A(:,2)
    REQUIRE(B(0, 2) == 1.0);  // A(0,0)
    REQUIRE(B(0, 0) == 2.0);  // A(0,2)
    REQUIRE(B(1, 1) == 3.0);  // A(1,1)
    REQUIRE(B(2, 2) == 4.0);  // A(2,0)
    REQUIRE(B(3, 1) == 5.0);  // A(3,1)
    REQUIRE(B(3, 0) == 6.0);  // A(3,2)
}

TEST_CASE("Permutation on rectangular 3x5 row permute", "[sparse][perm][edge]") {
    mat::compressed2D<double> A(3, 5);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 1.0;
        ins[1][2] << 2.0;
        ins[2][4] << 3.0;
    }
    // Reverse rows: perm = [2, 1, 0]
    std::vector<std::size_t> perm = {2, 1, 0};
    auto B = util::row_permute(A, perm);

    // B(pinv[old], :) = A(old, :)
    // pinv = [2, 1, 0], so B(2,:)=A(0,:), B(1,:)=A(1,:), B(0,:)=A(2,:)
    REQUIRE(B(2, 0) == 1.0);
    REQUIRE(B(1, 2) == 2.0);
    REQUIRE(B(0, 4) == 3.0);
}

// ---- Scatter edge cases ----

TEST_CASE("Scatter with many clear cycles", "[sparse][scatter][edge]") {
    util::sparse_accumulator<double> acc(10);
    for (int cycle = 0; cycle < 20; ++cycle) {
        acc.clear();
        acc.scatter(cycle % 10, static_cast<double>(cycle));
        REQUIRE(acc.nnz() == 1);
        REQUIRE(acc(cycle % 10) == static_cast<double>(cycle));
    }
}

TEST_CASE("Scatter with large accumulator", "[sparse][scatter][edge]") {
    std::size_t n = 1000;
    util::sparse_accumulator<double> acc(n);
    // Scatter every 10th entry
    for (std::size_t i = 0; i < n; i += 10)
        acc.scatter(i, static_cast<double>(i));
    REQUIRE(acc.nnz() == 100);
    REQUIRE(acc(0) == 0.0);
    REQUIRE(acc(990) == 990.0);
    REQUIRE(acc(5) == 0.0);  // not set
}

// ---- Triangular solve edge cases ----

TEST_CASE("Lower triangular solve with identity matrix", "[sparse][trisolve][edge]") {
    mat::compressed2D<double> I(3, 3);
    {
        mat::inserter<mat::compressed2D<double>> ins(I);
        ins[0][0] << 1.0;
        ins[1][1] << 1.0;
        ins[2][2] << 1.0;
    }
    auto L = util::crs_to_csc(I);
    std::vector<double> x = {7.0, 8.0, 9.0};
    factorization::dense_lower_solve(L, x);
    REQUIRE(x[0] == 7.0);
    REQUIRE(x[1] == 8.0);
    REQUIRE(x[2] == 9.0);
}

TEST_CASE("Upper triangular solve with diagonal matrix", "[sparse][trisolve][edge]") {
    mat::compressed2D<double> D(3, 3);
    {
        mat::inserter<mat::compressed2D<double>> ins(D);
        ins[0][0] << 2.0;
        ins[1][1] << 3.0;
        ins[2][2] << 4.0;
    }
    auto U = util::crs_to_csc(D);
    std::vector<double> x = {6.0, 12.0, 20.0};
    factorization::dense_upper_solve(U, x);
    REQUIRE_THAT(x[0], Catch::Matchers::WithinAbs(3.0, 1e-12));
    REQUIRE_THAT(x[1], Catch::Matchers::WithinAbs(4.0, 1e-12));
    REQUIRE_THAT(x[2], Catch::Matchers::WithinAbs(5.0, 1e-12));
}
