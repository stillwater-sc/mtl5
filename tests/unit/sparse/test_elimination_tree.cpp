#include <catch2/catch_test_macros.hpp>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/sparse/util/csc.hpp>
#include <mtl/sparse/analysis/elimination_tree.hpp>

using namespace mtl;
using namespace mtl::sparse;

TEST_CASE("Elimination tree of tridiagonal matrix", "[sparse][etree]") {
    // Tridiagonal 4x4 matrix (symmetric):
    // [2 1 0 0]
    // [1 2 1 0]
    // [0 1 2 1]
    // [0 0 1 2]
    // The etree should be a path: 0->1->2->3
    mat::compressed2D<double> A(4, 4);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        for (std::size_t i = 0; i < 4; ++i) {
            ins[i][i] << 2.0;
            if (i + 1 < 4) {
                ins[i][i + 1] << 1.0;
                ins[i + 1][i] << 1.0;
            }
        }
    }

    auto csc = util::crs_to_csc(A);
    auto parent = analysis::elimination_tree(csc);

    REQUIRE(parent.size() == 4);
    REQUIRE(parent[0] == 1);
    REQUIRE(parent[1] == 2);
    REQUIRE(parent[2] == 3);
    REQUIRE(parent[3] == analysis::no_parent);
}

TEST_CASE("Elimination tree of arrow matrix", "[sparse][etree]") {
    // Arrow matrix 4x4:
    // [1 1 1 1]
    // [1 2 0 0]
    // [1 0 3 0]
    // [1 0 0 4]
    // The etree: parent[0] = no_parent (row 0 connects to all),
    // but since we process upper triangle: col 0 has entries at rows 0,
    // cols 1,2,3 have entry at row 0.
    // etree: parent[1]=0? No — etree is: parent[j] = first subdiag nz in col j.
    // For upper triangular: column 0 has row 0 only (diag), so we look at
    // which columns have row index 0 in upper triangle.
    // Upper triangle entries: (0,0),(0,1),(0,2),(0,3),(1,1),(2,2),(3,3)
    // Column 0: row 0 (diag only) -> parent[0] = no_parent? No, col 0 has
    // off-diag entries in col 1,2,3 with row 0.
    // Actually, etree processes: for each col k, for each row i < k in col k:
    //   walk from i to k.
    // Col 1: row 0 < 1, so parent[0] = 1
    // Col 2: row 0 < 2, walk from 0: ancestor[0]=2, parent[0] already set,
    //   but 0 already has parent 1, so we walk: 0->1 (ancestor[0]=1 from col 1).
    //   Wait, let me re-trace with path compression.
    //
    // Actually for an arrow matrix with dense first row/col, after Cholesky
    // the etree is a path 0->1->2->3. Let me just verify the structure.

    mat::compressed2D<double> A(4, 4);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 10.0; ins[0][1] << 1.0; ins[0][2] << 1.0; ins[0][3] << 1.0;
        ins[1][0] << 1.0;  ins[1][1] << 2.0;
        ins[2][0] << 1.0;  ins[2][2] << 3.0;
        ins[3][0] << 1.0;  ins[3][3] << 4.0;
    }

    auto csc = util::crs_to_csc(A);
    auto parent = analysis::elimination_tree(csc);

    REQUIRE(parent.size() == 4);
    // Column 1 upper has row 0, so parent[0] gets set.
    // Column 2 upper has row 0, walk from 0 through ancestor chain.
    // Column 3 upper has row 0, walk from 0 through ancestor chain.
    // For arrow: parent = [1, 2, 3, no_parent] (chain through fill-in)
    REQUIRE(parent[0] == 1);
    REQUIRE(parent[1] == 2);
    REQUIRE(parent[2] == 3);
    REQUIRE(parent[3] == analysis::no_parent);
}

TEST_CASE("Elimination tree of diagonal matrix", "[sparse][etree]") {
    // Diagonal matrix: no off-diagonal entries, all nodes are roots
    mat::compressed2D<double> A(3, 3);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 1.0;
        ins[1][1] << 2.0;
        ins[2][2] << 3.0;
    }

    auto csc = util::crs_to_csc(A);
    auto parent = analysis::elimination_tree(csc);

    REQUIRE(parent[0] == analysis::no_parent);
    REQUIRE(parent[1] == analysis::no_parent);
    REQUIRE(parent[2] == analysis::no_parent);
}

TEST_CASE("Elimination tree from CRS overload", "[sparse][etree]") {
    mat::compressed2D<double> A(3, 3);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 2.0; ins[0][1] << 1.0;
        ins[1][0] << 1.0; ins[1][1] << 2.0; ins[1][2] << 1.0;
        ins[2][1] << 1.0; ins[2][2] << 2.0;
    }

    // Should work directly from CRS
    auto parent = analysis::elimination_tree(A);
    REQUIRE(parent.size() == 3);
    REQUIRE(parent[0] == 1);
    REQUIRE(parent[1] == 2);
    REQUIRE(parent[2] == analysis::no_parent);
}

TEST_CASE("Column counts for tridiagonal matrix", "[sparse][etree][colcounts]") {
    mat::compressed2D<double> A(4, 4);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        for (std::size_t i = 0; i < 4; ++i) {
            ins[i][i] << 2.0;
            if (i + 1 < 4) {
                ins[i][i + 1] << 1.0;
                ins[i + 1][i] << 1.0;
            }
        }
    }

    auto csc = util::crs_to_csc(A);
    auto parent = analysis::elimination_tree(csc);
    auto counts = analysis::column_counts(csc, parent);

    // Tridiagonal: Cholesky L has 2 entries per column (diag + one subdiag)
    // except the last column which has only the diagonal.
    // Actually for tridiagonal, L is bidiagonal: each column has the diagonal
    // and one entry below it, except column n-1.
    REQUIRE(counts[0] == 2);  // diag + one below
    REQUIRE(counts[1] == 2);
    REQUIRE(counts[2] == 2);
    REQUIRE(counts[3] == 1);  // just diagonal

    REQUIRE(analysis::total_nnz(counts) == 7);
}
