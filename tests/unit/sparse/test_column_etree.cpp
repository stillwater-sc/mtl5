#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <random>
#include <set>
#include <stdexcept>
#include <vector>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/sparse/util/csc.hpp>
#include <mtl/sparse/analysis/elimination_tree.hpp>
#include <mtl/sparse/analysis/postorder.hpp>
#include <mtl/sparse/analysis/column_etree.hpp>
#include <mtl/sparse/factorization/sparse_lu.hpp>

using namespace mtl;
using namespace mtl::sparse;
using analysis::no_parent;

// ---------------------------------------------------------------------------
// Brute-force oracle: build the symmetric pattern of A^T A explicitly, then run
// the existing symmetric etree / column_counts on it. The near-linear
// column_elimination_tree / column_counts_ata (which never form A^T A) must
// match this on every matrix.
// ---------------------------------------------------------------------------
static util::csc_matrix<double> ata_pattern(const mat::compressed2D<double>& A) {
    std::size_t n = A.num_cols();
    const auto& rp = A.ref_major();
    const auto& ci = A.ref_minor();
    std::vector<std::set<std::size_t>> cols(n);
    for (std::size_t r = 0; r < A.num_rows(); ++r) {
        for (std::size_t p = rp[r]; p < rp[r + 1]; ++p)
            for (std::size_t q = rp[r]; q < rp[r + 1]; ++q)
                cols[ci[p]].insert(ci[q]);   // columns sharing row r are adjacent in A^T A
    }
    util::csc_matrix<double> S;
    S.nrows = n; S.ncols = n;
    S.col_ptr.assign(n + 1, 0);
    for (std::size_t j = 0; j < n; ++j) S.col_ptr[j + 1] = S.col_ptr[j] + cols[j].size();
    S.row_ind.resize(S.col_ptr[n]);
    S.values.assign(S.col_ptr[n], 1.0);
    std::size_t pos = 0;
    for (std::size_t j = 0; j < n; ++j)
        for (std::size_t i : cols[j]) S.row_ind[pos++] = i;   // std::set is sorted
    return S;
}

// Random matrix with a strong diagonal (structurally nonsingular for LU).
static mat::compressed2D<double> random_unsym(std::size_t n, double density,
                                              unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::uniform_real_distribution<double> val(-1.0, 1.0);
    mat::compressed2D<double> A(n, n);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        for (std::size_t i = 0; i < n; ++i) {
            ins[i][i] << static_cast<double>(n) + 1.0;   // diagonal dominance
            for (std::size_t j = 0; j < n; ++j)
                if (i != j && u(rng) < density) ins[i][j] << val(rng);
        }
    }
    return A;
}

TEST_CASE("Column etree matches symmetric etree of A^T A", "[sparse][col_etree]") {
    std::vector<mat::compressed2D<double>> mats;
    mats.push_back(random_unsym(20, 0.15, 1));
    mats.push_back(random_unsym(40, 0.08, 2));
    mats.push_back(random_unsym(30, 0.30, 3));
    mats.push_back(random_unsym(15, 0.50, 4));

    for (auto& A : mats) {
        auto S = ata_pattern(A);
        auto oracle_parent = analysis::elimination_tree(S);             // #178 symmetric
        auto parent = analysis::column_elimination_tree(util::crs_to_csc(A));
        REQUIRE(parent == oracle_parent);

        // Column counts: same etree -> same counts as symmetric chol(A^T A).
        auto post = analysis::tree_postorder(parent);
        auto oracle_counts = analysis::column_counts(S, oracle_parent);
        auto counts = analysis::column_counts_ata(util::crs_to_csc(A), parent, post);
        REQUIRE(counts == oracle_counts);
    }
}

TEST_CASE("Column etree: hand-checked small cases", "[sparse][col_etree]") {
    SECTION("diagonal -> all roots") {
        mat::compressed2D<double> A(4, 4);
        { mat::inserter<mat::compressed2D<double>> ins(A);
          for (std::size_t i = 0; i < 4; ++i) ins[i][i] << 2.0; }
        auto parent = analysis::column_elimination_tree(util::crs_to_csc(A));
        for (std::size_t j = 0; j < 4; ++j) REQUIRE(parent[j] == no_parent);
    }
    SECTION("lower bidiagonal -> A^T A tridiagonal -> path etree") {
        // A(i,i)=1, A(i+1,i)=1. Columns i and i+1 share row i+1 => A^T A
        // tridiagonal => column etree is the chain 0->1->2->3.
        std::size_t n = 5;
        mat::compressed2D<double> A(n, n);
        { mat::inserter<mat::compressed2D<double>> ins(A);
          for (std::size_t i = 0; i < n; ++i) {
              ins[i][i] << 1.0;
              if (i + 1 < n) ins[i + 1][i] << 1.0;
          } }
        auto parent = analysis::column_elimination_tree(util::crs_to_csc(A));
        for (std::size_t j = 0; j + 1 < n; ++j) REQUIRE(parent[j] == j + 1);
        REQUIRE(parent[n - 1] == no_parent);
    }
}

TEST_CASE("Predicted fill is a valid upper bound on actual LU fill",
          "[sparse][col_etree][fill]") {
    for (unsigned seed : {11u, 22u, 33u, 44u}) {
        auto A = random_unsym(25, 0.12, seed);
        auto sa = analysis::analyze_unsymmetric(A);

        // Actual nnz(L)+nnz(U) from a real factorization (natural ordering).
        auto sym = factorization::sparse_lu_symbolic(A);
        auto num = factorization::sparse_lu_numeric(A, sym);
        std::size_t actual = num.L.nnz() + num.U.nnz();

        REQUIRE(sa.fill_lu_bound >= actual);          // the static bound holds
        REQUIRE(sa.fill_chol_ata >= A.num_rows());    // at least the diagonal
    }
}

TEST_CASE("Unsymmetric supernode partition: structural extremes",
          "[sparse][col_etree][supernodal]") {
    SECTION("diagonal => all singleton supernodes") {
        mat::compressed2D<double> A(6, 6);
        { mat::inserter<mat::compressed2D<double>> ins(A);
          for (std::size_t i = 0; i < 6; ++i) ins[i][i] << 3.0; }
        auto sa = analysis::analyze_unsymmetric(A);
        REQUIRE(sa.nsuper == 6);
        REQUIRE(sa.sn_first.size() == 7);
        REQUIRE(sa.fill_lu_bound == 6);   // diagonal only: nnz(L)=nnz(U)=n, bound=2n-n
    }
    SECTION("rectangular matrix is rejected") {
        mat::compressed2D<double> A(3, 5);
        { mat::inserter<mat::compressed2D<double>> ins(A); ins[0][0] << 1.0; }
        REQUIRE_THROWS_AS(analysis::analyze_unsymmetric(A), std::invalid_argument);
    }
    SECTION("dense => single supernode") {
        std::size_t n = 8;
        mat::compressed2D<double> A(n, n);
        { mat::inserter<mat::compressed2D<double>> ins(A);
          for (std::size_t i = 0; i < n; ++i)
              for (std::size_t j = 0; j < n; ++j)
                  ins[i][j] << (i == j ? static_cast<double>(n) : 0.5); }
        auto sa = analysis::analyze_unsymmetric(A);
        REQUIRE(sa.nsuper == 1);
        REQUIRE(sa.col_perm.size() == n);
    }
}
