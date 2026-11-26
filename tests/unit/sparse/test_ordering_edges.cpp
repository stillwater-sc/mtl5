// Sparse orderings, symbolic analysis and permutations: the edge cases of
// issue 865 (QA sweep #704).
//
//   - colamd reduces fill on short, wide matrices (its dense-row threshold is
//     sized by the columns, as in CSparse);
//   - amd rejects a non-square matrix; colamd and maximum_matching take a
//     rectangular one, block_triangular_form rejects it;
//   - the util/permutation.hpp functions check sizes and indices in every build;
//   - elimination_tree and column_counts with a 32-bit SizeType;
//   - rcm is linear in the number of components, with the same order;
//   - column_counts against symbolic_cholesky_pattern; find_supernodes,
//     column_supernodes, reach() and sparse_lower_solve directly.
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/sparse/analysis/column_etree.hpp>
#include <mtl/sparse/analysis/elimination_tree.hpp>
#include <mtl/sparse/analysis/supernodes.hpp>
#include <mtl/sparse/factorization/triangular_solve.hpp>
#include <mtl/sparse/ordering/amd.hpp>
#include <mtl/sparse/ordering/colamd.hpp>
#include <mtl/sparse/ordering/dulmage_mendelsohn.hpp>
#include <mtl/sparse/ordering/rcm.hpp>
#include <mtl/sparse/util/csc.hpp>
#include <mtl/sparse/util/permutation.hpp>

using namespace mtl;
namespace an = mtl::sparse::analysis;
namespace od = mtl::sparse::ordering;
namespace ut = mtl::sparse::util;

namespace {

using csr     = mat::compressed2D<double>;
using entries = std::vector<std::tuple<std::size_t, std::size_t, double>>;

csr make(std::size_t m, std::size_t n, const entries& e) {
    csr A(m, n);
    {
        mat::inserter<csr> ins(A);
        for (const auto& [i, j, v] : e) ins[i][j] << v;
    }
    return A;
}

/// A random symmetric pattern with a full diagonal.
csr random_symmetric(std::size_t n, std::size_t extra, unsigned seed) {
    std::mt19937 g(seed);
    std::uniform_int_distribution<std::size_t> c(0, n - 1);
    std::set<std::pair<std::size_t, std::size_t>> s;
    for (std::size_t i = 0; i < n; ++i) s.insert({i, i});
    for (std::size_t k = 0; k < extra; ++k) {
        const std::size_t i = c(g), j = c(g);
        s.insert({i, j});
        s.insert({j, i});
    }
    entries e;
    for (const auto& [i, j] : s) e.emplace_back(i, j, i == j ? double(n) : -1.0);
    return make(n, n, e);
}

bool is_permutation_of(const std::vector<std::size_t>& p, std::size_t n) {
    return p.size() == n && ut::is_valid_permutation(p);
}

/// rcm as it was before issue 865: an O(n) scan for each component's start.
std::vector<std::size_t> rcm_reference(const csr& A) {
    const std::size_t n = A.num_rows();
    const auto& starts = A.ref_major();
    const auto& indices = A.ref_minor();
    std::vector<std::size_t> degree(n);
    for (std::size_t i = 0; i < n; ++i) degree[i] = starts[i + 1] - starts[i];
    std::vector<bool> visited(n, false);
    std::vector<std::size_t> order;
    while (order.size() < n) {
        std::size_t start = 0, min_deg = n + 1;
        for (std::size_t i = 0; i < n; ++i)
            if (!visited[i] && degree[i] < min_deg) { min_deg = degree[i]; start = i; }
        std::queue<std::size_t> bfs;
        bfs.push(start);
        visited[start] = true;
        while (!bfs.empty()) {
            const std::size_t node = bfs.front();
            bfs.pop();
            order.push_back(node);
            std::vector<std::size_t> nb;
            for (std::size_t k = starts[node]; k < starts[node + 1]; ++k)
                if (!visited[indices[k]]) nb.push_back(indices[k]);
            std::sort(nb.begin(), nb.end(), [&](std::size_t a, std::size_t b) { return degree[a] < degree[b]; });
            for (std::size_t v : nb)
                if (!visited[v]) { visited[v] = true; bfs.push(v); }
        }
    }
    std::reverse(order.begin(), order.end());
    return order;
}

} // namespace

// --- COLAMD on short and wide matrices -------------------------------------------------

namespace {

/// nnz of the Cholesky factor of (A Q)^T (A Q), the fill colamd minimizes.
std::size_t ata_fill(const csr& A, const std::vector<std::size_t>& q) {
    const std::size_t m = A.num_rows(), n = A.num_cols();
    std::set<std::pair<std::size_t, std::size_t>> s;
    const auto& rp = A.ref_major();
    const auto& ci = A.ref_minor();
    for (std::size_t j = 0; j < n; ++j) s.insert({j, j});
    for (std::size_t r = 0; r < m; ++r)
        for (std::size_t a = rp[r]; a < rp[r + 1]; ++a)
            for (std::size_t b = rp[r]; b < rp[r + 1]; ++b) s.insert({ci[a], ci[b]});
    entries e;
    for (const auto& [i, j] : s) e.emplace_back(i, j, 1.0);
    const auto C = ut::crs_to_csc(ut::symmetric_permute(make(n, n, e), q));
    return an::total_nnz(an::column_counts(C, an::elimination_tree(C)));
}

} // namespace

TEST_CASE("colamd reduces fill on short, wide matrices", "[sparse][ordering][colamd][865]") {
    // 3x10, row r = {0, 3r+1, 3r+2, 3r+3}: column 0 meets every other in A^T A.
    // The dense-row threshold was dense_threshold(m = 3) = 1 (the n - 2 clamp):
    // every row was dropped, A^T A was empty and colamd returned the identity,
    // which eliminates column 0 first and fills all of the rest.
    const csr B = make(3, 10, {{0, 0, 1.0}, {0, 1, 1.0}, {0, 2, 1.0}, {0, 3, 1.0}, {1, 0, 1.0}, {1, 4, 1.0},
                               {1, 5, 1.0}, {1, 6, 1.0}, {2, 0, 1.0}, {2, 7, 1.0}, {2, 8, 1.0}, {2, 9, 1.0}});
    const auto q = od::colamd{}(B);
    REQUIRE(is_permutation_of(q, 10));
    const auto id = ut::identity_permutation(10);
    CHECK(q.back() == 0);
    CHECK(ata_fill(B, q) < ata_fill(B, id));
    // 2x6, rows {0,1,5} and {2,3}: the threshold was 0, the same identity.
    const csr A = make(2, 6, {{0, 0, 1.0}, {0, 1, 1.0}, {0, 5, 1.0}, {1, 2, 1.0}, {1, 3, 1.0}});
    const auto p = od::colamd{}(A);
    REQUIRE(is_permutation_of(p, 6));
    CHECK(p != ut::identity_permutation(6));
    // Tall: 6x2 has nothing to drop either way.
    const csr C = make(6, 2, {{0, 0, 1.0}, {1, 1, 1.0}, {2, 0, 1.0}, {3, 1, 1.0}, {4, 0, 1.0}, {5, 1, 1.0}});
    CHECK(is_permutation_of(od::colamd{}(C), 2));
    // Tall with n = 1 and n = 3 columns, dense_threshold(n) of -1 and 1 (issue
    // 888): every row may count as dense; the result must still be a
    // permutation of the columns, as CSparse's.
    const csr D = make(5, 1, {{0, 0, 1.0}, {2, 0, 1.0}, {4, 0, 1.0}});
    CHECK(is_permutation_of(od::colamd{}(D), 1));
    const csr E = make(6, 3, {{0, 0, 1.0}, {0, 1, 1.0}, {0, 2, 1.0}, {1, 1, 1.0}, {2, 2, 1.0},
                              {3, 0, 1.0}, {4, 1, 1.0}, {4, 2, 1.0}, {5, 0, 1.0}});
    CHECK(is_permutation_of(od::colamd{}(E), 3));
}

// --- non-square input ---------------------------------------------------------------------

TEST_CASE("orderings and matchings on non-square input", "[sparse][ordering][865]") {
    // 3x5: amd's A + A^T overflowed the heap under NDEBUG.
    const csr W = make(3, 5, {{0, 0, 1.0}, {0, 4, 1.0}, {1, 1, 1.0}, {2, 2, 1.0}, {2, 3, 1.0}});
    CHECK_THROWS_AS(od::amd{}(W), std::invalid_argument);
    CHECK_THROWS_AS(od::rcm{}(W), std::invalid_argument);
    CHECK_THROWS_AS(od::block_triangular_form(W), std::invalid_argument);
    CHECK(is_permutation_of(od::colamd{}(W), 5));

    // maximum_matching is defined for rectangular A.
    const csr wide = make(2, 3, {{0, 0, 1.0}, {0, 1, 1.0}, {1, 1, 1.0}, {1, 2, 1.0}});
    const auto mw = od::maximum_matching(wide);
    REQUIRE(mw.size() == 2);
    CHECK(mw[0] >= 0);
    CHECK(mw[1] >= 0);
    CHECK(mw[0] != mw[1]);
    const csr tall = make(3, 2, {{0, 0, 1.0}, {1, 0, 1.0}, {2, 1, 1.0}});
    const auto mt = od::maximum_matching(tall);
    REQUIRE(mt.size() == 3);
    CHECK(std::count(mt.begin(), mt.end(), -1) == 1);
    CHECK(mt[2] == 1);
}

// --- permutation utilities ----------------------------------------------------------------

TEST_CASE("permutation utilities check sizes and indices in every build", "[sparse][permutation][865]") {
    CHECK_THROWS_AS(ut::invert_permutation({0, 5}), std::invalid_argument);
    CHECK_THROWS_AS(ut::invert_permutation({1, 1}), std::invalid_argument);
    CHECK(ut::invert_permutation({2, 0, 1}) == std::vector<std::size_t>({1, 2, 0}));
    CHECK_THROWS_AS(ut::compose_permutations({0, 1}, {0, 1, 2}), std::invalid_argument);
    CHECK_THROWS_AS(ut::compose_permutations({0, 1}, {0, 7}), std::invalid_argument);
    const csr A = make(3, 3, {{0, 0, 1.0}, {1, 1, 2.0}, {2, 2, 3.0}, {0, 2, 4.0}});
    CHECK_THROWS_AS(ut::symmetric_permute(A, {0, 1}), std::invalid_argument);
    CHECK_THROWS_AS(ut::symmetric_permute(make(2, 3, {{0, 0, 1.0}}), {0, 1}), std::invalid_argument);
    CHECK_THROWS_AS(ut::column_permute(A, {0, 1}), std::invalid_argument);
    CHECK_THROWS_AS(ut::column_permute(A, {0, 1, 3}), std::invalid_argument);
    CHECK_THROWS_AS(ut::row_permute(A, {0, 1, 2, 3}), std::invalid_argument);
    CHECK_THROWS_AS(ut::permute_vector(std::vector<double>{1, 2}, {0, 2}), std::invalid_argument);
    CHECK_THROWS_AS(ut::ipermute_vector(std::vector<double>{1, 2}, {0}), std::invalid_argument);
    const auto B = ut::symmetric_permute(A, {2, 0, 1});
    CHECK(B(0, 0) == 3.0);
    CHECK(B(1, 0) == 4.0);   // A(0, 2) lands at (pinv[0], pinv[2]) = (1, 0)
}

// --- elimination tree with a 32-bit SizeType ------------------------------------------------

TEST_CASE("elimination_tree and column_counts with a 32-bit SizeType", "[sparse][etree][865]") {
    for (unsigned seed : {1u, 2u, 3u}) {
        const csr A = random_symmetric(60, 80, seed);
        const auto C64 = ut::crs_to_csc(A);
        ut::csc_matrix<double, std::uint32_t> C32;
        C32.nrows = static_cast<std::uint32_t>(C64.nrows);
        C32.ncols = static_cast<std::uint32_t>(C64.ncols);
        C32.col_ptr.assign(C64.col_ptr.begin(), C64.col_ptr.end());
        C32.row_ind.assign(C64.row_ind.begin(), C64.row_ind.end());
        C32.values = C64.values;
        // `node = ancestor[node]` truncated no_parent to 2^32 - 1, and the
        // walk indexed ancestor[2^32 - 1].
        const auto p64 = an::elimination_tree(C64);
        const auto p32 = an::elimination_tree(C32);
        CHECK(p32 == p64);
        CHECK(an::column_counts(C32, p32) == an::column_counts(C64, p64));
    }
}

// --- rcm ---------------------------------------------------------------------------------------

TEST_CASE("rcm: the same order as before, linear in the components", "[sparse][ordering][rcm][865]") {
    for (unsigned seed : {1u, 2u, 3u, 4u}) {
        // Sparse random patterns: several components, ties in degree.
        const csr A = random_symmetric(200, 60, seed);
        CHECK(od::rcm{}(A) == rcm_reference(A));
    }
    {
        entries e;
        for (std::size_t i = 0; i < 50; ++i) e.emplace_back(i, i, 1.0);
        const csr D = make(50, 50, e);
        std::vector<std::size_t> rev(50);
        std::iota(rev.rbegin(), rev.rend(), std::size_t{0});
        CHECK(od::rcm{}(D) == rev);
    }
    // A diagonal matrix: n components. n = 20000 took 0.60 s, 40000 2.38 s.
    // 16x in n is ~16x in time now; the rescan took ~256x. The bound is 64x.
    auto t = [](std::size_t n) {
        entries e;
        for (std::size_t i = 0; i < n; ++i) e.emplace_back(i, i, 1.0);
        const csr D = make(n, n, e);
        double best = std::numeric_limits<double>::infinity();
        for (int r = 0; r < 3; ++r) {
            const auto t0 = std::chrono::steady_clock::now();
            const auto p = od::rcm{}(D);
            const auto t1 = std::chrono::steady_clock::now();
            REQUIRE(p.size() == n);
            best = std::min(best, std::chrono::duration<double>(t1 - t0).count());
        }
        return best;
    };
    const double small = t(4000), large = t(64000);
    INFO("t(4000) = " << small << " s, t(64000) = " << large << " s");
    CHECK(large < 64.0 * std::max(small, 1e-5));
}

// --- symbolic analysis ---------------------------------------------------------------------

TEST_CASE("column_counts agrees with symbolic_cholesky_pattern", "[sparse][analysis][865]") {
    for (unsigned seed : {1u, 2u, 3u, 4u, 5u}) {
        const csr A = random_symmetric(80, 120, seed);
        const auto C = ut::crs_to_csc(A);
        const auto parent = an::elimination_tree(C);
        const auto counts = an::column_counts(C, parent);
        std::vector<std::size_t> Lp, Li;
        an::symbolic_cholesky_pattern(C, parent, Lp, Li);
        REQUIRE(Lp.size() == 81);
        for (std::size_t j = 0; j < 80; ++j) {
            CAPTURE(seed, j);
            CHECK(counts[j] == Lp[j + 1] - Lp[j]);
            CHECK(Li[Lp[j]] == j);   // the diagonal first
            CHECK(std::is_sorted(Li.begin() + Lp[j], Li.begin() + Lp[j + 1]));
        }
    }
}

TEST_CASE("find_supernodes on known structures", "[sparse][analysis][supernodes][865]") {
    auto partition = [](const csr& A) {
        const auto C = ut::crs_to_csc(A);
        return an::find_supernodes(C, an::elimination_tree(C));
    };
    {   // dense 4x4: one supernode, rows 0..3
        entries e;
        for (std::size_t i = 0; i < 4; ++i)
            for (std::size_t j = 0; j < 4; ++j) e.emplace_back(i, j, i == j ? 4.0 : 1.0);
        const auto sp = partition(make(4, 4, e));
        CHECK(sp.nsuper == 1);
        CHECK(sp.sn_first == std::vector<std::size_t>({0, 4}));
        CHECK(std::vector<std::size_t>(sp.row_idx.begin() + sp.lnz_ptr[0], sp.row_idx.begin() + sp.lnz_ptr[1])
              == std::vector<std::size_t>({0, 1, 2, 3}));
        CHECK(sp.col_counts == std::vector<std::size_t>({4, 3, 2, 1}));
    }
    {   // diagonal: no parent, n supernodes
        const auto sp = partition(make(5, 5, {{0, 0, 1.0}, {1, 1, 1.0}, {2, 2, 1.0}, {3, 3, 1.0}, {4, 4, 1.0}}));
        CHECK(sp.nsuper == 5);
        CHECK(sp.col_counts == std::vector<std::size_t>(5, 1));
    }
    {   // tridiagonal: counts 2, ..., 2, 1; only the last two columns nest
        entries e;
        for (std::size_t i = 0; i < 6; ++i) {
            e.emplace_back(i, i, 4.0);
            if (i + 1 < 6) { e.emplace_back(i, i + 1, -1.0); e.emplace_back(i + 1, i, -1.0); }
        }
        const auto sp = partition(make(6, 6, e));
        CHECK(sp.nsuper == 5);
        CHECK(sp.sn_first.back() == 6);
        CHECK(sp.super[4] == sp.super[5]);
        CHECK(sp.super[3] != sp.super[4]);
    }
    {   // empty
        const auto sp = partition(csr(0, 0));
        CHECK(sp.nsuper == 0);
    }
}

TEST_CASE("column_supernodes merges only-child chains with nested counts", "[sparse][analysis][supernodes][865]") {
    constexpr std::size_t none = an::no_parent;
    // A chain 0 -> 1 -> 2 -> 3 with counts 4, 3, 2, 1: one supernode.
    std::size_t nsuper = 0;
    std::vector<std::size_t> super, first;
    an::column_supernodes({1, 2, 3, none}, {4, 3, 2, 1}, 4, nsuper, super, first);
    CHECK(nsuper == 1);
    CHECK(first == std::vector<std::size_t>({0, 4}));
    // Column 2 has two children (0 and 1): 0 | 1 | 2 3.
    an::column_supernodes({2, 2, 3, none}, {3, 3, 2, 1}, 4, nsuper, super, first);
    CHECK(nsuper == 3);
    CHECK(first == std::vector<std::size_t>({0, 1, 2, 4}));
    // Counts that do not nest split the chain: 3, 2 nest; 2, 2 do not.
    an::column_supernodes({1, 2, none}, {3, 2, 2}, 3, nsuper, super, first);
    CHECK(first == std::vector<std::size_t>({0, 2, 3}));
    an::column_supernodes({}, {}, 0, nsuper, super, first);
    CHECK(nsuper == 0);
}

TEST_CASE("reach() and sparse_lower_solve", "[sparse][triangular][865]") {
    // L, 5x5 lower: column j has entries below at the rows listed.
    //   col 0: rows 0, 2      col 1: rows 1, 3      col 2: rows 2, 4
    //   col 3: rows 3, 4      col 4: row 4
    ut::csc_matrix<double> L;
    L.nrows = L.ncols = 5;
    L.col_ptr = {0, 2, 4, 6, 8, 9};
    L.row_ind = {0, 2, 1, 3, 2, 4, 3, 4, 4};
    L.values  = {2.0, 1.0, 4.0, -1.0, 3.0, 2.0, 5.0, 1.0, 2.0};
    std::vector<std::size_t> xi(5), stack(5);
    std::vector<bool> marked(5, false);
    // b = e0: the reach is 0 -> 2 -> 4, in topological order.
    const std::size_t top = sparse::factorization::reach(L.col_ptr, L.row_ind, 5, {0}, xi, stack, marked);
    CHECK(std::vector<std::size_t>(xi.begin() + top, xi.end()) == std::vector<std::size_t>({0, 2, 4}));
    std::vector<double> x = {6.0, 0.0, 0.0, 0.0, 0.0};
    sparse::factorization::sparse_lower_solve(L, x, xi, top);
    // Dense forward substitution for reference.
    std::vector<double> y = {6.0, 0.0, 0.0, 0.0, 0.0};
    for (std::size_t j = 0; j < 5; ++j) {
        y[j] /= L.values[L.col_ptr[j]];
        for (std::size_t p = L.col_ptr[j] + 1; p < L.col_ptr[j + 1]; ++p) y[L.row_ind[p]] -= L.values[p] * y[j];
    }
    CHECK(x == y);
    CHECK(x[1] == 0.0);   // outside the reach: untouched
    CHECK(x[3] == 0.0);
    // b = e1 + e3: reach {1, 3, 4} (marked is the caller's to clear).
    std::fill(marked.begin(), marked.end(), false);
    const std::size_t top2 = sparse::factorization::reach(L.col_ptr, L.row_ind, 5, {1, 3}, xi, stack, marked);
    std::vector<std::size_t> r(xi.begin() + top2, xi.end());
    CHECK(std::set<std::size_t>(r.begin(), r.end()) == std::set<std::size_t>({1, 3, 4}));
    // Topological: 1 before 3 before 4.
    auto pos = [&](std::size_t v) { return std::find(r.begin(), r.end(), v) - r.begin(); };
    CHECK(pos(1) < pos(3));
    CHECK(pos(3) < pos(4));
}

// A node with many children (issue 907): reach() rescanned the node's column
// from the start each time a child returned, O(d^2) for d children; it now
// resumes where it stopped. A star: column 0 holds rows 1..d, and each child
// has one child of its own, d + i. Checked against a dense forward
// substitution, through the public sparse_lower_solve.
TEST_CASE("reach() and sparse_lower_solve on a node with many children", "[sparse][triangular][907]") {
    const std::size_t d = 3000, n = 2 * d + 1;
    ut::csc_matrix<double> L;
    L.nrows = L.ncols = n;
    L.col_ptr.push_back(0);
    L.row_ind.push_back(0); L.values.push_back(2.0);
    for (std::size_t i = 1; i <= d; ++i) { L.row_ind.push_back(i); L.values.push_back(-1.0 / double(i)); }
    L.col_ptr.push_back(L.row_ind.size());
    for (std::size_t j = 1; j < n; ++j) {
        L.row_ind.push_back(j); L.values.push_back(1.5);
        if (j <= d) { L.row_ind.push_back(d + j); L.values.push_back(0.25); }
        L.col_ptr.push_back(L.row_ind.size());
    }
    std::vector<std::size_t> xi(n), stack(n), pos(n);
    std::vector<bool> marked(n, false);
    const std::size_t top = sparse::factorization::reach(L.col_ptr, L.row_ind, n, {0}, xi, stack, pos, marked);
    REQUIRE(n - top == n);   // everything is reachable from 0
    REQUIRE(xi[top] == 0);
    std::vector<std::size_t> where(n);
    for (std::size_t q = top; q < n; ++q) where[xi[q]] = q;
    for (std::size_t i = 1; i <= d; ++i) {   // topological: parent before child
        REQUIRE(where[0] < where[i]);
        REQUIRE(where[i] < where[d + i]);
    }
    std::vector<double> x(n, 0.0), y(n, 0.0);
    x[0] = y[0] = 6.0;
    sparse::factorization::sparse_lower_solve(L, x, xi, top);
    sparse::factorization::dense_lower_solve(L, y);
    REQUIRE(x == y);

    // The overload without a position workspace gives the same order.
    std::vector<std::size_t> xi2(n);
    std::fill(marked.begin(), marked.end(), false);
    const std::size_t top2 = sparse::factorization::reach(L.col_ptr, L.row_ind, n, {0}, xi2, stack, marked);
    REQUIRE(top2 == top);
    REQUIRE(xi2 == xi);
}

// The test above checks reach()'s result on the star, not its cost (issue
// 926). From d = 4000 to d = 32000 (8x) linear work takes ~8x as long; the
// old rescan of the node's column, O(d^2), took ~64x. The bound is 24x, a
// factor of 3 from either, and each time is the best of several runs.
TEST_CASE("reach() on a node with many children is linear in the children", "[sparse][triangular][907][926]") {
    auto star = [](std::size_t d) {
        const std::size_t n = 2 * d + 1;
        ut::csc_matrix<double> L;
        L.nrows = L.ncols = n;
        L.col_ptr.push_back(0);
        L.row_ind.push_back(0); L.values.push_back(2.0);
        for (std::size_t i = 1; i <= d; ++i) { L.row_ind.push_back(i); L.values.push_back(-1.0); }
        L.col_ptr.push_back(L.row_ind.size());
        for (std::size_t j = 1; j < n; ++j) {
            L.row_ind.push_back(j); L.values.push_back(1.5);
            if (j <= d) { L.row_ind.push_back(d + j); L.values.push_back(0.25); }
            L.col_ptr.push_back(L.row_ind.size());
        }
        return L;
    };
    auto time_reach = [](const ut::csc_matrix<double>& L) {
        const std::size_t n = L.ncols;
        std::vector<std::size_t> xi(n), stack(n), pos(n);
        std::vector<bool> marked(n, false);
        double best = 1e30;
        for (int r = 0; r < 7; ++r) {
            std::fill(marked.begin(), marked.end(), false);
            const auto t0 = std::chrono::steady_clock::now();
            const std::size_t top = sparse::factorization::reach(L.col_ptr, L.row_ind, n, {0}, xi, stack, pos, marked);
            best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
            REQUIRE(top == 0);
        }
        return best;
    };
    const auto small = star(4000), large = star(32000);
    const double ratio = time_reach(large) / time_reach(small);
    INFO("reach time ratio for 8x the children: " << ratio);
    CHECK(ratio < 24.0);
}
