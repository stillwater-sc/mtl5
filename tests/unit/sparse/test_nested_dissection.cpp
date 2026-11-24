// Nested dissection ordering (issues 919 and 922).
//
//   - edge cases, for both separator methods: empty, 1x1, disconnected, empty
//     rows, an unsymmetric pattern, a non-square matrix, a long path (deep
//     separator tree), a star and a complete graph;
//   - below the cutoff it is exactly AMD;
//   - the multilevel separator's invariants: no edge between A and B, the
//     balance bound, and no vertex of S that touches only one side;
//   - determinism: a pinned hash of the permutation;
//   - fill and factor flops against AMD on 2-D and 3-D grid Laplacians, and the
//     growth exponents over a range of 2-D grids (level structure); multilevel
//     against level structure and AMD on grids and irregular graphs;
//   - correct solves through sparse_cholesky, sparse_lu, supernodal_ldlt and
//     supernodal_lu, under both methods.
//
// Level-structure separators (nd_separator::level_structure, cutoff 64,
// balance 0.7), Cholesky nnz(L) and flops (sum of squared column counts)
// relative to AMD:
//
//   2-D k x k   k=32   64     128    256    512
//   nnz         1.125  1.041  0.920  0.844  0.780
//   flops       1.148  0.944  0.692  0.618  0.538
//
//   3-D k^3     k=12   16     20     24     28     32
//   nnz         0.950  0.914  0.790  0.775  0.684  0.641
//   flops       0.766  0.683  0.520  0.526  0.404  0.355
//
// So on small 2-D grids AMD is as good or better; nested dissection wins as
// the grid grows, and in 3-D from the smallest size measured.
//
// Multilevel separators (the default), nnz(L) and flops relative to AMD, with
// the level structure's in parentheses:
//
//   2-D k x k     k=64          128           256           512
//   nnz           1.020 (1.041) 0.882 (0.920) 0.818 (0.844) 0.755 (0.780)
//   flops         0.885 (0.944) 0.620 (0.692) 0.580 (0.618) 0.499 (0.538)
//
//   3-D k^3       k=12          16            20            32
//   nnz           0.880 (0.950) 0.833 (0.914) 0.728 (0.790) 0.600 (0.641)
//   flops         0.701 (0.766) 0.595 (0.683) 0.467 (0.520) 0.327 (0.355)
//
//   irregular                      nnz           flops
//   grid 256^2, 60 holes r=12      1.060 (1.131) 0.963 (1.162)
//   random geometric 2-D, 50k      1.169 (2.045) 1.383 (6.117)
//   random geometric 3-D, 30k      0.701 (1.442) 0.444 (1.533)
//
// Multilevel is never worse than the level structure by more than 2% (2-D,
// k=32) and is much better on irregular graphs, where a breadth-first level
// can be far wider than the best separator. The bounds below leave margin
// around these numbers.
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/sparse/factorization/sparse_cholesky.hpp>
#include <mtl/sparse/factorization/sparse_lu.hpp>
#include <mtl/sparse/factorization/supernodal_ldlt.hpp>
#include <mtl/sparse/factorization/supernodal_lu.hpp>
#include <mtl/sparse/ordering/amd.hpp>
#include <mtl/sparse/ordering/nested_dissection.hpp>
#include <mtl/sparse/ordering/ordering_concepts.hpp>
#include <mtl/sparse/util/permutation.hpp>

using namespace mtl;
using namespace mtl::sparse;
using Mat = mat::compressed2D<double>;

namespace {

/// The 5-point (dim 2) or 7-point (dim 3) Laplacian on a grid of side k.
Mat grid_laplacian(std::size_t k, int dim) {
    const std::size_t kz = dim == 2 ? 1 : k;
    const std::size_t n = k * k * kz;
    Mat A(n, n);
    {
        mat::inserter<Mat> ins(A);
        for (std::size_t z = 0; z < kz; ++z)
            for (std::size_t r = 0; r < k; ++r)
                for (std::size_t c = 0; c < k; ++c) {
                    const std::size_t i = (z * k + r) * k + c;
                    ins[i][i] << 2.0 * dim;
                    if (c > 0)     ins[i][i - 1] << -1.0;
                    if (c + 1 < k) ins[i][i + 1] << -1.0;
                    if (r > 0)     ins[i][i - k] << -1.0;
                    if (r + 1 < k) ins[i][i + k] << -1.0;
                    if (dim == 3) {
                        if (z > 0)     ins[i][i - k * k] << -1.0;
                        if (z + 1 < k) ins[i][i + k * k] << -1.0;
                    }
                }
    }
    return A;
}

/// Cholesky nnz(L) and factor flops (sum of squared column counts) for an ordering.
template <typename Ordering>
std::pair<double, double> cholesky_cost(const Mat& A, const Ordering& ord) {
    const auto sym = factorization::sparse_cholesky_symbolic(A, ord);
    double flops = 0.0;
    for (std::size_t c : sym.col_counts) flops += static_cast<double>(c) * static_cast<double>(c);
    return {static_cast<double>(sym.nnz_L), flops};
}

double rel_residual(const Mat& A, const vec::dense_vector<double>& x,
                    const vec::dense_vector<double>& b) {
    const auto& starts = A.ref_major();
    const auto& idx = A.ref_minor();
    const auto& val = A.ref_data();
    double r2 = 0.0, b2 = 0.0;
    for (std::size_t i = 0; i < A.num_rows(); ++i) {
        double s = 0.0;
        for (std::size_t k = starts[i]; k < starts[i + 1]; ++k) s += val[k] * x(idx[k]);
        r2 += (s - b(i)) * (s - b(i));
        b2 += b(i) * b(i);
    }
    return std::sqrt(r2 / b2);
}

/// Laplacian-like SPD matrix (degree + 1 on the diagonal, -1 per edge) from an
/// edge list.
Mat from_edges(std::size_t n, const std::vector<std::pair<std::size_t, std::size_t>>& edges) {
    std::vector<double> d(n, 1.0);
    for (auto [a, b] : edges) { d[a] += 1.0; d[b] += 1.0; }
    Mat A(n, n);
    {
        mat::inserter<Mat> ins(A);
        for (std::size_t i = 0; i < n; ++i) ins[i][i] << d[i];
        for (auto [a, b] : edges) { ins[a][b] << -1.0; ins[b][a] << -1.0; }
    }
    return A;
}

/// 64-bit LCG (Knuth's MMIX constants): integer-only, so the test graphs are the
/// same on every platform.
struct lcg {
    std::uint64_t s;
    std::uint64_t next() { return s = s * 6364136223846793005ull + 1442695040888963407ull; }
};

/// Random geometric graph: n points with integer coordinates in [0, 2^16)^dim,
/// joined when their squared distance is below r2. Integer arithmetic only.
Mat random_geometric(std::size_t n, int dim, std::int64_t r2, std::uint64_t seed) {
    lcg g{seed};
    std::vector<std::array<std::int64_t, 3>> p(n);
    for (auto& q : p)
        for (int t = 0; t < 3; ++t) q[t] = t < dim ? static_cast<std::int64_t>(g.next() >> 48) : 0;
    std::vector<std::size_t> byx(n);
    for (std::size_t i = 0; i < n; ++i) byx[i] = i;
    std::sort(byx.begin(), byx.end(), [&](std::size_t a, std::size_t b) {
        return p[a][0] != p[b][0] ? p[a][0] < p[b][0] : a < b;
    });
    std::vector<std::pair<std::size_t, std::size_t>> edges;
    for (std::size_t a = 0; a < n; ++a)
        for (std::size_t b = a + 1; b < n; ++b) {
            const auto& u = p[byx[a]];
            const auto& v = p[byx[b]];
            const std::int64_t dx = v[0] - u[0];
            if (dx * dx >= r2) break;
            const std::int64_t d2 = dx * dx + (v[1] - u[1]) * (v[1] - u[1]) + (v[2] - u[2]) * (v[2] - u[2]);
            if (d2 < r2) edges.push_back({byx[a], byx[b]});
        }
    return from_edges(n, edges);
}

/// A k x k grid with `holes` square holes of side `side`, at random positions;
/// the remaining vertices are renumbered.
Mat grid_with_holes(std::size_t k, int holes, std::size_t side, std::uint64_t seed) {
    lcg g{seed};
    std::vector<char> gone(k * k, 0);
    for (int h = 0; h < holes; ++h) {
        const std::size_t r0 = static_cast<std::size_t>(g.next() >> 33) % k;
        const std::size_t c0 = static_cast<std::size_t>(g.next() >> 33) % k;
        for (std::size_t r = r0; r < std::min(k, r0 + side); ++r)
            for (std::size_t c = c0; c < std::min(k, c0 + side); ++c) gone[r * k + c] = 1;
    }
    std::vector<std::size_t> id(k * k);
    std::size_t n = 0;
    for (std::size_t i = 0; i < k * k; ++i) id[i] = gone[i] ? 0 : n++;
    std::vector<std::pair<std::size_t, std::size_t>> edges;
    for (std::size_t r = 0; r < k; ++r)
        for (std::size_t c = 0; c < k; ++c) {
            const std::size_t i = r * k + c;
            if (gone[i]) continue;
            if (c + 1 < k && !gone[i + 1]) edges.push_back({id[i], id[i + 1]});
            if (r + 1 < k && !gone[i + k]) edges.push_back({id[i], id[i + k]});
        }
    return from_edges(n, edges);
}

/// The graph of a symmetric matrix's off-diagonal pattern, with unit weights.
ordering::detail::nd_graph graph_of(const Mat& A) {
    ordering::detail::nd_graph g;
    const auto& s = A.ref_major();
    const auto& j = A.ref_minor();
    const std::size_t n = A.num_rows();
    g.xadj.assign(n + 1, 0);
    g.vwgt.assign(n, 1);
    for (std::size_t r = 0; r < n; ++r) {
        for (std::size_t k = s[r]; k < s[r + 1]; ++k)
            if (j[k] != r) {
                g.adj.push_back(static_cast<ordering::detail::idx>(j[k]));
                g.ewgt.push_back(1);
            }
        g.xadj[r + 1] = static_cast<ordering::detail::idx>(g.adj.size());
    }
    return g;
}

} // namespace

static_assert(FillReducingOrdering<ordering::nested_dissection, Mat>);

using ordering::nd_separator;
constexpr auto level_structure = nd_separator::level_structure;
constexpr auto multilevel = nd_separator::multilevel;

TEST_CASE("nested_dissection: the default separator is multilevel", "[sparse][ordering][nd]") {
    REQUIRE(ordering::nested_dissection{}.method == multilevel);
    REQUIRE(ordering::nested_dissection{32}.method == multilevel);
    REQUIRE(ordering::nested_dissection{level_structure}.method == level_structure);
    REQUIRE(ordering::nested_dissection{level_structure}.cutoff == 64);
}

TEST_CASE("nested_dissection: edge cases give valid permutations", "[sparse][ordering][nd]") {
    const auto method = GENERATE(level_structure, multilevel);
    const ordering::nested_dissection nd{method};

    SECTION("empty and 1x1") {
        REQUIRE(nd(Mat(0, 0)).empty());
        Mat one(1, 1);
        { mat::inserter<Mat> ins(one); ins[0][0] << 2.0; }
        REQUIRE(nd(one) == std::vector<std::size_t>{0});
    }

    SECTION("non-square is rejected") {
        REQUIRE_THROWS_AS(nd(Mat(3, 5)), std::invalid_argument);
    }

    SECTION("disconnected: a diagonal matrix and two disjoint grids") {
        Mat D(500, 500);
        { mat::inserter<Mat> ins(D); for (std::size_t i = 0; i < 500; ++i) ins[i][i] << 1.0; }
        REQUIRE(util::is_valid_permutation(nd(D)));

        // Two 20x20 grids side by side, each larger than the cutoff.
        const auto G = grid_laplacian(20, 2);
        const std::size_t m = G.num_rows();
        Mat B(2 * m, 2 * m);
        {
            mat::inserter<Mat> ins(B);
            const auto& s = G.ref_major(); const auto& j = G.ref_minor(); const auto& v = G.ref_data();
            for (std::size_t blk = 0; blk < 2; ++blk)
                for (std::size_t r = 0; r < m; ++r)
                    for (std::size_t k = s[r]; k < s[r + 1]; ++k)
                        ins[blk * m + r][blk * m + j[k]] << v[k];
        }
        const auto p = nd(B);
        REQUIRE(p.size() == 2 * m);
        REQUIRE(util::is_valid_permutation(p));
    }

    SECTION("empty rows") {
        // A 30x30 grid with every third row and column left out of the pattern.
        const std::size_t n = 900;
        Mat A(n, n);
        {
            mat::inserter<Mat> ins(A);
            for (std::size_t i = 0; i < n; ++i) {
                if (i % 3 == 0) continue;
                ins[i][i] << 4.0;
                if (i + 1 < n && (i + 1) % 3 != 0) { ins[i][i + 1] << -1.0; ins[i + 1][i] << -1.0; }
                if (i + 30 < n && (i + 30) % 3 != 0) { ins[i][i + 30] << -1.0; ins[i + 30][i] << -1.0; }
            }
        }
        REQUIRE(util::is_valid_permutation(nd(A)));
    }

    SECTION("an unsymmetric pattern is ordered as A + A^T") {
        // The strictly lower triangle of a grid Laplacian, plus its diagonal.
        const auto G = grid_laplacian(25, 2);
        const std::size_t n = G.num_rows();
        Mat L(n, n);
        {
            mat::inserter<Mat> ins(L);
            const auto& s = G.ref_major(); const auto& j = G.ref_minor(); const auto& v = G.ref_data();
            for (std::size_t r = 0; r < n; ++r)
                for (std::size_t k = s[r]; k < s[r + 1]; ++k)
                    if (j[k] <= r) ins[r][j[k]] << v[k];
        }
        const auto p = nd(L);
        REQUIRE(util::is_valid_permutation(p));
        // The symmetrized pattern is the grid's, so the ordering should cut its
        // fill as well as ordering the grid directly. (The permutations can
        // differ: neighbour order in the symmetrized pattern differs.)
        auto fixed = [&p](const Mat&) { return p; };
        const double from_lower = cholesky_cost(G, fixed).first;
        const double from_grid = cholesky_cost(G, nd).first;
        REQUIRE(from_lower < 1.1 * from_grid);
    }

    SECTION("a long path: deep separator tree, no recursion") {
        const std::size_t n = 100000;
        Mat P(n, n);
        {
            mat::inserter<Mat> ins(P);
            for (std::size_t i = 0; i < n; ++i) {
                ins[i][i] << 2.0;
                if (i + 1 < n) { ins[i][i + 1] << -1.0; ins[i + 1][i] << -1.0; }
            }
        }
        // Each split removes a one-vertex separator and leaves at most 70% of
        // the path, so about 20 levels of dissection down to the cutoff; every
        // pending piece lives on the explicit work stack.
        const auto p = nd(P);
        REQUIRE(util::is_valid_permutation(p));
    }

    SECTION("a star: diameter 2, no level separates it") {
        // Only a vertex separator works: the centre. Eliminating the leaves
        // first gives no fill at all, which AMD finds too.
        const std::size_t n = 1000;
        std::vector<std::pair<std::size_t, std::size_t>> e;
        for (std::size_t i = 1; i < n; ++i) e.push_back({0, i});
        const auto S = from_edges(n, e);
        const auto p = nd(S);
        REQUIRE(util::is_valid_permutation(p));
        REQUIRE(cholesky_cost(S, nd).first == cholesky_cost(S, ordering::amd{}).first);
    }

    SECTION("a complete graph has no separator") {
        const std::size_t n = 120;
        std::vector<std::pair<std::size_t, std::size_t>> e;
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = i + 1; j < n; ++j) e.push_back({i, j});
        REQUIRE(util::is_valid_permutation(nd(from_edges(n, e))));
    }
}

TEST_CASE("nested_dissection: at or below the cutoff it is AMD", "[sparse][ordering][nd]") {
    const auto method = GENERATE(level_structure, multilevel);
    const auto A = grid_laplacian(8, 2);             // 64 vertices, one component
    REQUIRE(ordering::nested_dissection{64, 0.7, method}(A) == ordering::amd{}(A));
    REQUIRE(ordering::nested_dissection{16, 0.7, method}(A) != ordering::amd{}(A));
}

TEST_CASE("multilevel_separator: separates, balances, and is minimal",
          "[sparse][ordering][nd]") {
    using namespace ordering::detail;
    struct named { const char* name; Mat A; };
    const std::vector<named> graphs = {
        {"2-D grid 64x64", grid_laplacian(64, 2)},
        {"3-D grid 16^3", grid_laplacian(16, 3)},
        {"grid with holes", grid_with_holes(96, 12, 9, 7)},
        {"random geometric 2-D", random_geometric(4000, 2, 2734261, 11)},
    };
    for (const auto& [name, A] : graphs) {
        INFO(name);
        const auto g = graph_of(A);
        const double balance = 0.7;
        const auto where = multilevel_separator(g, balance, 1);
        REQUIRE(where.size() == A.num_rows());
        const nd_weights pw = nd_part_weights(g, where);
        REQUIRE(pw.w[nd_part_a] > 0);
        REQUIRE(pw.w[nd_part_b] > 0);
        for (idx v = 0; v < g.size(); ++v) {
            bool touches[3] = {false, false, false};
            for (idx p = g.xadj[v]; p < g.xadj[v + 1]; ++p) touches[where[g.adj[p]]] = true;
            // No edge joins A and B.
            if (where[v] == nd_part_a) REQUIRE_FALSE(touches[nd_part_b]);
            if (where[v] == nd_part_b) REQUIRE_FALSE(touches[nd_part_a]);
            // Minimal: every separator vertex touches both sides, so none can
            // leave S on its own.
            if (where[v] == nd_part_s) REQUIRE((touches[nd_part_a] && touches[nd_part_b]));
        }
        // Refinement keeps the larger side within `balance` of A + B; the final
        // thinning can move up to |S| more vertices onto one side.
        const double big = static_cast<double>(std::max(pw.w[nd_part_a], pw.w[nd_part_b]));
        REQUIRE(big <= balance * static_cast<double>(pw.w[nd_part_a] + pw.w[nd_part_b])
                           + static_cast<double>(pw.w[nd_part_s]));
    }

    SECTION("a grid's separator is about one row") {
        // Measured: 64 vertices on the 64 x 64 grid, one straight row.
        const auto g = graph_of(grid_laplacian(64, 2));
        const auto pw = nd_part_weights(g, multilevel_separator(g, 0.7, 1));
        REQUIRE(pw.w[nd_part_s] <= 66);
    }
}

TEST_CASE("nested_dissection: multilevel is deterministic", "[sparse][ordering][nd]") {
    // Integer-only test graphs, a fixed-seed integer RNG and totally ordered
    // priorities: the same permutation with every compiler and standard
    // library, pinned here by its FNV-1a hash.
    auto fnv = [](const std::vector<std::size_t>& p) {
        std::uint64_t h = 1469598103934665603ull;
        for (std::size_t v : p) { h ^= static_cast<std::uint64_t>(v); h *= 1099511628211ull; }
        return h;
    };
    const ordering::nested_dissection nd;
    const auto A = random_geometric(3000, 2, 3645681, 5);
    const auto p = nd(A);
    REQUIRE(util::is_valid_permutation(p));
    REQUIRE(nd(A) == p);
    const auto B = grid_with_holes(60, 8, 7, 3);
    const auto q = nd(B);
    INFO("hashes " << fnv(p) << " " << fnv(q));
    REQUIRE(fnv(p) == 8123638548852251361ull);
    REQUIRE(fnv(q) == 12346266757541259587ull);
}

TEST_CASE("nested_dissection: less fill and work than AMD on large grids",
          "[sparse][ordering][nd]") {
    // The level-structure separators; the multilevel ones are tested below.
    const ordering::nested_dissection nd{level_structure};

    SECTION("2-D, k = 256: nnz 0.84x and flops 0.62x AMD's, measured") {
        const auto A = grid_laplacian(256, 2);
        const auto [nnz_amd, fl_amd] = cholesky_cost(A, ordering::amd{});
        const auto [nnz_nd, fl_nd] = cholesky_cost(A, nd);
        INFO("nnz " << nnz_nd << " vs " << nnz_amd << ", flops " << fl_nd << " vs " << fl_amd);
        REQUIRE(nnz_nd < 0.92 * nnz_amd);
        REQUIRE(fl_nd < 0.75 * fl_amd);
    }

    SECTION("3-D, k = 20: nnz 0.79x and flops 0.52x AMD's, measured") {
        const auto A = grid_laplacian(20, 3);
        const auto [nnz_amd, fl_amd] = cholesky_cost(A, ordering::amd{});
        const auto [nnz_nd, fl_nd] = cholesky_cost(A, nd);
        INFO("nnz " << nnz_nd << " vs " << nnz_amd << ", flops " << fl_nd << " vs " << fl_amd);
        REQUIRE(nnz_nd < 0.9 * nnz_amd);
        REQUIRE(fl_nd < 0.7 * fl_amd);
    }

    SECTION("2-D growth from k = 64 to 256: lower exponents than AMD") {
        // Exponents in n = k^2. Measured: fill 1.14 (ND) vs 1.22 (AMD), against
        // about 1.10 for n log n; flops 1.53 vs 1.68, against the optimal 1.5.
        const auto small = grid_laplacian(64, 2), large = grid_laplacian(256, 2);
        const double span = std::log(16.0);
        auto growth = [&](const auto& ord) {
            const auto a = cholesky_cost(small, ord), b = cholesky_cost(large, ord);
            return std::pair{std::log(b.first / a.first) / span, std::log(b.second / a.second) / span};
        };
        const auto [fill_amd, flops_amd] = growth(ordering::amd{});
        const auto [fill_nd, flops_nd] = growth(nd);
        INFO("fill " << fill_nd << " vs " << fill_amd << ", flops " << flops_nd << " vs " << flops_amd);
        REQUIRE(fill_nd < fill_amd - 0.04);
        REQUIRE(flops_nd < flops_amd - 0.08);
        REQUIRE(flops_nd < 1.6);
    }
}

TEST_CASE("nested_dissection: multilevel beats level structures, and AMD in 3-D",
          "[sparse][ordering][nd]") {
    const ordering::nested_dissection ml{multilevel}, ls{level_structure};
    struct row { const char* name; Mat A; };

    SECTION("grids: at least as good as level structures") {
        // Measured nnz ML/LS: 0.969 (2-D, k=256), 0.922 (3-D, k=20).
        for (const auto& [name, A] : {row{"2-D k=256", grid_laplacian(256, 2)},
                                      row{"3-D k=20", grid_laplacian(20, 3)}}) {
            INFO(name);
            const auto [nnz_ml, fl_ml] = cholesky_cost(A, ml);
            const auto [nnz_ls, fl_ls] = cholesky_cost(A, ls);
            const auto [nnz_amd, fl_amd] = cholesky_cost(A, ordering::amd{});
            INFO("nnz " << nnz_ml / nnz_amd << " (LS " << nnz_ls / nnz_amd << "), flops "
                        << fl_ml / fl_amd << " (LS " << fl_ls / fl_amd << ") of AMD's");
            REQUIRE(nnz_ml < nnz_ls);
            REQUIRE(fl_ml < fl_ls);
            REQUIRE(nnz_ml < 0.88 * nnz_amd);
            REQUIRE(fl_ml < 0.66 * fl_amd);
        }
    }

    SECTION("irregular graphs: much better than level structures") {
        // Measured nnz ML/LS, flops ML/LS:
        //                                  nnz    flops   (of AMD's: nnz, flops)
        //   random geometric 2-D (n=10000)  0.614  0.305   (1.173, 1.449)
        //   random geometric 3-D (n=10000)  0.500  0.265   (0.754, 0.460)
        //   grid with holes (128^2)         0.941  0.799   (1.047, 0.893)
        const auto R2 = random_geometric(10000, 2, 1093704, 21);
        const auto R3 = random_geometric(10000, 3, 18664726, 22);
        const auto H = grid_with_holes(128, 20, 12, 23);
        auto ratios = [&](const Mat& A) {
            const auto a = cholesky_cost(A, ml), b = cholesky_cost(A, ls), c = cholesky_cost(A, ordering::amd{});
            return std::array<double, 4>{a.first / b.first, a.second / b.second,
                                         a.first / c.first, a.second / c.second};
        };
        const auto r2 = ratios(R2), r3 = ratios(R3), h = ratios(H);
        INFO("rgg2 " << r2[0] << " " << r2[1] << " " << r2[2] << " " << r2[3]);
        INFO("rgg3 " << r3[0] << " " << r3[1] << " " << r3[2] << " " << r3[3]);
        INFO("holes " << h[0] << " " << h[1] << " " << h[2] << " " << h[3]);
        CHECK(r2[0] < 0.72);
        CHECK(r2[1] < 0.45);
        CHECK(r3[0] < 0.6);
        CHECK(r3[1] < 0.4);
        CHECK(r3[2] < 0.85);       // and beats AMD in 3-D
        CHECK(r3[3] < 0.6);
        CHECK(h[0] < 0.98);
        CHECK(h[1] < 0.9);
    }
}

TEST_CASE("nested_dissection: the factorizations solve correctly under it",
          "[sparse][ordering][nd]") {
    const auto A = grid_laplacian(40, 2);            // 1600 vertices: dissected
    const std::size_t n = A.num_rows();
    vec::dense_vector<double> b(n);
    for (std::size_t i = 0; i < n; ++i) b(i) = 1.0 + static_cast<double>(i % 7);
    const ordering::nested_dissection nd{GENERATE(level_structure, multilevel)};

    vec::dense_vector<double> x(n, 0.0);
    factorization::sparse_cholesky_solve(A, x, b, nd);
    REQUIRE(rel_residual(A, x, b) < 1e-12);

    x = vec::dense_vector<double>(n, 0.0);
    factorization::sparse_lu_solve(A, x, b, nd);
    REQUIRE(rel_residual(A, x, b) < 1e-12);

    x = vec::dense_vector<double>(n, 0.0);
    factorization::supernodal_ldlt_solve(A, x, b, nd);
    REQUIRE(rel_residual(A, x, b) < 1e-12);

    x = vec::dense_vector<double>(n, 0.0);
    factorization::supernodal_lu_solve(A, x, b, nd);
    REQUIRE(rel_residual(A, x, b) < 1e-12);
}
