// Nested dissection ordering (issue 919).
//
//   - edge cases: empty, 1x1, disconnected, empty rows, an unsymmetric
//     pattern, a non-square matrix, a long path (deep separator tree);
//   - below the cutoff it is exactly AMD;
//   - fill and factor flops against AMD on 2-D and 3-D grid Laplacians, and the
//     growth exponents over a range of 2-D grids;
//   - correct solves through sparse_cholesky, sparse_lu, supernodal_ldlt and
//     supernodal_lu.
//
// Measured with the defaults (cutoff 64, balance 0.7), Cholesky nnz(L) and
// flops (sum of squared column counts) relative to AMD:
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
// the grid grows, and in 3-D from the smallest size measured. The bounds below
// leave margin around these numbers.
#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <cmath>
#include <cstddef>
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

} // namespace

static_assert(FillReducingOrdering<ordering::nested_dissection, Mat>);

TEST_CASE("nested_dissection: edge cases give valid permutations", "[sparse][ordering][nd]") {
    const ordering::nested_dissection nd;

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
        const auto t0 = std::chrono::steady_clock::now();
        const auto p = nd(P);
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        REQUIRE(util::is_valid_permutation(p));
        REQUIRE(s < 10.0);   // generous: guards a quadratic regression, not speed
    }
}

TEST_CASE("nested_dissection: at or below the cutoff it is AMD", "[sparse][ordering][nd]") {
    const auto A = grid_laplacian(8, 2);             // 64 vertices, one component
    REQUIRE(ordering::nested_dissection{64}(A) == ordering::amd{}(A));
    REQUIRE(ordering::nested_dissection{16}(A) != ordering::amd{}(A));
}

TEST_CASE("nested_dissection: less fill and work than AMD on large grids",
          "[sparse][ordering][nd]") {
    const ordering::nested_dissection nd;

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

TEST_CASE("nested_dissection: the factorizations solve correctly under it",
          "[sparse][ordering][nd]") {
    const auto A = grid_laplacian(40, 2);            // 1600 vertices: dissected
    const std::size_t n = A.num_rows();
    vec::dense_vector<double> b(n);
    for (std::size_t i = 0; i < n; ++i) b(i) = 1.0 + static_cast<double>(i % 7);
    const ordering::nested_dissection nd;

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
