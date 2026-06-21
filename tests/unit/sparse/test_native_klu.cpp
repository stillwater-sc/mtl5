// MTL5 -- Tests for native KLU (BTF + block-wise Gilbert-Peierls LU).
// Covers steps 4-6 of the native KLU plan (issue #114): per-block factorization,
// block back-substitution, and the one-shot solve.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/generators/poisson.hpp>
#include <mtl/sparse/factorization/native_klu.hpp>
#include <mtl/sparse/factorization/sparse_lu.hpp>

#ifdef MTL5_HAS_KLU
#include <mtl/interface/klu.hpp>
#endif

using namespace mtl;
using Catch::Matchers::WithinAbs;
using mtl::sparse::factorization::native_klu_factor;
using mtl::sparse::factorization::native_klu_solve;

namespace {

template <typename Value>
double residual_inf_norm(const mat::compressed2D<Value>& A,
                         const vec::dense_vector<Value>& x,
                         const vec::dense_vector<Value>& b) {
    const auto& rp = A.ref_major();
    const auto& ci = A.ref_minor();
    const auto& dat = A.ref_data();
    double max_r = 0.0;
    for (std::size_t r = 0; r < A.num_rows(); ++r) {
        Value ax = Value{0};
        for (std::size_t k = rp[r]; k < rp[r + 1]; ++k)
            ax += dat[k] * x(static_cast<int>(ci[k]));
        max_r = std::max(max_r,
            static_cast<double>(std::abs(ax - b(static_cast<int>(r)))));
    }
    return max_r;
}

// Relative residual ||A*x - b||_inf / ||b||_inf (scale-independent).
double relative_residual(const mat::compressed2D<double>& A,
                         const vec::dense_vector<double>& x,
                         const vec::dense_vector<double>& b) {
    double bnorm = 0.0;
    for (std::size_t i = 0; i < b.size(); ++i)
        bnorm = std::max(bnorm, std::abs(b(static_cast<int>(i))));
    double r = residual_inf_norm(A, x, b);
    return (bnorm > 0.0) ? r / bnorm : r;
}

// Total nonzeros in the L and U factors across all diagonal blocks.
std::size_t factor_fill(const sparse::factorization::klu_numeric<double>& fac) {
    std::size_t fill = 0;
    for (const auto& blk : fac.block_numeric)
        fill += blk.L.nnz() + blk.U.nnz();
    return fill;
}

// Unsymmetric tridiagonal: A(i,i)=4, A(i,i-1)=-1, A(i,i+1)=-2 (irreducible).
mat::compressed2D<double> make_unsym_tridiag(std::size_t n) {
    mat::compressed2D<double> A(n, n);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        for (std::size_t i = 0; i < n; ++i) {
            if (i > 0)     ins[i][i - 1] << -1.0;
            ins[i][i] << 4.0;
            if (i + 1 < n) ins[i][i + 1] << -2.0;
        }
    }
    return A;
}

} // namespace

TEST_CASE("native KLU solves a block-triangular system", "[sparse][klu][native]") {
    // Two 2x2 diagonal blocks {0,1}, {2,3} with upper coupling A(0,3) -> 2 blocks.
    //   [ 2  1  0  5 ]
    //   [ 1  3  0  0 ]
    //   [ 0  0  4  1 ]
    //   [ 0  0  2  5 ]
    mat::compressed2D<double> A(4, 4);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 2.0; ins[0][1] << 1.0; ins[0][3] << 5.0;
        ins[1][0] << 1.0; ins[1][1] << 3.0;
        ins[2][2] << 4.0; ins[2][3] << 1.0;
        ins[3][2] << 2.0; ins[3][3] << 5.0;
    }

    auto fac = native_klu_factor(A);
    REQUIRE(fac.nblocks() == 2);

    vec::dense_vector<double> b = {1.0, 2.0, 3.0, 4.0};
    vec::dense_vector<double> x(4, 0.0);
    fac.solve(x, b);

    REQUIRE(residual_inf_norm(A, x, b) < 1e-10);
}

TEST_CASE("native KLU one-shot agrees with native sparse LU",
          "[sparse][klu][native]") {
    for (std::size_t n : {5u, 16u, 41u}) {
        auto A = make_unsym_tridiag(n);

        vec::dense_vector<double> b(n);
        for (std::size_t i = 0; i < n; ++i)
            b(static_cast<int>(i)) = static_cast<double>((i % 7) + 1);

        vec::dense_vector<double> x_klu(n, 0.0);
        native_klu_solve(A, x_klu, b);

        vec::dense_vector<double> x_lu(n, 0.0);
        sparse::factorization::sparse_lu_solve(A, x_lu, b);

        for (std::size_t i = 0; i < n; ++i)
            REQUIRE_THAT(x_klu(static_cast<int>(i)),
                         WithinAbs(x_lu(static_cast<int>(i)), 1e-9));
        REQUIRE(residual_inf_norm(A, x_klu, b) < 1e-9);
    }
}

TEST_CASE("native KLU handles a fully reducible (triangular) matrix",
          "[sparse][klu][native]") {
    // Lower-triangular with full diagonal: n singleton blocks.
    std::size_t n = 6;
    mat::compressed2D<double> A(n, n);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j <= i; ++j)
                ins[i][j] << (i == j ? 3.0 : 1.0);
        }
    }

    auto fac = native_klu_factor(A);
    REQUIRE(fac.nblocks() == n);

    vec::dense_vector<double> b(n, 1.0), x(n, 0.0);
    fac.solve(x, b);
    REQUIRE(residual_inf_norm(A, x, b) < 1e-10);
}

TEST_CASE("native KLU is generic over the value type (float)",
          "[sparse][klu][native]") {
    std::size_t n = 8;
    mat::compressed2D<float> A(n, n);
    {
        mat::inserter<mat::compressed2D<float>> ins(A);
        for (std::size_t i = 0; i < n; ++i) {
            if (i > 0)     ins[i][i - 1] << -1.0f;
            ins[i][i] << 4.0f;
            if (i + 1 < n) ins[i][i + 1] << -1.0f;
        }
    }
    vec::dense_vector<float> b(n, 1.0f), x(n, 0.0f);
    native_klu_solve(A, x, b);
    REQUIRE(residual_inf_norm(A, x, b) < 1e-4);
}

TEST_CASE("native KLU rejects non-square and structurally singular matrices",
          "[sparse][klu][native]") {
    SECTION("non-square") {
        mat::compressed2D<double> A(2, 3);
        {
            mat::inserter<mat::compressed2D<double>> ins(A);
            ins[0][0] << 1.0;
            ins[1][1] << 1.0;
        }
        REQUIRE_THROWS_AS(native_klu_factor(A), std::invalid_argument);
    }
    SECTION("structurally singular (empty column)") {
        mat::compressed2D<double> A(3, 3);
        {
            mat::inserter<mat::compressed2D<double>> ins(A);
            ins[0][0] << 1.0; ins[0][1] << 1.0;
            ins[1][0] << 1.0; ins[1][1] << 1.0;
            ins[2][0] << 1.0;            // column 2 empty
        }
        REQUIRE_THROWS_AS(native_klu_factor(A), std::runtime_error);
    }
}

TEST_CASE("native KLU on 2D Poisson scales sub-quadratically",
          "[sparse][klu][native][scaling][poisson]") {
    // 2D Poisson (5-point Laplacian) is the canonical fill-test matrix. With a
    // good ordering its factorization is O(n^1.5) flops / O(n log n) fill -- not
    // literally O(nnz), but far from the old O(n^2) blowup. As the grid refines,
    // n quadruples per step; an O(n^2) implementation would grow time ~16x per
    // step. We solve 32x32, 64x64, 128x128, 256x256 grids, check the relative
    // residual, and assert both time and factor fill grow sub-quadratically.
    struct Row { std::size_t N, n, nnz, fill; double time_ms; };
    std::vector<Row> rows;

    for (std::size_t N : {32u, 64u, 128u, 256u}) {
        auto A = generators::poisson2d_dirichlet<double>(N, N);
        std::size_t n = A.num_rows();

        // b = A * ones, exact solution all-ones.
        vec::dense_vector<double> ones(n, 1.0), b(n, 0.0);
        const auto& rp = A.ref_major();
        const auto& ci = A.ref_minor();
        const auto& dat = A.ref_data();
        for (std::size_t r = 0; r < n; ++r) {
            double s = 0.0;
            for (std::size_t k = rp[r]; k < rp[r + 1]; ++k)
                s += dat[k] * ones(static_cast<int>(ci[k]));
            b(static_cast<int>(r)) = s;
        }

        vec::dense_vector<double> x(n, 0.0);
        auto t0 = std::chrono::steady_clock::now();
        auto fac = native_klu_factor(A);
        fac.solve(x, b);
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        INFO("Poisson " << N << "x" << N << ": n=" << n << " nnz=" << A.nnz()
             << " fill=" << factor_fill(fac) << " time=" << ms << "ms");
        REQUIRE(relative_residual(A, x, b) < 1e-8);
        rows.push_back({N, n, A.nnz(), factor_fill(fac), ms});
    }

    // Surface the scaling table in the test log.
    for (const auto& r : rows)
        UNSCOPED_INFO("Poisson " << r.N << "x" << r.N << "  n=" << r.n
            << "  nnz=" << r.nnz << "  fill=" << r.fill
            << "  time=" << r.time_ms << " ms");

    // Fill is deterministic: O(n log n)-ish, so each 4x-n refinement grows fill
    // well under the 16x an O(n^2) factorization would require.
    double fill_ratio = double(rows.back().fill) / double(rows[rows.size() - 2].fill);
    REQUIRE(fill_ratio < 8.0);

    // Time growth over the last refinement (n quadruples). Measured ~7-8x
    // (O(n^1.5)); an O(n^2) factorization would be ~16x. A ratio cancels the
    // build-dependent per-op constant, so this is stable across Debug/Release/
    // sanitizer builds, unlike an absolute wall-clock ceiling. Both timings are
    // large (tens-to-hundreds of ms), so jitter is small.
    double time_ratio = rows.back().time_ms / rows[rows.size() - 2].time_ms;
    REQUIRE(time_ratio < 13.0);
}

TEST_CASE("native KLU refactor reuses structure for same-pattern matrices",
          "[sparse][klu][native][refactor]") {
    using sparse::factorization::native_klu_refactor;

    // A1 and A2 share the sparsity pattern (block-triangular circuit shape) but
    // differ in values; refactor must reuse A1's symbolic structure and produce
    // the correct solve for A2.
    auto build = [](double d) {
        std::size_t n = 6;
        mat::compressed2D<double> A(n, n);
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << d;   ins[0][1] << 1.0;  ins[0][3] << 5.0;
        ins[1][0] << 1.0; ins[1][1] << d;
        ins[2][2] << d;   ins[2][3] << 1.0;
        ins[3][2] << 2.0; ins[3][3] << d;
        ins[4][4] << d;   ins[4][5] << -1.0;
        ins[5][4] << -1.0; ins[5][5] << d;
        return A;
    };

    auto A1 = build(4.0);
    auto fac = native_klu_factor(A1);

    SECTION("refactor on the same matrix reproduces factor") {
        auto re = native_klu_refactor(A1, fac);
        vec::dense_vector<double> b(6, 1.0), xf(6, 0.0), xr(6, 0.0);
        fac.solve(xf, b);
        re.solve(xr, b);
        for (std::size_t i = 0; i < 6; ++i)
            REQUIRE_THAT(xr(static_cast<int>(i)), WithinAbs(xf(static_cast<int>(i)), 1e-12));
    }

    SECTION("refactor solves a same-pattern matrix with new values") {
        auto A2 = build(9.0);
        auto re = native_klu_refactor(A2, fac);
        vec::dense_vector<double> b = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        vec::dense_vector<double> xr(6, 0.0), xfresh(6, 0.0);
        re.solve(xr, b);
        native_klu_solve(A2, xfresh, b);
        for (std::size_t i = 0; i < 6; ++i)
            REQUIRE_THAT(xr(static_cast<int>(i)), WithinAbs(xfresh(static_cast<int>(i)), 1e-9));
        REQUIRE(relative_residual(A2, xr, b) < 1e-9);
    }
}

#ifdef MTL5_HAS_KLU
TEST_CASE("native KLU agrees with the external SuiteSparse KLU binding",
          "[sparse][klu][native][interface]") {
    for (std::size_t n : {6u, 20u}) {
        auto A = make_unsym_tridiag(n);
        vec::dense_vector<double> b(n);
        for (std::size_t i = 0; i < n; ++i)
            b(static_cast<int>(i)) = static_cast<double>((i % 5) + 1);

        vec::dense_vector<double> x_native(n, 0.0);
        native_klu_solve(A, x_native, b);

        vec::dense_vector<double> x_ext(n, 0.0);
        interface::klu_solve(A, x_ext, b);

        for (std::size_t i = 0; i < n; ++i)
            REQUIRE_THAT(x_native(static_cast<int>(i)),
                         WithinAbs(x_ext(static_cast<int>(i)), 1e-9));
    }
}
#endif
