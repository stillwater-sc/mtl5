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
#include <mtl/operation/norms.hpp>
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
// Extended (double-over-float) accumulator to exercise the accumulator policy
// threaded through native KLU (#122 seam, native_klu threading).
struct wide_acc { double v = 0.0; };
} // namespace

namespace mtl::sparse::factorization {
template <>
struct accumulator_traits<wide_acc, float> {
    static void  clear(wide_acc& a)                  { a.v = 0.0; }
    static void  assign(wide_acc& a, const float& x) { a.v = static_cast<double>(x); }
    static float value(const wide_acc& a)            { return static_cast<float>(a.v); }
    static void  add_product(wide_acc& a, const float& m, const float& x) {
        a.v += static_cast<double>(m) * static_cast<double>(x);
    }
};
} // namespace mtl::sparse::factorization

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
        fill += blk.factorL().nnz() + blk.factorU().nnz();
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
    // literally O(nnz), but far from the old O(n^2) blowup. As the grid refines
    // n quadruples per step, and an O(n^2) implementation would grow FILL ~16x
    // per step. We solve 32x32, 64x64 and 128x128 grids, check the relative
    // residual, and assert the factor fill grows sub-quadratically.
    //
    // FILL, NOT TIME. This test used to assert a wall-clock ratio as well
    // (#454). It could not be stable: a ratio of two single timings, against a
    // threshold only ~2x above the expected value, in the tier that gates every
    // PR on shared cloud runners. Sampled three times on an IDLE machine the
    // statistic itself moved 31% (5.24, 5.91, 6.87), and CI produced 15.0
    // against a limit of 13.0 on a PR that changed no C++ at all.
    //
    // Nothing algorithmic was lost by removing it, because fill is the
    // DETERMINISTIC observable of the same property: a sparse factorization's
    // asymptotic cost is driven by fill growth, and the clock measures fill plus
    // noise. The timing claim now lives in the regression tier
    // (tests/regression/sparse/test_klu_scaling.cpp), which is gated, runs on
    // one controlled platform, and takes a minimum of several samples per size.
    //
    // Dropping the 256x256 grid with it takes this test from ~10 s to ~1.5 s.
    // The fill ratio is a per-refinement quantity, so 64 -> 128 tests it exactly
    // as well as 128 -> 256 did.
    struct Row { std::size_t N, n, nnz, fill; double time_ms; };
    std::vector<Row> rows;

    for (std::size_t N : {32u, 64u, 128u}) {
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
    // well under the 16x an O(n^2) factorization would require. No clock is
    // involved, so this holds identically on a loaded runner.
    double fill_ratio = double(rows.back().fill) / double(rows[rows.size() - 2].fill);
    REQUIRE(fill_ratio < 8.0);
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

TEST_CASE("native KLU threads a custom accumulator policy through the blocks",
          "[sparse][klu][native][accumulator]") {
    // Block-triangular float circuit matrix; factor it with the default (float)
    // accumulator and with the extended double accumulator. The wide accumulator
    // must thread through BTF + every block and solve at least as accurately.
    std::size_t n = 6;
    mat::compressed2D<float> A(n, n);
    {
        mat::inserter<mat::compressed2D<float>> ins(A);
        ins[0][0] << 3.0f;  ins[0][1] << -1.0f; ins[0][5] << -2.0f;
        ins[1][0] << -1.0f; ins[1][1] << 3.0f;  ins[1][2] << -1.0f;
        ins[2][1] << -1.0f; ins[2][2] << 3.0f;
        ins[3][3] << 4.0f;  ins[3][4] << -1.0f;
        ins[4][3] << -1.0f; ins[4][4] << 4.0f;  ins[4][5] << -1.0f;
        ins[5][4] << -1.0f; ins[5][5] << 4.0f;
    }
    vec::dense_vector<float> b(n, 1.0f), xd(n, 0.0f), xw(n, 0.0f);

    auto fac_default = native_klu_factor(A);                                  // Accumulator = float
    fac_default.solve(xd, b);

    auto fac_wide = native_klu_factor<wide_acc>(A); // double accumulation
    fac_wide.solve(xw, b);

    // Residual ||A x - b||_inf in double for each.
    auto resid = [&](const vec::dense_vector<float>& x) {
        const auto& rp = A.ref_major(); const auto& ci = A.ref_minor(); const auto& dat = A.ref_data();
        double m = 0.0;
        for (std::size_t r = 0; r < n; ++r) {
            double ax = 0.0;
            for (std::size_t k = rp[r]; k < rp[r + 1]; ++k)
                ax += static_cast<double>(dat[k]) * static_cast<double>(x(static_cast<int>(ci[k])));
            m = std::max(m, std::abs(ax - static_cast<double>(b(static_cast<int>(r)))));
        }
        return m;
    };
    // Both solve correctly; the extended accumulator is no worse.
    REQUIRE(resid(xd) < 1e-4);
    REQUIRE(resid(xw) <= resid(xd) + 1e-12);
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

// Issue #123: opt-in zero-pivot perturbation threaded through the native-KLU driver.
TEST_CASE("native KLU zero-pivot perturbation (#123)", "[sparse][klu][native][perturb]") {
    // A numerically singular diagonal block (structurally fine: BTF matches it).
    mtl::mat::compressed2D<double> A(2, 2);
    { mtl::mat::inserter<mtl::mat::compressed2D<double>> ins(A);
      ins[0][0] << 1.0; ins[0][1] << 1.0;
      ins[1][0] << 1.0; ins[1][1] << 1.0; }

    // Default (perturbation off): hard throw on the singular block.
    REQUIRE_THROWS_AS(native_klu_factor(A), std::runtime_error);

    // Perturbation on: completes; the perturbed pivots are reported.
    auto fac = native_klu_factor(A, /*threshold=*/1.0, /*scale=*/true, /*pivot_perturb=*/1e-8);
    REQUIRE(fac.num_perturbed >= 1);
    mtl::vec::dense_vector<double> b = {1.0, 2.0}, x(2, 0.0);
    fac.solve(x, b);
    REQUIRE(std::isfinite(x(0)));
    REQUIRE(std::isfinite(x(1)));

    // A well-conditioned matrix never perturbs, even with perturbation enabled.
    mtl::mat::compressed2D<double> G(3, 3);
    { mtl::mat::inserter<mtl::mat::compressed2D<double>> ins(G);
      ins[0][0] << 4.0; ins[0][1] << 1.0;
      ins[1][0] << 1.0; ins[1][1] << 5.0; ins[1][2] << 2.0;
      ins[2][1] << 1.0; ins[2][2] << 6.0; }
    auto gfac = native_klu_factor(G, 1.0, true, 1e-8);
    REQUIRE(gfac.num_perturbed == 0);
    mtl::vec::dense_vector<double> gb = {1.0, 2.0, 3.0}, gx(3, 0.0);
    gfac.solve(gx, gb);
    REQUIRE(relative_residual(G, gx, gb) < 1e-12);
}

// 1x1 blocks take their own path in solve and refactor (issue 908): the
// triangular solves reduce to (b / L(0,0)) / U(0,0), and the refactor to L =
// {1}, U = {a}. A bidiagonal matrix is all 1x1 blocks; a mixed one has 1x1
// and larger blocks. Each is checked against the general per-block path,
// lu_numeric::solve on the block, through the solution of a fresh factor
// (bit for bit: the same divisions), and the refactor against a fresh factor.
TEST_CASE("native_klu: 1x1 blocks in solve and refactor (issue 908)", "[sparse][klu][908]") {
    using csr = mat::compressed2D<double>;
    const std::size_t n = 40;
    auto make = [&](double shift, bool mixed) {
        csr A(n, n);
        {   // the inserter commits its entries in its destructor, before the return
            mat::inserter<csr> ins(A, 4);
            for (std::size_t i = 0; i < n; ++i) {
                ins[i][i] << 3.0 + shift + double(i % 5);
                if (i + 1 < n) ins[i][i + 1] << -1.0;
                if (mixed && i % 7 == 3 && i + 1 < n) ins[i + 1][i] << 0.5;   // a 2x2 block
            }
        }
        return A;
    };
    for (const bool mixed : {false, true}) {
        CAPTURE(mixed);
        const csr A = make(0.0, mixed);
        REQUIRE(A.nnz() == 2 * n - 1 + (mixed ? 6 : 0));   // the fixture is the intended system
        auto fac = native_klu_factor(A);
        if (!mixed) REQUIRE(fac.nblocks() == n);
        vec::dense_vector<double> b(n), x(n), y(n);
        for (std::size_t i = 0; i < n; ++i) b(i) = 1.0 + double(i % 3);
        fac.solve(x, b);
        // The block back-substitution, written out with lu_numeric::solve on
        // every block: the general path.
        {
            std::vector<double> w(n);
            for (std::size_t i = 0; i < n; ++i) w[i] = b(fac.btf.row_perm[i]) * (fac.row_scale.empty() ? 1.0 : fac.row_scale[fac.btf.row_perm[i]]);
            const auto& rp = fac.B.ref_major(); const auto& ci = fac.B.ref_minor(); const auto& d = fac.B.ref_data();
            for (std::size_t blk = fac.nblocks(); blk-- > 0; ) {
                const std::size_t r0 = fac.btf.blocks[blk], r1 = fac.btf.blocks[blk + 1];
                vec::dense_vector<double> rhs(r1 - r0), sol(r1 - r0);
                for (std::size_t i = r0; i < r1; ++i) {
                    double v = w[i];
                    for (std::size_t k = rp[i]; k < rp[i + 1]; ++k)
                        if (fac.block_of[ci[k]] > blk) v -= d[k] * w[ci[k]];
                    rhs(i - r0) = v;
                }
                fac.block_numeric[blk].solve(sol, rhs);
                for (std::size_t i = r0; i < r1; ++i) w[i] = sol(i - r0);
            }
            for (std::size_t j = 0; j < n; ++j) y(fac.btf.col_perm[j]) = w[j];
        }
        for (std::size_t i = 0; i < n; ++i) REQUIRE(x(i) == y(i));

        // Refactor with new values on the same pattern: the same solution as
        // a fresh factor of the new matrix.
        const csr A2 = make(1.5, mixed);
        const auto re = sparse::factorization::native_klu_refactor(A2, fac);
        const auto fresh = native_klu_factor(A2);
        vec::dense_vector<double> xr(n), xf(n);
        re.solve(xr, b);
        fresh.solve(xf, b);
        for (std::size_t i = 0; i < n; ++i) REQUIRE(xr(i) == xf(i));
        for (const auto& bn : re.block_numeric) REQUIRE(bn.num_perturbed == 0);
    }
    // A refactor whose 1x1 entry became 0 takes the general path, and its error.
    csr Z(2, 2);
    { mat::inserter<csr> ins(Z); ins[0][0] << 2.0; ins[0][1] << 1.0; ins[1][1] << 3.0; }
    const auto fz = native_klu_factor(Z);
    csr Z0(2, 2);
    { mat::inserter<csr> ins(Z0); ins[0][0] << 2.0; ins[0][1] << 1.0; ins[1][1] << 0.0; }
    REQUIRE_THROWS_AS(sparse::factorization::native_klu_refactor(Z0, fz), std::runtime_error);
}

// lu_numeric::solve with the caller's workspace gives the same bits as the
// one that allocates its own, and reuses the buffer across sizes (issue 908).
TEST_CASE("lu_numeric::solve with a caller workspace (issue 908)", "[sparse][lu][908]") {
    using csr = mat::compressed2D<double>;
    std::vector<double> w;
    for (const std::size_t n : {std::size_t{9}, std::size_t{3}, std::size_t{17}}) {
        csr A(n, n);
        {
            mat::inserter<csr> ins(A, 4);
            for (std::size_t i = 0; i < n; ++i) {
                ins[i][i] << 4.0 + double(i % 3);
                if (i > 0) ins[i][i - 1] << -1.25;
                if (i + 2 < n) ins[i][i + 2] << 0.5;
            }
        }
        const auto sym = sparse::factorization::sparse_lu_symbolic(A);
        const auto lu = sparse::factorization::sparse_lu_numeric(A, sym);
        vec::dense_vector<double> b(n), x1(n), x2(n);
        for (std::size_t i = 0; i < n; ++i) b(i) = double(i) - 2.5;
        lu.solve(x1, b);
        lu.solve(x2, b, w);
        REQUIRE(w.size() == n);
        for (std::size_t i = 0; i < n; ++i) REQUIRE(x1(i) == x2(i));
    }
}

// Issue 946: a refactorization that keeps a pivot the new values made tiny,
// but not zero, returns normally and solves wrongly. rgrowth() and rcond()
// are how a caller finds out, as klu_rgrowth and klu_rcond in SuiteSparse.
namespace {
using csr946 = mat::compressed2D<double>;

// [[a00, 1], [1, 1]], one irreducible 2 x 2 block.
csr946 two_by_two(double a00) {
    csr946 A(2, 2);
    {
        mat::inserter<csr946> ins(A);
        ins[0][0] << a00; ins[0][1] << 1.0; ins[1][0] << 1.0; ins[1][1] << 1.0;
    }   // finalized before the return, elided or not
    return A;
}

// [[5, 1, 0], [0, a11, 1], [0, 1, 1]]: a 1 x 1 block, then the 2 x 2 one.
csr946 two_blocks(double a11) {
    csr946 A(3, 3);
    {
        mat::inserter<csr946> ins(A);
        ins[0][0] << 5.0; ins[0][1] << 1.0;
        ins[1][1] << a11; ins[1][2] << 1.0;
        ins[2][1] << 1.0; ins[2][2] << 1.0;
    }
    return A;
}

double max_error_from_ones(const vec::dense_vector<double>& x) {
    double e = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) e = std::max(e, std::abs(x(i) - 1.0));
    return e;
}
} // namespace

TEST_CASE("native_klu: rgrowth and rcond expose a refactor on stale pivots (issue 946)",
          "[sparse][klu][946]") {
    using sparse::factorization::native_klu_refactor;

    SECTION("the issue's 2 x 2 repro") {
        const auto A1 = two_by_two(2.0), A2 = two_by_two(1e-20);
        vec::dense_vector<double> b(2), x(2, 0.0);
        b(0) = 1e-20 + 1.0; b(1) = 2.0;   // x = (1, 1)

        const auto fac = native_klu_factor(A1);
        CHECK(fac.rgrowth() > 0.1);
        CHECK(fac.rcond() > 0.1);

        const auto re = native_klu_refactor(A2, fac);
        re.solve(x, b);
        REQUIRE(re.num_perturbed == 0);              // nothing else says so
        REQUIRE(max_error_from_ones(x) > 0.5);       // the solve is wrong
        CHECK(re.rgrowth() < 1e-15);
        CHECK(re.rcond() < 1e-30);

        const auto fresh = native_klu_factor(A2);
        fresh.solve(x, b);
        REQUIRE(max_error_from_ones(x) < 1e-15);
        CHECK(fresh.rgrowth() > 0.1);
        CHECK(fresh.rcond() > 0.1);
    }

    SECTION("values perturbed by 1% keep the growth of a fresh factorization") {
        const auto fac = native_klu_factor(two_by_two(2.0));
        const auto re  = native_klu_refactor(two_by_two(2.02), fac);
        CHECK(re.rgrowth() > 0.1);
        CHECK(re.rgrowth() <= 1.0);
    }

    SECTION("the bad block need not be the first: the minimum is over all blocks") {
        const auto fac = native_klu_factor(two_blocks(2.0));
        REQUIRE(fac.nblocks() == 2);
        CHECK(fac.rgrowth() > 0.1);
        const auto re = native_klu_refactor(two_blocks(1e-20), fac);
        CHECK(re.rgrowth() < 1e-15);
        CHECK(re.rcond() < 1e-30);
        CHECK(native_klu_factor(two_blocks(1e-20)).rgrowth() > 0.1);
    }

    SECTION("a NaN in the values is reported as NaN, not as a good growth") {
        const auto fac = native_klu_factor(two_blocks(2.0));
        const auto re  = native_klu_refactor(two_blocks(std::nan("")), fac);
        CHECK(std::isnan(re.rgrowth()));
        CHECK(std::isnan(re.rcond()));
    }

    SECTION("a NaN in a coupling between blocks is reported too") {
        // [[1, c], [0, 1]]: two 1 x 1 blocks, c above them. solve() reads c.
        auto upper = [](double c) {
            csr946 A(2, 2);
            {
                mat::inserter<csr946> ins(A);
                ins[0][0] << 1.0; ins[0][1] << c; ins[1][1] << 1.0;
            }
            return A;
        };
        const auto fac = native_klu_factor(upper(1.0));
        REQUIRE(fac.nblocks() == 2);
        CHECK(fac.rgrowth() == 1.0);
        CHECK(std::isnan(native_klu_refactor(upper(std::nan("")), fac).rgrowth()));
    }

    SECTION("an empty matrix") {
        const auto fac = native_klu_factor(csr946(0, 0));
        CHECK(fac.rgrowth() == 1.0);
        CHECK(fac.rcond() == 1.0);
    }
}

TEST_CASE("lu_numeric: rgrowth and rcond, by hand (issue 946)", "[sparse][lu][946]") {
    namespace fz = sparse::factorization;
    // Natural order, pivot on 2: U = [[2, 1], [0, 0.5]]. Every column's largest
    // entry is A's own, so the growth is 1, and rcond = 0.5 / 2.
    const auto A1 = two_by_two(2.0);
    auto lu = fz::sparse_lu_numeric(A1, fz::sparse_lu_symbolic(A1));
    REQUIRE(lu.row_perm[0] == 0);
    CHECK(lu.rgrowth(A1) == 1.0);
    CHECK(lu.rcond() == 0.25);

    // Refactored with 1e-20 in the pivot's place: U = [[1e-20, 1], [0, 1 - 1e20]].
    // Column 1: max|A| = 1 over max|U| = 1e20. rcond = 1e-20 / 1e20.
    const auto A2 = two_by_two(1e-20);
    fz::sparse_lu_refactor_in_place(A2, lu);
    CHECK_THAT(lu.rgrowth(A2), Catch::Matchers::WithinRel(1e-20, 1e-12));
    CHECK_THAT(lu.rcond(), Catch::Matchers::WithinRel(1e-40, 1e-12));

    REQUIRE_THROWS_AS(lu.rgrowth(csr946(3, 3)), std::invalid_argument);

    fz::lu_numeric<double> none;
    none.symbolic.n = 2;
    REQUIRE_THROWS_AS(none.rcond(), std::logic_error);
    REQUIRE_THROWS_AS(none.rgrowth(A1), std::logic_error);
}

// Issue 946's optional items: solve_transpose and condest (Hager-Higham, as
// klu_condest), and native_klu_refactor's opt-in growth threshold.
namespace {
// (A^T x)_j = sum_i A(i, j) x_i, from A's CSR arrays.
vec::dense_vector<double> times_transpose(const csr946& A, const vec::dense_vector<double>& x) {
    vec::dense_vector<double> y(A.num_cols(), 0.0);
    const auto& rp = A.ref_major();
    const auto& ci = A.ref_minor();
    const auto& dat = A.ref_data();
    for (std::size_t i = 0; i < A.num_rows(); ++i)
        for (std::size_t k = rp[i]; k < rp[i + 1]; ++k) y(ci[k]) += dat[k] * x(i);
    return y;
}

// Unsymmetric, reducible: three diagonal blocks (2 x 2, 1 x 1, 3 x 3) with
// coupling above them, and rows of very different scale.
csr946 three_blocks() {
    csr946 A(6, 6);
    {
        mat::inserter<csr946> ins(A);
        ins[0][0] << 1e-3; ins[0][1] << 2.0;  ins[0][3] << 5.0;
        ins[1][0] << 3.0;  ins[1][1] << 1.0;  ins[1][2] << -1.0;
        ins[2][2] << 7.0;  ins[2][4] << 1.0;
        ins[3][3] << 4.0e3; ins[3][4] << -1.0e3; ins[3][5] << 2.0e3;
        ins[4][3] << 1.0;  ins[4][4] << 3.0;
        ins[5][4] << -2.0; ins[5][5] << 0.5;  ins[5][3] << 1.0;
    }
    return A;
}

// The exact ||A||_1 ||A^-1||_1, through n solves for the columns of A^-1.
template <typename Fac>
double exact_cond1(const csr946& A, const Fac& fac) {
    const std::size_t n = A.num_rows();
    double inv = 0.0;
    for (std::size_t j = 0; j < n; ++j) {
        vec::dense_vector<double> e(n, 0.0), c(n, 0.0);
        e(j) = 1.0;
        fac.solve(c, e);
        double t = 0.0;
        for (std::size_t i = 0; i < n; ++i) t += std::abs(c(i));
        inv = std::max(inv, t);
    }
    return sparse::factorization::detail::csr_norm1(A) * inv;
}
} // namespace

TEST_CASE("solve_transpose: A^T x = b through lu_numeric and klu_numeric (issue 946)", "[sparse][klu][946]") {
    namespace fz = sparse::factorization;
    for (const auto& A : {three_blocks(), make_unsym_tridiag(12), two_by_two(1e-20)}) {
        const std::size_t n = A.num_rows();
        vec::dense_vector<double> b(n);
        for (std::size_t i = 0; i < n; ++i) b(i) = 1.0 + double(i % 4) - 0.25 * double(i);
        for (const bool scale : {true, false}) {
            const auto fac = native_klu_factor(A, 1.0, scale);
            vec::dense_vector<double> x(n, 0.0);
            fac.solve_transpose(x, b);
            const auto r = times_transpose(A, x);
            double err = 0.0;
            for (std::size_t i = 0; i < n; ++i) err = std::max(err, std::abs(r(i) - b(i)));
            CHECK(err < 1e-12 * (1.0 + mtl::infinity_norm(b)) * fac.condest(A));
            // In place, as solve allows.
            vec::dense_vector<double> y = b;
            fac.solve_transpose(y, y);
            for (std::size_t i = 0; i < n; ++i) CHECK(y(i) == x(i));
        }
        const auto lu = fz::sparse_lu_numeric(A, fz::sparse_lu_symbolic(A));
        vec::dense_vector<double> x(n, 0.0);
        lu.solve_transpose(x, b);
        const auto r = times_transpose(A, x);
        for (std::size_t i = 0; i < n; ++i) CHECK(std::abs(r(i) - b(i)) < 1e-9 * (1.0 + std::abs(b(i))));
    }
    vec::dense_vector<double> wrong(3);
    REQUIRE_THROWS_AS(native_klu_factor(three_blocks()).solve_transpose(wrong, wrong), std::invalid_argument);
}

TEST_CASE("condest: a lower bound of cond_1, and near it (issue 946)", "[sparse][klu][946]") {
    namespace fz = sparse::factorization;
    csr946 D(4, 4);   // diag(1, 1e-8, 3, 1): cond_1 = 3e8 exactly
    {
        mat::inserter<csr946> ins(D);
        ins[0][0] << 1.0; ins[1][1] << 1e-8; ins[2][2] << 3.0; ins[3][3] << 1.0;
    }
    std::vector<csr946> cases{three_blocks(), make_unsym_tridiag(20), two_blocks(2.0), D};
    {
        auto P = generators::poisson2d_dirichlet<double>(8, 8);
        cases.push_back(P);
    }
    for (const auto& A : cases) {
        const auto fac = native_klu_factor(A);
        const double exact = exact_cond1(A, fac);
        const double est = fac.condest(A);
        CAPTURE(A.num_rows(), exact, est);
        CHECK(est <= exact * (1.0 + 1e-10));
        CHECK(est >= exact / 3.0);
        const auto lu = fz::sparse_lu_numeric(A, fz::sparse_lu_symbolic(A));
        const double lest = lu.condest(A);
        CHECK(lest <= exact * (1.0 + 1e-10));
        CHECK(lest >= exact / 3.0);
    }
    CHECK_THAT(native_klu_factor(D).condest(D), Catch::Matchers::WithinRel(3e8, 1e-12));
    CHECK(native_klu_factor(csr946(0, 0)).condest(csr946(0, 0)) == 1.0);
    REQUIRE_THROWS_AS(native_klu_factor(D).condest(csr946(3, 3)), std::invalid_argument);
    const auto lu = fz::sparse_lu_numeric(D, fz::sparse_lu_symbolic(D));
    REQUIRE_THROWS_AS(lu.condest(csr946(3, 3)), std::invalid_argument);
}

TEST_CASE("native_klu_refactor: the opt-in growth threshold (issue 946)", "[sparse][klu][946]") {
    using sparse::factorization::klu_growth_error;
    using sparse::factorization::native_klu_refactor;
    const auto fac = native_klu_factor(two_by_two(2.0));

    // The issue's stale pivot: rejected, with the measured growth.
    try {
        (void)native_klu_refactor(two_by_two(1e-20), fac, {.min_rgrowth = 1e-8});
        FAIL("no klu_growth_error");
    } catch (const klu_growth_error& e) {
        CHECK(e.rgrowth < 1e-15);
        CHECK(std::string(e.what()).find("factor afresh") != std::string::npos);
    }
    // A NaN growth is rejected too.
    REQUIRE_THROWS_AS(native_klu_refactor(two_by_two(std::nan("")), fac, {.min_rgrowth = 1e-8}), klu_growth_error);
    // Values that still fit pass, and the result is the unchecked one.
    const auto ok = native_klu_refactor(two_by_two(2.02), fac, {.min_rgrowth = 1e-8});
    CHECK(ok.rgrowth() > 0.1);
    // 0, the default, checks nothing: the stale refactorization is returned.
    const auto unchecked = native_klu_refactor(two_by_two(1e-20), fac, {});
    CHECK(unchecked.rgrowth() < 1e-15);
    // prev is untouched by a rejection: it still solves A1 x = b.
    vec::dense_vector<double> b(2), x(2, 0.0);
    b(0) = 3.0; b(1) = 2.0;   // A1 = [[2, 1], [1, 1]], x = (1, 1)
    fac.solve(x, b);
    CHECK_THAT(x(0), WithinAbs(1.0, 1e-15));
    CHECK_THAT(x(1), WithinAbs(1.0, 1e-15));
}

// Review of #956: condest is independent of A's scale. Unscaled, ||A||_1
// overflowed for A = 1e308 [[1, 1], [1, -1]] (cond_1 = 2), and 2 ||y||_1
// overflowed in the last probe for A = 2^-1022 I (cond_1 = 1).
TEST_CASE("condest: a matrix at the ends of double's range (issue 946)", "[sparse][klu][946]") {
    namespace fz = sparse::factorization;
    auto scaled = [](const csr946& A, double s) {
        csr946 B(A.num_rows(), A.num_cols());
        {
            mat::inserter<csr946> ins(B);
            const auto& rp = A.ref_major();
            const auto& ci = A.ref_minor();
            const auto& dat = A.ref_data();
            for (std::size_t i = 0; i < A.num_rows(); ++i)
                for (std::size_t k = rp[i]; k < rp[i + 1]; ++k) ins[i][ci[k]] << s * dat[k];
        }
        return B;
    };
    auto both = [](const csr946& A) {
        const auto lu = fz::sparse_lu_numeric(A, fz::sparse_lu_symbolic(A));
        return std::pair{native_klu_factor(A).condest(A), lu.condest(A)};
    };
    csr946 H(2, 2);
    {
        mat::inserter<csr946> ins(H);
        ins[0][0] << 1e308; ins[0][1] << 1e308; ins[1][0] << 1e308; ins[1][1] << -1e308;
    }
    const auto [kh, lh] = both(H);
    CHECK_THAT(kh, Catch::Matchers::WithinRel(2.0, 1e-12));
    CHECK_THAT(lh, Catch::Matchers::WithinRel(2.0, 1e-12));

    csr946 T(2, 2);
    {
        mat::inserter<csr946> ins(T);
        ins[0][0] << 0x1p-1022; ins[1][1] << 0x1p-1022;
    }
    const auto [kt, lt] = both(T);
    CHECK(kt == 1.0);
    CHECK(lt == 1.0);

    // An exact power-of-two scaling changes nothing.
    const auto A = three_blocks();
    const auto [k1, l1] = both(A);
    for (const double s : {0x1p600, 0x1p-600}) {
        const auto [ks, ls] = both(scaled(A, s));
        CHECK_THAT(ks, Catch::Matchers::WithinRel(k1, 1e-12));
        CHECK_THAT(ls, Catch::Matchers::WithinRel(l1, 1e-12));
    }
}
