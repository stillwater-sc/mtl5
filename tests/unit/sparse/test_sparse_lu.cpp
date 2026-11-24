#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/sparse/factorization/sparse_lu.hpp>
#include <mtl/sparse/ordering/colamd.hpp>
#include <mtl/sparse/ordering/rcm.hpp>

using namespace mtl;
using namespace mtl::sparse;

// Helper: compute relative residual ||Ax - b|| / ||b||
static double relative_residual(
    const mat::compressed2D<double>& A,
    const vec::dense_vector<double>& x,
    const vec::dense_vector<double>& b)
{
    std::size_t n = b.size();
    double res_norm = 0.0;
    double b_norm = 0.0;
    const auto& starts = A.ref_major();
    const auto& indices = A.ref_minor();
    const auto& data = A.ref_data();
    for (std::size_t i = 0; i < n; ++i) {
        double Ax_i = 0.0;
        for (std::size_t k = starts[i]; k < starts[i + 1]; ++k)
            Ax_i += data[k] * x(indices[k]);
        double ri = Ax_i - b(i);
        res_norm += ri * ri;
        b_norm += b(i) * b(i);
    }
    if (b_norm == 0.0) return std::sqrt(res_norm);
    return std::sqrt(res_norm) / std::sqrt(b_norm);
}

TEST_CASE("Sparse LU symbolic analysis", "[sparse][lu]") {
    mat::compressed2D<double> A(3, 3);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 2.0; ins[0][1] << 1.0;
        ins[1][0] << 1.0; ins[1][1] << 3.0; ins[1][2] << 1.0;
        ins[2][1] << 1.0; ins[2][2] << 2.0;
    }

    auto sym = factorization::sparse_lu_symbolic(A);

    REQUIRE(sym.n == 3);
    REQUIRE(sym.col_perm.size() == 3);
    REQUIRE(sym.col_pinv.size() == 3);
}

TEST_CASE("Sparse LU solve: 3x3 symmetric system", "[sparse][lu]") {
    // A = [[4 -1  0]
    //      [-1  4 -1]
    //      [0  -1  4]]
    mat::compressed2D<double> A(3, 3);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 4.0;  ins[0][1] << -1.0;
        ins[1][0] << -1.0; ins[1][1] << 4.0;  ins[1][2] << -1.0;
        ins[2][1] << -1.0; ins[2][2] << 4.0;
    }

    vec::dense_vector<double> b = {1.0, 2.0, 3.0};
    vec::dense_vector<double> x(3, 0.0);

    factorization::sparse_lu_solve(A, x, b);

    double rr = relative_residual(A, x, b);
    REQUIRE(rr < 1e-12);
}

TEST_CASE("Sparse LU solve: 3x3 unsymmetric system", "[sparse][lu]") {
    // Unsymmetric matrix:
    // A = [[3  1  0]
    //      [0  4  2]
    //      [1  0  5]]
    mat::compressed2D<double> A(3, 3);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 3.0; ins[0][1] << 1.0;
        ins[1][1] << 4.0; ins[1][2] << 2.0;
        ins[2][0] << 1.0; ins[2][2] << 5.0;
    }

    vec::dense_vector<double> b = {4.0, 10.0, 11.0};
    vec::dense_vector<double> x(3, 0.0);

    factorization::sparse_lu_solve(A, x, b);

    double rr = relative_residual(A, x, b);
    REQUIRE(rr < 1e-12);
}

TEST_CASE("Sparse LU solve: requires pivoting", "[sparse][lu]") {
    // Matrix where natural ordering needs pivoting:
    // A = [[0  1]
    //      [1  1]]
    // Without pivoting, first pivot is 0 -> singular.
    // With pivoting, swap rows -> works.
    mat::compressed2D<double> A(2, 2);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][1] << 1.0;
        ins[1][0] << 1.0; ins[1][1] << 1.0;
    }

    vec::dense_vector<double> b = {3.0, 4.0};
    vec::dense_vector<double> x(2, 0.0);

    factorization::sparse_lu_solve(A, x, b);

    double rr = relative_residual(A, x, b);
    REQUIRE(rr < 1e-12);
}

TEST_CASE("Sparse LU solve: 5x5 tridiagonal", "[sparse][lu]") {
    std::size_t n = 5;
    mat::compressed2D<double> A(n, n);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        for (std::size_t i = 0; i < n; ++i) {
            ins[i][i] << 4.0;
            if (i + 1 < n) {
                ins[i][i + 1] << -1.0;
                ins[i + 1][i] << -1.0;
            }
        }
    }

    vec::dense_vector<double> b = {1.0, 0.0, -1.0, 2.0, 0.5};
    vec::dense_vector<double> x(n, 0.0);

    factorization::sparse_lu_solve(A, x, b);

    double rr = relative_residual(A, x, b);
    REQUIRE(rr < 1e-12);
}

TEST_CASE("Sparse LU solve with RCM ordering", "[sparse][lu]") {
    std::size_t n = 5;
    mat::compressed2D<double> A(n, n);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        for (std::size_t i = 0; i < n; ++i) {
            ins[i][i] << 4.0;
            if (i + 1 < n) {
                ins[i][i + 1] << -1.0;
                ins[i + 1][i] << -1.0;
            }
        }
    }

    vec::dense_vector<double> b = {1.0, 2.0, 3.0, 4.0, 5.0};
    vec::dense_vector<double> x(n, 0.0);

    factorization::sparse_lu_solve(A, x, b, ordering::rcm{});

    double rr = relative_residual(A, x, b);
    REQUIRE(rr < 1e-12);
}

TEST_CASE("Sparse LU solve: dense unsymmetric via sparse", "[sparse][lu]") {
    // Fully dense 3x3 unsymmetric:
    // A = [[2  3  1]
    //      [4  7  5]
    //      [6  18 22]]
    mat::compressed2D<double> A(3, 3);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 2.0;  ins[0][1] << 3.0;  ins[0][2] << 1.0;
        ins[1][0] << 4.0;  ins[1][1] << 7.0;  ins[1][2] << 5.0;
        ins[2][0] << 6.0;  ins[2][1] << 18.0; ins[2][2] << 22.0;
    }

    vec::dense_vector<double> b = {6.0, 16.0, 46.0};
    vec::dense_vector<double> x(3, 0.0);

    factorization::sparse_lu_solve(A, x, b);

    double rr = relative_residual(A, x, b);
    REQUIRE(rr < 1e-12);
}

TEST_CASE("Sparse LU solve: 1x1 matrix", "[sparse][lu]") {
    mat::compressed2D<double> A(1, 1);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 5.0;
    }

    vec::dense_vector<double> b = {15.0};
    vec::dense_vector<double> x(1, 0.0);

    factorization::sparse_lu_solve(A, x, b);

    REQUIRE_THAT(x(0), Catch::Matchers::WithinAbs(3.0, 1e-12));
}

TEST_CASE("Sparse LU throws for singular matrix", "[sparse][lu]") {
    // Singular: [[1 1], [1 1]]
    mat::compressed2D<double> A(2, 2);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 1.0; ins[0][1] << 1.0;
        ins[1][0] << 1.0; ins[1][1] << 1.0;
    }

    auto sym = factorization::sparse_lu_symbolic(A);
    REQUIRE_THROWS_AS(
        factorization::sparse_lu_numeric(A, sym),
        std::runtime_error);
}

TEST_CASE("Sparse LU with threshold pivoting", "[sparse][lu]") {
    // A = [[0.6  1]
    //      [1.0  1]]
    // With threshold=1.0 (full pivoting): |0.6| < 1.0*|1.0|, swaps rows
    // With threshold=0.5: |0.6| >= 0.5*|1.0|, does NOT swap rows
    mat::compressed2D<double> A(2, 2);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 0.6; ins[0][1] << 1.0;
        ins[1][0] << 1.0; ins[1][1] << 1.0;
    }

    auto sym = factorization::sparse_lu_symbolic(A);

    // Full pivoting: should swap rows (0.6 < 1.0*1.0)
    auto full = factorization::sparse_lu_numeric(A, sym, 1.0);
    // Relaxed threshold: should NOT swap rows (0.6 >= 0.5*1.0)
    auto relaxed = factorization::sparse_lu_numeric(A, sym, 0.5);

    // Verify different pivot choices
    REQUIRE(full.row_perm != relaxed.row_perm);

    // Both should still produce correct solves
    vec::dense_vector<double> b = {1.6, 2.0};

    vec::dense_vector<double> x1(2, 0.0);
    full.solve(x1, b);
    REQUIRE(relative_residual(A, x1, b) < 1e-12);

    vec::dense_vector<double> x2(2, 0.0);
    relaxed.solve(x2, b);
    REQUIRE(relative_residual(A, x2, b) < 1e-12);
}

TEST_CASE("Sparse LU solve: larger 10x10 system", "[sparse][lu]") {
    std::size_t n = 10;
    mat::compressed2D<double> A(n, n);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        for (std::size_t i = 0; i < n; ++i) {
            ins[i][i] << 4.0;
            if (i + 1 < n) {
                ins[i][i + 1] << -1.0;
                ins[i + 1][i] << -1.0;
            }
        }
        // Add some unsymmetric entries
        ins[0][n - 1] << 0.5;
        ins[n - 1][0] << -0.5;
    }

    vec::dense_vector<double> b(n, 1.0);
    vec::dense_vector<double> x(n, 0.0);

    factorization::sparse_lu_solve(A, x, b, ordering::rcm{});

    double rr = relative_residual(A, x, b);
    REQUIRE(rr < 1e-12);
}

TEST_CASE("Sparse LU symbolic reuse for same pattern", "[sparse][lu]") {
    mat::compressed2D<double> A1(3, 3);
    {
        mat::inserter<mat::compressed2D<double>> ins(A1);
        ins[0][0] << 2.0; ins[0][1] << 1.0;
        ins[1][0] << 1.0; ins[1][1] << 3.0; ins[1][2] << 1.0;
        ins[2][1] << 1.0; ins[2][2] << 2.0;
    }

    mat::compressed2D<double> A2(3, 3);
    {
        mat::inserter<mat::compressed2D<double>> ins(A2);
        ins[0][0] << 5.0; ins[0][1] << 2.0;
        ins[1][0] << 2.0; ins[1][1] << 8.0; ins[1][2] << 3.0;
        ins[2][1] << 3.0; ins[2][2] << 6.0;
    }

    auto sym = factorization::sparse_lu_symbolic(A1);

    vec::dense_vector<double> b = {1.0, 2.0, 3.0};

    auto num1 = factorization::sparse_lu_numeric(A1, sym);
    vec::dense_vector<double> x1(3, 0.0);
    num1.solve(x1, b);
    REQUIRE(relative_residual(A1, x1, b) < 1e-12);

    auto num2 = factorization::sparse_lu_numeric(A2, sym);
    vec::dense_vector<double> x2(3, 0.0);
    num2.solve(x2, b);
    REQUIRE(relative_residual(A2, x2, b) < 1e-12);
}

// Issue #123: opt-in zero-pivot perturbation for low-precision robustness.
TEST_CASE("Sparse LU zero-pivot perturbation (#123)", "[sparse][lu][perturb]") {
    SECTION("exactly singular: default throws; perturbation completes") {
        mat::compressed2D<double> A(2, 2);
        { mat::inserter<mat::compressed2D<double>> ins(A);
          ins[0][0] << 1.0; ins[0][1] << 1.0;
          ins[1][0] << 1.0; ins[1][1] << 1.0; }
        auto sym = factorization::sparse_lu_symbolic(A);

        REQUIRE_THROWS_AS(factorization::sparse_lu_numeric(A, sym), std::runtime_error);

        auto num = factorization::sparse_lu_numeric(A, sym, /*threshold=*/1.0, /*pivot_perturb=*/1e-8);
        REQUIRE(num.num_perturbed >= 1);                 // the collapsed pivot was perturbed
        vec::dense_vector<double> b = {1.0, 2.0}, x(2, 0.0);
        num.solve(x, b);                                 // no throw; finite (if inexact) result
        REQUIRE(std::isfinite(x(0)));
        REQUIRE(std::isfinite(x(1)));
    }

    SECTION("truly empty pivot column still throws even with perturbation on") {
        // Column 1 has no entries -> no pivot row to perturb (nothing to perturb):
        // perturbation must not paper over a genuine singularity.
        mat::compressed2D<double> A(2, 2);
        { mat::inserter<mat::compressed2D<double>> ins(A);
          ins[0][0] << 1.0; }
        auto sym = factorization::sparse_lu_symbolic(A);
        REQUIRE_THROWS_AS(
            factorization::sparse_lu_numeric(A, sym, /*threshold=*/1.0, /*pivot_perturb=*/1e-8),
            std::runtime_error);
    }

    SECTION("refactor reports a clean perturbation count (no stale carry-over)") {
        // Factor a singular matrix WITH perturbation (num_perturbed >= 1), then
        // refactor a well-conditioned same-pattern matrix: the refactor path does
        // not perturb, so its count must reset to 0 rather than carrying prev's.
        mat::compressed2D<double> A1(2, 2);
        { mat::inserter<mat::compressed2D<double>> ins(A1);
          ins[0][0] << 1.0; ins[0][1] << 1.0;
          ins[1][0] << 1.0; ins[1][1] << 1.0; }
        auto sym = factorization::sparse_lu_symbolic(A1);
        auto prev = factorization::sparse_lu_numeric(A1, sym, 1.0, 1e-8);
        REQUIRE(prev.num_perturbed >= 1);

        mat::compressed2D<double> A2(2, 2);   // same pattern, well-conditioned
        { mat::inserter<mat::compressed2D<double>> ins(A2);
          ins[0][0] << 2.0; ins[0][1] << 1.0;
          ins[1][0] << 1.0; ins[1][1] << 3.0; }
        auto re = factorization::sparse_lu_refactor(A2, prev);
        REQUIRE(re.num_perturbed == 0);       // refactor count is clean
        vec::dense_vector<double> b = {1.0, 2.0}, x(2, 0.0);
        re.solve(x, b);
        REQUIRE(relative_residual(A2, x, b) < 1e-12);
    }

    SECTION("nonsingular: perturbation never fires (clean factor, same accuracy)") {
        mat::compressed2D<double> A(3, 3);
        { mat::inserter<mat::compressed2D<double>> ins(A);
          ins[0][0] << 4.0; ins[0][1] << 1.0;
          ins[1][0] << 1.0; ins[1][1] << 5.0; ins[1][2] << 2.0;
          ins[2][1] << 1.0; ins[2][2] << 6.0; }
        auto sym = factorization::sparse_lu_symbolic(A);
        auto num = factorization::sparse_lu_numeric(A, sym, /*threshold=*/1.0, /*pivot_perturb=*/1e-8);
        REQUIRE(num.num_perturbed == 0);                 // well-conditioned -> no perturbation
        vec::dense_vector<double> b = {1.0, 2.0, 3.0}, x(3, 0.0);
        num.solve(x, b);
        REQUIRE(relative_residual(A, x, b) < 1e-12);     // accuracy unaffected
    }

    SECTION("low precision (float) that throws completes when perturbed") {
        // [[1,1],[1,1+d]] is nonsingular in double, but d = 5e-8 is below float
        // epsilon so 1+d rounds to 1.0f and the column-1 pivot cancels to zero.
        const double d = 5e-8;
        mat::compressed2D<float> Af(2, 2);
        { mat::inserter<mat::compressed2D<float>> ins(Af);
          ins[0][0] << 1.0f; ins[0][1] << 1.0f;
          ins[1][0] << 1.0f; ins[1][1] << static_cast<float>(1.0 + d); }
        auto symf = factorization::sparse_lu_symbolic(Af);

        REQUIRE_THROWS_AS(factorization::sparse_lu_numeric(Af, symf), std::runtime_error);

        auto numf = factorization::sparse_lu_numeric(Af, symf, /*threshold=*/1.0f, /*pivot_perturb=*/1e-6f);
        REQUIRE(numf.num_perturbed >= 1);
        vec::dense_vector<float> bf = {2.0f, 2.0f}, xf(2, 0.0f);
        numf.solve(xf, bf);                              // completes instead of throwing
        REQUIRE(std::isfinite(xf(0)));
        REQUIRE(std::isfinite(xf(1)));
    }
}

// A factor with `symbolic` installed (n > 0) but no factor must reject solve
// rather than read empty factor arrays (#307).
TEST_CASE("Sparse LU solve rejects a missing factor", "[sparse][lu][edge]") {
    factorization::lu_numeric<double> num;
    num.symbolic.n = 3;                       // symbolic set, set_factor never called
    vec::dense_vector<double> x(3), b(3);
    for (int i = 0; i < 3; ++i) b(i) = 1.0;
    REQUIRE_THROWS_AS(num.solve(x, b), std::logic_error);
}

// A k x k five-point grid, made unsymmetric by random off-diagonal factors in
// [0.5, 1.5] and given a diagonal weakened by a random factor in [0.05, 1]. Full
// partial pivoting leaves the diagonal for about half the columns; a small
// threshold should keep nearly every pivot on it.
static mat::compressed2D<double> weak_diagonal_grid_916(std::size_t k) {
    const std::size_t n = k * k;
    // mt19937's output is fully specified; uniform_real_distribution is not, so
    // map the raw output by hand to get the same matrix on every toolchain.
    std::mt19937 gen(916);
    auto uniform = [&gen](double lo, double hi) {
        return lo + (hi - lo) * (static_cast<double>(gen()) / 4294967296.0);
    };
    auto off = [&] { return uniform(0.5, 1.5); };
    auto dia = [&] { return uniform(0.05, 1.0); };
    mat::compressed2D<double> A(n, n);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        for (std::size_t r = 0; r < k; ++r)
            for (std::size_t c = 0; c < k; ++c) {
                const std::size_t i = r * k + c;
                ins[i][i] << dia();
                if (c > 0)     ins[i][i - 1] << -0.25 * off();
                if (c + 1 < k) ins[i][i + 1] << -0.25 * off();
                if (r > 0)     ins[i][i - k] << -0.25 * off();
                if (r + 1 < k) ins[i][i + k] << -0.25 * off();
            }
    }
    return A;
}

/// Pivots taken on the diagonal of the permuted column: row_perm[k] == col_perm[k].
static std::size_t diagonal_pivots_916(const std::vector<std::size_t>& row_perm,
                                       const std::vector<std::size_t>& col_perm) {
    std::size_t d = 0;
    for (std::size_t i = 0; i < row_perm.size(); ++i)
        if (row_perm[i] == col_perm[i]) ++d;
    return d;
}

static bool is_identity_916(const std::vector<std::size_t>& p) {
    for (std::size_t i = 0; i < p.size(); ++i)
        if (p[i] != i) return false;
    return true;
}

TEST_CASE("Threshold pivoting prefers the permuted diagonal q[k] (#916)", "[sparse][lu]") {
    // Column k of the factored matrix is A(:, q[k]), so the diagonal the
    // threshold should prefer is row q[k]. Preferring row k instead made the
    // threshold nearly inert under any non-natural ordering.
    const auto A = weak_diagonal_grid_916(20);
    const std::size_t n = A.num_rows();
    vec::dense_vector<double> b(n, 1.0);

    for (int which = 0; which < 2; ++which) {
        auto sym = which == 0 ? factorization::sparse_lu_symbolic(A, ordering::rcm{})
                              : factorization::sparse_lu_symbolic(A, ordering::colamd{});
        INFO("ordering " << (which == 0 ? "rcm" : "colamd"));
        REQUIRE_FALSE(is_identity_916(sym.col_perm));

        auto full    = factorization::sparse_lu_numeric(A, sym, 1.0);
        auto relaxed = factorization::sparse_lu_numeric(A, sym, 0.01);
        const auto dfull = diagonal_pivots_916(full.row_perm, sym.col_perm);
        const auto drel  = diagonal_pivots_916(relaxed.row_perm, sym.col_perm);
        // The matrix exercises pivoting: full pivoting leaves the diagonal often.
        REQUIRE(dfull < 3 * n / 4);
        // A small threshold keeps nearly every pivot on the diagonal. Before the
        // fix it was about a third of n, near the full-pivoting count.
        REQUIRE(drel >= 85 * n / 100);
        REQUIRE(relaxed.factorL().nnz() + relaxed.factorU().nnz()
                < full.factorL().nnz() + full.factorU().nnz());

        vec::dense_vector<double> x(n, 0.0);
        relaxed.solve(x, b);
        REQUIRE(relative_residual(A, x, b) < 1e-10);
    }
}

TEST_CASE("Threshold pivoting keeps a nonzero pivot when threshold * max underflows (#916)",
          "[sparse][lu]") {
    // A = [[0, s], [s, 0]] with s = denorm_min and q = [1, 0]. The diagonal of
    // column 0 of A(:, q) is A(1, 1) = 0, and threshold * s rounds to 0, so a
    // bare ">= threshold * max" test would pick the zero diagonal.
    const double s = std::numeric_limits<double>::denorm_min();
    mat::compressed2D<double> A(2, 2);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][1] << s;
        ins[1][0] << s;
    }
    auto swap_cols = [](const auto&) { return std::vector<std::size_t>{1, 0}; };
    auto sym = factorization::sparse_lu_symbolic(A, swap_cols);
    auto num = factorization::sparse_lu_numeric(A, sym, 0.01);
    const auto& U = num.factorU();
    for (std::size_t k = 0; k < 2; ++k)          // U(k,k) is stored last in column k
        REQUIRE(U.values[U.col_ptr[k + 1] - 1] != 0.0);
}
