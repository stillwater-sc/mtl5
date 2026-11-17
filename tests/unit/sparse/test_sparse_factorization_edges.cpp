// Sparse direct factorizations: the edge cases of issue 864 (QA sweep #704).
//
//   - iterative_refine evaluates its last correction;
//   - sparse_lu_numeric and native_klu_factor reject a threshold outside (0, 1];
//   - the three refactors reject an entry outside the prior factor's pattern
//     and accept one in its fill;
//   - supernodal_ldlt_numeric rejects an entry outside the symbolic pattern, and
//     the numeric phases a symbolic analysis of the wrong size;
//   - Cholesky and LDL^T reject a NaN or infinite pivot;
//   - sparse_qr scales a column whose squares overflow or underflow, keeps the
//     exact-zero rank test, and handles 1x1 and empty systems;
//   - shift_diagonal and shift_invert_operator reject a non-square A;
//   - sparse_cholesky_numeric is linear where it was quadratic.
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/sparse/iterative_refine.hpp>
#include <mtl/sparse/ordering/amd.hpp>
#include <mtl/sparse/factorization/sparse_lu.hpp>
#include <mtl/sparse/factorization/supernodal_lu.hpp>
#include <mtl/sparse/factorization/native_klu.hpp>
#include <mtl/sparse/factorization/sparse_cholesky.hpp>
#include <mtl/sparse/factorization/sparse_ldlt.hpp>
#include <mtl/sparse/factorization/supernodal_ldlt.hpp>
#include <mtl/sparse/factorization/sparse_qr.hpp>
#include <mtl/sparse/eigen/shift_invert.hpp>

using namespace mtl;
namespace fz = mtl::sparse::factorization;

namespace {

using M   = mat::compressed2D<double>;
using Vec = vec::dense_vector<double>;
using entries = std::vector<std::tuple<std::size_t, std::size_t, double>>;

M make(std::size_t m, std::size_t n, const entries& e) {
    M A(m, n);
    {
        mat::inserter<M> ins(A);
        for (const auto& [i, j, v] : e) ins[i][j] << v;
    }
    return A;
}

/// tridiag(-1, 4, -1), plus `extra` entries.
M tridiagonal(std::size_t n, const entries& extra = {}) {
    entries e;
    for (std::size_t i = 0; i < n; ++i) {
        e.emplace_back(i, i, 4.0);
        if (i + 1 < n) { e.emplace_back(i, i + 1, -1.0); e.emplace_back(i + 1, i, -1.0); }
    }
    e.insert(e.end(), extra.begin(), extra.end());
    return make(n, n, e);
}

Vec ones(std::size_t n) { return Vec(n, 1.0); }

/// The identity column ordering, so the factor's fill is the one the test names.
struct natural_order {
    template <typename Matrix>
    std::vector<std::size_t> operator()(const Matrix& A) const {
        std::vector<std::size_t> p(A.num_cols());
        for (std::size_t i = 0; i < p.size(); ++i) p[i] = i;
        return p;
    }
};

double residual_inf(const M& A, const Vec& x, const Vec& b) {
    double r = 0.0;
    const auto& rp = A.ref_major(); const auto& ci = A.ref_minor(); const auto& d = A.ref_data();
    for (std::size_t i = 0; i < A.num_rows(); ++i) {
        double ax = 0.0;
        for (std::size_t k = rp[i]; k < rp[i + 1]; ++k) ax += d[k] * x(ci[k]);
        const double ri = std::abs(b(i) - ax);
        // std::max(r, NaN) keeps r: a NaN solution must not pass as residual 0.
        if (!std::isfinite(ri)) return std::numeric_limits<double>::infinity();
        r = std::max(r, ri);
    }
    return r;
}

} // namespace

// --- iterative_refine --------------------------------------------------------

TEST_CASE("iterative_refine: max_iter = 1 keeps its one correction", "[sparse][refine][864]") {
    const M A = tridiagonal(3);
    const Vec b = ones(3);
    const auto fac = fz::sparse_lu_numeric(A, fz::sparse_lu_symbolic(A));
    for (bool scaled : {false, true}) {
        Vec x(3, 0.0);
        sparse::refine_options opt;
        opt.max_iter = 1;
        opt.scaled = scaled;
        const auto res = sparse::iterative_refine(A, fac, b, x, opt);
        // It returned x = 0, iters = 1, rel_residual = 1: the correction was
        // applied, never evaluated, and dropped for the starting x.
        CHECK(res.iters == 1);
        CHECK(res.rel_residual < 1e-14);
        CHECK(residual_inf(A, x, b) < 1e-14);
    }
}

TEST_CASE("iterative_refine: max_iter = 2 and max_iter <= 0", "[sparse][refine][864]") {
    const M A = tridiagonal(3);
    const Vec b = ones(3);
    const auto fac = fz::sparse_lu_numeric(A, fz::sparse_lu_symbolic(A));
    {
        Vec x(3, 0.0);
        sparse::refine_options opt;
        opt.max_iter = 2;
        const auto res = sparse::iterative_refine(A, fac, b, x, opt);
        CHECK(res.iters == 2);
        CHECK(residual_inf(A, x, b) < 1e-14);
    }
    for (int max_iter : {0, -3}) {
        Vec x(3, 0.25);
        sparse::refine_options opt;
        opt.max_iter = max_iter;
        const auto res = sparse::iterative_refine(A, fac, b, x, opt);
        CHECK(res.iters == 0);
        CHECK(x(0) == 0.25);            // the supplied x, unchanged
        CHECK(std::abs(res.rel_residual - 0.5) < 1e-15);   // b - A x = (0.25, 0.5, 0.25)
    }
}

TEST_CASE("iterative_refine: rel_tol stops at the first evaluation that meets it", "[sparse][refine][864]") {
    const M A = tridiagonal(5);
    const Vec b = ones(5);
    const auto fac = fz::sparse_lu_numeric(A, fz::sparse_lu_symbolic(A));
    Vec x(5, 0.0);
    sparse::refine_options opt;
    opt.max_iter = 10;
    opt.rel_tol = 1e-12;
    const auto res = sparse::iterative_refine(A, fac, b, x, opt);
    CHECK(res.converged);
    CHECK(res.iters == 1);              // one correction from x = 0 solves it
}

TEST_CASE("iterative_refine: a NaN correction is not a converged residual 0", "[sparse][refine][185]") {
    // A correction solve that overflowed left NaN in the next residual, which
    // norm_inf measured 0 (std::max keeps its first argument against NaN): the
    // NaN iterate became the best, converged = true, rel_residual = 0.
    struct nan_solve {
        void solve(Vec& dx, const Vec&) const {
            for (std::size_t i = 0; i < dx.size(); ++i) dx(i) = std::numeric_limits<double>::quiet_NaN();
        }
    };
    const M A = tridiagonal(3);
    const Vec b = ones(3);
    Vec x(3, 0.0);
    sparse::refine_options opt;
    opt.max_iter = 3;
    opt.rel_tol = 1e-12;
    const auto res = sparse::iterative_refine(A, nan_solve{}, b, x, opt);
    CHECK_FALSE(res.converged);
    CHECK(res.rel_residual == 1.0);   // the best iterate is the starting x = 0
    for (std::size_t i = 0; i < 3; ++i) CHECK(x(i) == 0.0);
}

// --- threshold ---------------------------------------------------------------

TEST_CASE("sparse_lu_numeric and native_klu_factor reject a threshold outside (0, 1]", "[sparse][lu][864]") {
    const M P = make(2, 2, {{0, 1, 1.0}, {1, 0, 1.0}});   // [[0,1],[1,0]]
    const auto sym = fz::sparse_lu_symbolic(P);
    for (double t : {0.0, -0.5, 1.5, std::numeric_limits<double>::quiet_NaN()}) {
        // threshold 0 picked the zero pivot, and the solve returned NaN.
        CHECK_THROWS_AS(fz::sparse_lu_numeric(P, sym, t), std::invalid_argument);
        CHECK_THROWS_AS(fz::native_klu_factor(P, t), std::invalid_argument);
        Vec x(2), b = ones(2);
        CHECK_THROWS_AS(fz::sparse_lu_solve(P, x, b, sparse::ordering::amd{}, t), std::invalid_argument);
    }
    for (double t : {1.0, 0.1}) {
        const auto fac = fz::sparse_lu_numeric(P, sym, t);
        Vec x(2), b(2);
        b(0) = 1.0; b(1) = 2.0;
        fac.solve(x, b);
        CHECK(x(0) == 2.0);
        CHECK(x(1) == 1.0);
    }
}

// --- refactor with a changed pattern -------------------------------------------

TEST_CASE("refactor rejects an entry outside the prior factor's pattern", "[sparse][refactor][864]") {
    const M A1 = tridiagonal(4);
    const M A2 = tridiagonal(4, {{0, 2, 0.5}});   // the sweep's extra A(0,2)
    const Vec b = ones(4);

    SECTION("sparse_lu_refactor") {
        auto fac = fz::sparse_lu_numeric(A1, fz::sparse_lu_symbolic(A1));
        // It returned with a row-0 residual of -0.43.
        CHECK_THROWS_AS(fz::sparse_lu_refactor(A2, fac), std::invalid_argument);
        CHECK_THROWS_AS(fz::sparse_lu_refactor_in_place(A2, fac), std::invalid_argument);
        Vec x(4);
        fac.solve(x, b);                           // fac is unchanged: it solves A1
        CHECK(residual_inf(A1, x, b) < 1e-14);
    }
    SECTION("supernodal_lu_refactor") {
        const auto sym = fz::supernodal_lu_symbolic_analyze(A1, natural_order{});
        auto fac = fz::supernodal_lu_numeric(A1, sym);
        CHECK_THROWS_AS(fz::supernodal_lu_refactor(A2, fac), std::invalid_argument);
        CHECK_THROWS_AS(fz::supernodal_lu_refactor_in_place(A2, fac), std::invalid_argument);
    }
    SECTION("native_klu_refactor, inside a diagonal block") {
        const auto fac = fz::native_klu_factor(A1);
        CHECK_THROWS_AS(fz::native_klu_refactor(A2, fac), std::invalid_argument);
    }
    SECTION("native_klu_refactor, below the block triangular form") {
        // Upper bidiagonal: three 1x1 blocks. A(2,0) couples the last block
        // back to the first, where the solve never reads.
        const M U1 = make(3, 3, {{0, 0, 2.0}, {0, 1, 1.0}, {1, 1, 2.0}, {1, 2, 1.0}, {2, 2, 2.0}});
        const M U2 = make(3, 3, {{0, 0, 2.0}, {0, 1, 1.0}, {1, 1, 2.0}, {1, 2, 1.0}, {2, 2, 2.0},
                                 {2, 0, 1.0}});
        const auto fac = fz::native_klu_factor(U1);
        REQUIRE(fac.nblocks() == 3);
        CHECK_THROWS_AS(fz::native_klu_refactor(U2, fac), std::invalid_argument);
    }
}

TEST_CASE("refactor accepts new values and entries in the factor's fill", "[sparse][refactor][864]") {
    // [[4,1,1],[1,4,0],[1,0,4]]: eliminating row 0 fills L(2,1) and U(1,2).
    const M A1 = make(3, 3, {{0, 0, 4.0}, {0, 1, 1.0}, {0, 2, 1.0}, {1, 0, 1.0}, {1, 1, 4.0},
                             {2, 0, 1.0}, {2, 2, 4.0}});
    const M A2 = make(3, 3, {{0, 0, 5.0}, {0, 1, 1.0}, {0, 2, 2.0}, {1, 0, 1.0}, {1, 1, 4.0},
                             {1, 2, 0.5}, {2, 0, 1.0}, {2, 1, -0.5}, {2, 2, 4.0}});
    const Vec b = ones(3);
    {
        const auto fac = fz::sparse_lu_numeric(A1, fz::sparse_lu_symbolic(A1));
        const auto re = fz::sparse_lu_refactor(A2, fac);
        Vec x(3);
        re.solve(x, b);
        CHECK(residual_inf(A2, x, b) < 1e-14);
    }
    {
        const auto sym = fz::supernodal_lu_symbolic_analyze(A1, natural_order{});
        const auto re = fz::supernodal_lu_refactor(A2, fz::supernodal_lu_numeric(A1, sym));
        Vec x(3);
        re.solve(x, b);
        CHECK(residual_inf(A2, x, b) < 1e-14);
    }
    {
        const auto re = fz::native_klu_refactor(A2, fz::native_klu_factor(A1));
        Vec x(3);
        re.solve(x, b);
        CHECK(residual_inf(A2, x, b) < 1e-14);
    }
}

TEST_CASE("native_klu_refactor: 1x1 and empty", "[sparse][klu][864]") {
    const auto f1 = fz::native_klu_factor(make(1, 1, {{0, 0, 2.0}}));
    const auto r1 = fz::native_klu_refactor(make(1, 1, {{0, 0, 4.0}}), f1);
    Vec x(1), b(1, 8.0);
    r1.solve(x, b);
    CHECK(x(0) == 2.0);

    const M E(0, 0);
    const auto f0 = fz::native_klu_factor(E);
    const auto r0 = fz::native_klu_refactor(E, f0);
    Vec x0(0), b0(0);
    r0.solve(x0, b0);
    CHECK(r0.num_rows() == 0);
}

// --- Cholesky and LDL^T --------------------------------------------------------

TEST_CASE("Cholesky and LDL^T reject a NaN or infinite pivot", "[sparse][cholesky][ldlt][864]") {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    for (double d : {nan, inf}) {
        const M A = make(2, 2, {{0, 0, d}, {1, 1, 1.0}});
        // diag(NaN, 1) gave L(0,0) = NaN with no error.
        CHECK_THROWS_AS(fz::sparse_cholesky_numeric(A, fz::sparse_cholesky_symbolic(A)),
                        std::runtime_error);
        CHECK_THROWS_AS(fz::sparse_ldlt_numeric(A, fz::sparse_ldlt_symbolic(A)), std::runtime_error);
        CHECK_THROWS_AS(fz::supernodal_ldlt_numeric(A, fz::supernodal_ldlt_symbolic(A)),
                        std::runtime_error);
    }
    // A pivot that only reaches NaN in the elimination: inf - inf at (1,1).
    const M B = make(2, 2, {{0, 0, 1.0}, {0, 1, inf}, {1, 0, inf}, {1, 1, inf}});
    CHECK_THROWS_AS(fz::sparse_cholesky_numeric(B, fz::sparse_cholesky_symbolic(B)), std::runtime_error);
    CHECK_THROWS_AS(fz::sparse_ldlt_numeric(B, fz::sparse_ldlt_symbolic(B)), std::runtime_error);
}

TEST_CASE("numeric phases reject a symbolic analysis of the wrong size", "[sparse][cholesky][ldlt][864]") {
    const M A = tridiagonal(4);
    auto chol = fz::sparse_cholesky_symbolic(A);
    chol.perm.pop_back();
    CHECK_THROWS_AS(fz::sparse_cholesky_numeric(A, chol), std::invalid_argument);
    auto ldlt = fz::sparse_ldlt_symbolic(A);
    ldlt.parent.pop_back();
    CHECK_THROWS_AS(fz::sparse_ldlt_numeric(A, ldlt), std::invalid_argument);
    auto sn = fz::supernodal_ldlt_symbolic(A);
    sn.sperm.pop_back();
    CHECK_THROWS_AS(fz::supernodal_ldlt_numeric(A, sn), std::invalid_argument);
}

TEST_CASE("supernodal_ldlt_numeric rejects an entry outside the symbolic pattern", "[sparse][ldlt][864]") {
    const M A1 = tridiagonal(5);
    const M A2 = tridiagonal(5, {{0, 4, 0.5}, {4, 0, 0.5}});
    const auto sym = fz::supernodal_ldlt_symbolic(A1);
    // It wrote at P[c*m + NPOS], out of bounds.
    CHECK_THROWS_AS(fz::supernodal_ldlt_numeric(A2, sym), std::invalid_argument);
    CHECK_NOTHROW(fz::supernodal_ldlt_numeric(A1, sym));
}

// --- sparse_qr -------------------------------------------------------------------

TEST_CASE("sparse_qr scales a column whose squares overflow or underflow", "[sparse][qr][864]") {
    // 2^511 passes the bound on the sum of squares but overflows v0^2, the
    // reflector's own denominator (#868 review): beta was 0 and x wrong.
    for (double s : {1e200, 1e-200, 1e300, 1e-300, 0x1p511}) {
        // [[s, 1], [s, -1]] x = (2, 0): x = (1/s, 1).
        const M A = make(2, 2, {{0, 0, s}, {0, 1, 1.0}, {1, 0, s}, {1, 1, -1.0}});
        Vec b(2), x(2);
        b(0) = 2.0; b(1) = 0.0;
        // s = 1e200 gave R(0,0) = -inf and a wrong x; s = 1e-200 summed its
        // squares to 0 and kept the column as if it were already reduced.
        fz::sparse_qr_solve(A, x, b);
        CHECK(std::abs(x(0) * s - 1.0) < 1e-14);
        CHECK(std::abs(x(1) - 1.0) < 1e-14);

        const auto num = fz::sparse_qr_numeric(A, fz::sparse_qr_symbolic(A));
        const double r00 = num.R.values[num.R.col_ptr[1] - 1];
        CHECK(std::isfinite(r00));
        CHECK(std::abs(std::abs(r00) / (std::sqrt(2.0) * s) - 1.0) < 1e-14);
    }
}

TEST_CASE("sparse_qr: least squares on a scaled tall matrix", "[sparse][qr][864]") {
    // Rows (s, 0), (0, 1), (s, 1): the normal equations give x = (1/s, 1)
    // for b = (1, 1, 2); the system is consistent.
    for (double s : {1e250, 1e-250}) {
        const M A = make(3, 2, {{0, 0, s}, {1, 1, 1.0}, {2, 0, s}, {2, 1, 1.0}});
        Vec b(3), x(2);
        b(0) = 1.0; b(1) = 1.0; b(2) = 2.0;
        fz::sparse_qr_solve(A, x, b);
        CHECK(std::abs(x(0) * s - 1.0) < 1e-14);
        CHECK(std::abs(x(1) - 1.0) < 1e-14);
    }
}

TEST_CASE("sparse_qr: the exact-zero rank test, 1x1 and empty", "[sparse][qr][864]") {
    {   // A zero column: R(1,1) is exactly 0.
        const M A = make(2, 2, {{0, 0, 1.0}, {1, 0, 1.0}});
        Vec x(2), b = ones(2);
        CHECK_THROWS_AS(fz::sparse_qr_solve(A, x, b), std::runtime_error);
    }
    {   // A small but normal R(1,1) is solved, not rejected (issue 864's decision).
        const M A = make(2, 2, {{0, 0, 1.0}, {1, 1, 1e-300}});
        Vec x(2), b = ones(2);
        fz::sparse_qr_solve(A, x, b);
        CHECK(x(0) == 1.0);
        CHECK(std::abs(x(1) * 1e-300 - 1.0) < 1e-14);
    }
    {
        const M A = make(1, 1, {{0, 0, -3.0}});
        Vec x(1), b(1, 6.0);
        fz::sparse_qr_solve(A, x, b);
        CHECK(x(0) == -2.0);
    }
    {
        const M E(0, 0);
        Vec x(0), b(0);
        CHECK_NOTHROW(fz::sparse_qr_solve(E, x, b));
    }
}

// --- shift-invert ---------------------------------------------------------------

TEST_CASE("shift_diagonal and shift_invert_operator", "[sparse][eigen][864]") {
    SECTION("a non-square A throws") {
        // A 2x3 A with an entry in column 2 overflowed column_permute's buffer.
        const M A = make(2, 3, {{0, 0, 1.0}, {1, 2, 1.0}});
        CHECK_THROWS_AS(sparse::shift_diagonal(A, 0.5), std::invalid_argument);
        CHECK_THROWS_AS((sparse::shift_invert_operator<double>(A, 0.5)), std::invalid_argument);
    }
    SECTION("a missing diagonal entry is created") {
        const M A = make(2, 2, {{0, 0, 1.0}, {0, 1, 2.0}, {1, 0, 3.0}});
        const auto B = sparse::shift_diagonal(A, 0.5);
        CHECK(B(0, 0) == 0.5);
        CHECK(B(0, 1) == 2.0);
        CHECK(B(1, 0) == 3.0);
        CHECK(B(1, 1) == -0.5);
    }
    SECTION("sigma on an eigenvalue: the singular shift is perturbed, not thrown") {
        // [[2,1],[1,2]] has eigenvalues 1 and 3; A - I = [[1,1],[1,1]] has a
        // zero second pivot, which the operator perturbs to sqrt(eps) * max|B|.
        // (A column of B that is entirely zero is still singular and throws.)
        const M A = make(2, 2, {{0, 0, 2.0}, {0, 1, 1.0}, {1, 0, 1.0}, {1, 1, 2.0}});
        const sparse::shift_invert_operator<double> op(A, 1.0);
        Vec v(2);
        v(0) = 1.0; v(1) = -1.0;        // the eigenvector of eigenvalue 1
        const Vec y = op * v;
        CHECK(std::isfinite(y(0)));
        CHECK(std::isfinite(y(1)));
        CHECK(std::abs(y(0)) > 1e6);    // the near-null direction is amplified
        CHECK(std::abs(y(0) + y(1) - 1.0) < 1e-6);   // and y stays a solution of B y = v
    }
}

// --- cost ------------------------------------------------------------------------

namespace {

/// tridiag(-1, 4, -1) with a dense last row and column (-1/n), SPD. In the
/// natural order its elimination tree is a path, so the last row's n entries
/// each walked to the root: O(n^2) for an O(n) factor.
M arrow(std::size_t n) {
    entries e;
    for (std::size_t i = 0; i < n; ++i) {
        e.emplace_back(i, i, i + 1 == n ? 6.0 : 4.0);
        if (i + 1 < n) { e.emplace_back(i, i + 1, -1.0); e.emplace_back(i + 1, i, -1.0); }
        if (i + 2 < n) {
            e.emplace_back(i, n - 1, -1.0 / double(n));
            e.emplace_back(n - 1, i, -1.0 / double(n));
        }
    }
    return make(n, n, e);
}

double best_time(const M& A) {
    const auto sym = fz::sparse_cholesky_symbolic(A);
    double best = std::numeric_limits<double>::infinity();
    for (int r = 0; r < 3; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        const auto num = fz::sparse_cholesky_numeric(A, sym);
        const auto t1 = std::chrono::steady_clock::now();
        REQUIRE(num.factor().ncols == A.num_rows());
        best = std::min(best, std::chrono::duration<double>(t1 - t0).count());
    }
    return best;
}

} // namespace

TEST_CASE("sparse_cholesky_numeric is linear in n on an arrow matrix", "[sparse][cholesky][cost][864]") {
    // 16x in n: linear work takes ~16x as long, the old quadratic walks ~256x
    // (n = 16000 took 3.2 s, against 3.4 ms now, at -O2). The bound is 64x,
    // a factor of 4 from either, so the machine and build type do not matter.
    const M small = arrow(1000), large = arrow(16000);
    const double t_small = best_time(small), t_large = best_time(large);
    INFO("t(1000) = " << t_small << " s, t(16000) = " << t_large << " s");
    CHECK(t_large < 64.0 * std::max(t_small, 1e-5));

    // And the factor is right: solve against the matrix.
    const auto num = fz::sparse_cholesky_numeric(large, fz::sparse_cholesky_symbolic(large));
    Vec x(16000), b = ones(16000);
    num.solve(x, b);
    CHECK(residual_inf(large, x, b) < 1e-14 * 16000);
}
