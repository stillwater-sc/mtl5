// The configuration-1/2/3 accumulation matrix, solver rows (#586).
//
// The companion of test_accumulation_configs_mt.cpp, which covers dot, gemm,
// the norms and the determinism of the reductions. Same three configurations,
// same test-only exact accumulator standing in for a quire:
//
//   1. plain  (Acc = float)
//   2. FMA    (fma_accumulator<float>)
//   3. exact  (exact_test::exact_acc<float>)
//
// The rows here are SOLVERS, where the accumulator governs some of the
// arithmetic and not the rest (vector updates, pivots, square roots, the QR
// sweep stay in the element type). So nothing is asserted from theory: every
// ordering below was measured first, and what is asserted is what the
// measurement shows with a wide margin. Measured on this file's inputs
// (relative residual ||b - A x|| / ||b||, computed exactly):
//
//   sparse LU, Hilbert(n) in float         plain     fma       exact
//       n = 5                              2.4e-8    1.5e-8    1.1e-9
//       n = 8                              1.5e-8    3.5e-8    2.9e-9
//   sparse Cholesky, Hilbert(n) in float
//       n = 5                              3.0e-8    2.1e-8    7.2e-9
//       n = 8                              2.6e-8    1.3e-8    1.1e-9
//   sparse Cholesky, Hilbert(9)            not positive definite  factors
//   dense LU (#585)
//       n = 5                              1.8e-8    8.5e-9    1.1e-9
//       n = 8                              2.8e-8    1.3e-8    2.9e-9
//   dense Cholesky (#585)
//       n = 5                              3.0e-8    2.1e-8    7.2e-9
//       n = 8                              2.6e-8    1.3e-8    1.1e-9
//   dense inv, x = inv(A) b (#585)
//       n = 5                              1.4e-3    3.1e-4    2.5e-5
//       n = 8                              1.7e-1    6.9e-1    3.8e-3
//   dense Cholesky, Hilbert(9)             not positive definite  factors
//   LU refinement (float factor, double
//   residual), Hilbert(n), n = 9 .. 11     stalls at 1e-4 .. 5e-4   1e-12
//
// Dense and sparse LDLT showed no ordering until #602. Their terms are products
// of three values, and one product was rounded in the element type before the
// accumulator saw it; an accumulator that keeps it now also receives that
// product's rounding error (math::two_product), and exact is lowest at every
// n = 5 .. 10, by 1.06x (n = 6: 2.0e-8, 1.3e-8, 1.2e-8) up to 28x (n = 9:
// 3.4e-7, 1.3e-7, 1.2e-8). Dense and sparse give the same numbers. The dense
// solves form b(i) - sum in the accumulator, as the sparse ones do.
//
// And measured to show NO ordering, so asserted only to work:
//
//   * sparse QR on the same Hilbert matrices: the three are within a small
//     factor and swap places (n = 8: 1.9e-7, 1.1e-7, 5.0e-7).
//   * dense Bunch-Kaufman LDLT (#585), Hilbert(n) and the indefinite saddle
//     [[H, 1], [1^T, 0]], n = 4 .. 9 (one or two interchanges each): no
//     ordering. Hilbert(6): 4.9e-8, 1.1e-8, 1.5e-8; saddle n = 7: 9.4e-8,
//     2.3e-6, 8.0e-8. Not the #602 cause: its terms are L(i,p) W(j,p) with W
//     the stored Schur column, not a product formed for the accumulator.
//   * CG and GMRES in float: the accumulator covers the dot products and the
//     SpMV, but the recurrences update x, r and p in float, and that is what
//     limits them. Iteration counts and true residuals agree to a few percent.
//   * eigenvalue_symmetric: the accumulator reaches only the Householder
//     tridiagonalization. lambda_max is accurate in all three; lambda_min of
//     Hilbert(7) is NOT improved (plain 7.6e-3, exact 0.43 relative error).
//   * power_iteration on the 1-D Laplacian: identical in all three. The
//     matvec there is three well-scaled terms a row; there is nothing to lose.
//   * arnoldi on the operator below: no clean ordering (n = 32, k = 16: plain
//     6.4e-3, fma 9.7e-2, exact 1.3e-2). The Gram-Schmidt updates of w are in
//     float, and they cancel as heavily as the matvec does.
//
// And one eigen row that does order, since the Accumulator reaches the matvec
// (#591): power_iteration on A = P^-1 D P, an exactly representable non-normal
// matrix whose matvec cancels by four orders of magnitude (see `cancelling_
// similarity`), lambda_max = 2n, 5000 iterations, tol 1e-6:
//
//                                          plain     fma       exact
//       n = 16, k = 16  relative error     5.4e-3    2.0e-2    8.0e-6
//                       converged          no        no        yes, 14 its
//       n = 32, k = 32                     8.9e-2    8.2e-1    6.9e-5
//
// Before #591 all three were 5.4e-3 and none converged.
//
// The determinism row does hold by construction: CG with an exact accumulator
// is bit-identical threaded and serial; with plain float it is not.
//
// Built with -ffp-contract=off (tests/unit/CMakeLists.txt): configuration 1 is
// "the product rounds", and contraction would make it configuration 2.
//
// Sets MTL5_NUM_THREADS before the pool's first use.
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <numbers>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <vector>

#include <mtl/detail/thread_pool.hpp>
#include <mtl/itl/eigen/power_iteration.hpp>
#include <mtl/itl/iteration/basic_iteration.hpp>
#include <mtl/itl/krylov/cg.hpp>
#include <mtl/itl/krylov/gmres.hpp>
#include <mtl/itl/pc/identity.hpp>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/operation/cholesky.hpp>
#include <mtl/operation/eigenvalue_symmetric.hpp>
#include <mtl/operation/inv.hpp>
#include <mtl/operation/ldlt.hpp>
#include <mtl/operation/ldlt_bk.hpp>
#include <mtl/operation/lu.hpp>
#include <mtl/operation/operators.hpp>   // A * x, for power_iteration
#include <mtl/sparse/factorization/sparse_cholesky.hpp>
#include <mtl/sparse/factorization/sparse_ldlt.hpp>
#include <mtl/sparse/factorization/sparse_lu.hpp>
#include <mtl/sparse/factorization/sparse_qr.hpp>
#include <mtl/sparse/factorization/supernodal_lu.hpp>
#include <mtl/sparse/ordering/colamd.hpp>
#include <mtl/vec/dense_vector.hpp>

#include "exact_accumulator.hpp"

namespace {

const int g_set_threads = [] {
#if defined(_WIN32)
    _putenv_s("MTL5_NUM_THREADS", "4");
#else
    setenv("MTL5_NUM_THREADS", "4", /*overwrite=*/1);
#endif
    return 0;
}();

using exact_test::exact_acc;
template <typename T> using fma_acc = mtl::math::fma_accumulator<T>;
using smat = mtl::mat::compressed2D<float>;
using fvec = mtl::vec::dense_vector<float>;
namespace sf = mtl::sparse::factorization;

/// The Hilbert matrix, rounded to float, stored as T.
template <typename T>
mtl::mat::compressed2D<T> hilbert(std::size_t n) {
    mtl::mat::compressed2D<T> A(n, n);
    {
        mtl::mat::inserter<mtl::mat::compressed2D<T>> ins(A);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                ins[i][j] << static_cast<T>(static_cast<float>(1.0 / static_cast<double>(i + j + 1)));
    }
    return A;
}

/// 5-point Laplacian on an m x m grid, symmetrically scaled D L D so that the
/// condition number grows with `spread` (SPD for any spread).
smat laplacian_2d(std::size_t m, double spread) {
    const std::size_t n = m * m;
    smat A(n, n);
    auto d = [&](std::size_t i) { return std::pow(spread, static_cast<double>(i % 7) / 6.0); };
    {
        mtl::mat::inserter<smat> ins(A);
        for (std::size_t i = 0; i < m; ++i)
            for (std::size_t j = 0; j < m; ++j) {
                const std::size_t r = i * m + j;
                auto put = [&](std::size_t c, double v) {
                    ins[r][c] << static_cast<float>(d(r) * v * d(c));
                };
                put(r, 4.0);
                if (i > 0) put(r - m, -1.0);
                if (i + 1 < m) put(r + m, -1.0);
                if (j > 0) put(r - 1, -1.0);
                if (j + 1 < m) put(r + 1, -1.0);
            }
    }
    return A;
}

/// Upwind convection-diffusion on an m x m grid: nonsymmetric.
smat convection_diffusion(std::size_t m, float peclet) {
    const std::size_t n = m * m;
    smat A(n, n);
    {
        mtl::mat::inserter<smat> ins(A);
        for (std::size_t i = 0; i < m; ++i)
            for (std::size_t j = 0; j < m; ++j) {
                const std::size_t r = i * m + j;
                ins[r][r] << 4.0f;
                if (i > 0) ins[r][r - m] << -1.0f - peclet;
                if (i + 1 < m) ins[r][r + m] << -1.0f + peclet;
                if (j > 0) ins[r][r - 1] << -1.0f - peclet;
                if (j + 1 < m) ins[r][r + 1] << -1.0f + peclet;
            }
    }
    return A;
}

/// ||b - A x||_2 / ||b||_2, each residual entry computed exactly.
double rel_residual(const smat& A, const fvec& x, const fvec& b) {
    const auto& rp = A.ref_major();
    const auto& ci = A.ref_minor();
    const auto& v = A.ref_data();
    double s = 0.0, sb = 0.0;
    for (std::size_t r = 0; r < A.num_rows(); ++r) {
        auto a = exact_test::zero<float>();
        accumulator_add_product(a, b(r), 1.0f);
        for (std::size_t k = rp[r]; k < rp[r + 1]; ++k)
            accumulator_add_product(a, -v[k], x(ci[k]));
        const double e = accumulator_value(a, std::type_identity<double>{});
        s += e * e;
        sb += static_cast<double>(b(r)) * static_cast<double>(b(r));
    }
    return std::sqrt(s / sb);
}

/// b = A * x for a fixed, non-trivial x.
fvec rhs(const smat& A) {
    fvec x(A.num_rows()), b(A.num_rows());
    for (std::size_t i = 0; i < x.size(); ++i)
        x(i) = static_cast<float>(1 + (i * 7919) % 13) / 13.0f;
    mtl::mult<double>(A, x, b);
    return b;
}

template <typename Acc, typename Solve>
double direct_residual(const smat& A, Solve solve) {
    const fvec b = rhs(A);
    fvec x(A.num_rows(), 0.0f);
    solve.template operator()<Acc>(A, x, b);
    return rel_residual(A, x, b);
}

struct lu_solve {
    template <typename Acc> void operator()(const smat& A, fvec& x, const fvec& b) const {
        sf::sparse_lu_solve<Acc>(A, x, b);
    }
};
struct cholesky_solve {
    template <typename Acc> void operator()(const smat& A, fvec& x, const fvec& b) const {
        sf::sparse_cholesky_solve<Acc>(A, x, b);
    }
};
struct ldlt_solve {
    template <typename Acc> void operator()(const smat& A, fvec& x, const fvec& b) const {
        sf::sparse_ldlt_solve<Acc>(A, x, b);
    }
};
struct qr_solve {
    template <typename Acc> void operator()(const smat& A, fvec& x, const fvec& b) const {
        sf::sparse_qr_solve<Acc>(A, x, b);
    }
};

bool pool_is_threaded() {
    if (mtl::detail::thread_pool::instance().size() < 2) {
        WARN("pool < 2 workers: the threaded split is not exercised");
        return false;
    }
    return true;
}

} // namespace

// The serial twins of the accumulators: the same arithmetic without
// accumulator_merge, so the reductions in dot.hpp do not split (#582).
namespace serial_ns {

struct plain_float { float s; };
inline void accumulator_clear(plain_float& a) { a.s = 0.0f; }
template <std::same_as<float> V> void accumulator_assign(plain_float& a, const V& v) { a.s = v; }
template <std::same_as<float> V> void accumulator_add_product(plain_float& a, const V& m, const V& v) {
    a.s += m * v;
}
template <typename R> R accumulator_value(const plain_float& a, std::type_identity<R>) {
    return static_cast<R>(a.s);
}

struct exact_float { exact_acc<float> e; };
inline void accumulator_clear(exact_float& a) { accumulator_clear(a.e); }
template <std::same_as<float> V> void accumulator_assign(exact_float& a, const V& v) {
    accumulator_assign(a.e, v);
}
template <std::same_as<float> V> void accumulator_add_product(exact_float& a, const V& m, const V& v) {
    accumulator_add_product(a.e, m, v);
}
template <typename R> R accumulator_value(const exact_float& a, std::type_identity<R> t) {
    return accumulator_value(a.e, t);
}

} // namespace serial_ns

static_assert(!mtl::math::mergeable_accumulator<serial_ns::plain_float, float>);
static_assert(!mtl::math::mergeable_accumulator<serial_ns::exact_float, float>);

// ---------------------------------------------------------------------------
// Direct solvers.
// ---------------------------------------------------------------------------

TEST_CASE("[586:direct] sparse LU and Cholesky: configuration 3 cuts the residual",
          "[586:direct][accumulator]") {
    // Measured: at least 2.8x below both configurations 1 and 2 at every n
    // (header table), so a factor of 2 is asserted.
    for (std::size_t n = 5; n <= 8; ++n) {
        const auto A = hilbert<float>(n);
        INFO("Hilbert(" << n << ")");
        {
            const double r1 = direct_residual<float>(A, lu_solve{});
            const double r2 = direct_residual<fma_acc<float>>(A, lu_solve{});
            const double r3 = direct_residual<exact_acc<float>>(A, lu_solve{});
            INFO("LU: plain " << r1 << ", fma " << r2 << ", exact " << r3);
            CHECK(2.0 * r3 < r1);
            CHECK(2.0 * r3 < r2);
        }
        {
            const double r1 = direct_residual<float>(A, cholesky_solve{});
            const double r2 = direct_residual<fma_acc<float>>(A, cholesky_solve{});
            const double r3 = direct_residual<exact_acc<float>>(A, cholesky_solve{});
            INFO("Cholesky: plain " << r1 << ", fma " << r2 << ", exact " << r3);
            CHECK(2.0 * r3 < r1);
            CHECK(2.0 * r3 < r2);
        }
    }
}

TEST_CASE("[586:direct] sparse Cholesky of Hilbert(9): only configuration 3 factors it",
          "[586:direct][accumulator]") {
    // Hilbert(9) is positive definite, but its smallest eigenvalue (~5e-13) is
    // far below float's resolution. Summing the column updates with a rounding
    // accumulator drives the last pivot non-positive; summing them exactly
    // does not.
    const auto A = hilbert<float>(9);
    CHECK_THROWS_AS(direct_residual<float>(A, cholesky_solve{}), std::runtime_error);
    CHECK_THROWS_AS(direct_residual<fma_acc<float>>(A, cholesky_solve{}), std::runtime_error);
    double r3 = 1.0;
    REQUIRE_NOTHROW(r3 = direct_residual<exact_acc<float>>(A, cholesky_solve{}));
    CHECK(r3 < 1e-7);
}

TEST_CASE("[586:direct] sparse LDLT: configuration 3 is lowest (#602)",
          "[586:direct][accumulator]") {
    // Measured (header): exact lowest at every n, by as little as 1.06x (n = 6).
    for (std::size_t n = 5; n <= 8; ++n) {
        const auto A = hilbert<float>(n);
        INFO("Hilbert(" << n << ")");
        const double r1 = direct_residual<float>(A, ldlt_solve{});
        const double r2 = direct_residual<fma_acc<float>>(A, ldlt_solve{});
        const double r3 = direct_residual<exact_acc<float>>(A, ldlt_solve{});
        INFO("plain " << r1 << ", fma " << r2 << ", exact " << r3);
        CHECK(r1 < 1e-6);
        CHECK(r2 < 1e-6);
        CHECK(r3 < r1);
        CHECK(r3 < r2);
    }
}

TEST_CASE("[586:direct] sparse QR: every configuration solves; no ordering",
          "[586:direct][accumulator]") {
    // Measured: no consistent ordering between the configurations (header).
    for (std::size_t n = 5; n <= 8; ++n) {
        const auto A = hilbert<float>(n);
        INFO("Hilbert(" << n << ")");
        CHECK(direct_residual<float>(A, qr_solve{}) < 1e-6);
        CHECK(direct_residual<fma_acc<float>>(A, qr_solve{}) < 1e-6);
        CHECK(direct_residual<exact_acc<float>>(A, qr_solve{}) < 1e-6);
    }
}

// The dense direct factorizations (#585), on the same Hilbert matrices and
// right-hand sides as the sparse rows above. Each copies A into a dense2D and
// factors it in place; a failed factorization throws, as the sparse ones do.
mtl::mat::dense2D<float> dense(const smat& A) {
    mtl::mat::dense2D<float> D(A.num_rows(), A.num_cols());
    for (std::size_t i = 0; i < A.num_rows(); ++i)
        for (std::size_t j = 0; j < A.num_cols(); ++j) D(i, j) = A(i, j);
    return D;
}
struct dense_lu_solve {
    template <typename Acc> void operator()(const smat& A, fvec& x, const fvec& b) const {
        auto D = dense(A);
        if (mtl::lu_apply<Acc>(D, x, b) != 0) throw std::runtime_error("lu: singular");
    }
};
struct dense_cholesky_solve {
    template <typename Acc> void operator()(const smat& A, fvec& x, const fvec& b) const {
        auto D = dense(A);
        if (mtl::cholesky_factor<Acc>(D) != 0) throw std::runtime_error("cholesky: not positive definite");
        mtl::cholesky_solve<Acc>(D, x, b);
    }
};
struct dense_ldlt_solve {
    template <typename Acc> void operator()(const smat& A, fvec& x, const fvec& b) const {
        auto D = dense(A);
        if (mtl::ldlt_factor<Acc>(D) != 0) throw std::runtime_error("ldlt: zero pivot");
        mtl::ldlt_solve<Acc>(D, x, b);
    }
};
struct dense_ldlt_bk_solve {
    template <typename Acc> void operator()(const smat& A, fvec& x, const fvec& b) const {
        auto D = dense(A);
        mtl::bk_pivot_info piv;
        if (mtl::ldlt_bk_factor<Acc>(D, piv) != 0) throw std::runtime_error("ldlt_bk: singular block");
        mtl::ldlt_bk_solve<Acc>(D, piv, x, b);
    }
};
struct dense_inv_solve {
    template <typename Acc> void operator()(const smat& A, fvec& x, const fvec& b) const {
        const auto X = mtl::inv<Acc>(dense(A));
        mtl::mult<Acc>(X, b, x);   // x = inv(A) b, in the same configuration
    }
};

TEST_CASE("[586:direct] dense LU, Cholesky and inv: configuration 3 cuts the residual",
          "[586:direct][accumulator]") {
    // Measured (header table): LU at least 4.3x, Cholesky 2.9x and inv 10x below
    // both configurations 1 and 2 at every n; asserted with factors 2, 2 and 4.
    for (std::size_t n = 5; n <= 8; ++n) {
        const auto A = hilbert<float>(n);
        INFO("Hilbert(" << n << ")");
        auto row = [&](const char* name, auto solve, double factor) {
            const double r1 = direct_residual<float>(A, solve);
            const double r2 = direct_residual<fma_acc<float>>(A, solve);
            const double r3 = direct_residual<exact_acc<float>>(A, solve);
            INFO(name << ": plain " << r1 << ", fma " << r2 << ", exact " << r3);
            CHECK(factor * r3 < r1);
            CHECK(factor * r3 < r2);
        };
        row("LU", dense_lu_solve{}, 2.0);
        row("Cholesky", dense_cholesky_solve{}, 2.0);
        row("inv", dense_inv_solve{}, 4.0);
    }
}

TEST_CASE("[586:direct] dense Cholesky of Hilbert(9): only configuration 3 factors it",
          "[586:direct][accumulator]") {
    // As for the sparse Cholesky above.
    const auto A = hilbert<float>(9);
    CHECK_THROWS_AS(direct_residual<float>(A, dense_cholesky_solve{}), std::runtime_error);
    CHECK_THROWS_AS(direct_residual<fma_acc<float>>(A, dense_cholesky_solve{}), std::runtime_error);
    double r3 = 1.0;
    REQUIRE_NOTHROW(r3 = direct_residual<exact_acc<float>>(A, dense_cholesky_solve{}));
    CHECK(r3 < 1e-7);
}

TEST_CASE("[586:direct] dense LDLT: configuration 3 is lowest (#602)",
          "[586:direct][accumulator]") {
    // Measured (header): exact lowest at every n, by as little as 1.06x (n = 6).
    for (std::size_t n = 5; n <= 8; ++n) {
        const auto A = hilbert<float>(n);
        INFO("Hilbert(" << n << ")");
        const double r1 = direct_residual<float>(A, dense_ldlt_solve{});
        const double r2 = direct_residual<fma_acc<float>>(A, dense_ldlt_solve{});
        const double r3 = direct_residual<exact_acc<float>>(A, dense_ldlt_solve{});
        INFO("plain " << r1 << ", fma " << r2 << ", exact " << r3);
        CHECK(r1 < 1e-6);
        CHECK(r2 < 1e-6);
        CHECK(r3 < r1);
        CHECK(r3 < r2);
    }
}

/// The saddle-point matrix [[H, 1], [1^T, 0]], H = Hilbert(n): symmetric
/// indefinite, so Bunch-Kaufman interchanges and takes 2x2 pivots.
smat saddle_hilbert(std::size_t n) {
    smat A(n + 1, n + 1);
    {
        mtl::mat::inserter<smat> ins(A);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j)
                ins[i][j] << static_cast<float>(1.0 / static_cast<double>(i + j + 1));
            ins[i][n] << 1.0f;
            ins[n][i] << 1.0f;
        }
    }
    return A;
}

TEST_CASE("[586:direct] dense Bunch-Kaufman LDLT: every configuration solves; no ordering",
          "[586:direct][accumulator]") {
    // Measured (header): no configuration is reliably lowest -- fma is below
    // exact on Hilbert(4) and (6), and reaches 2.3e-6 on the saddle at n = 7.
    for (std::size_t n = 4; n <= 9; ++n) {
        for (const auto& A : {hilbert<float>(n), saddle_hilbert(n)}) {
            INFO("n = " << n << ", order " << A.num_rows());
            CHECK(direct_residual<float>(A, dense_ldlt_bk_solve{}) < 1e-5);
            CHECK(direct_residual<fma_acc<float>>(A, dense_ldlt_bk_solve{}) < 1e-5);
            CHECK(direct_residual<exact_acc<float>>(A, dense_ldlt_bk_solve{}) < 1e-5);
        }
    }
}

TEST_CASE("[586:refine] float-factor refinement: configuration 3 converges, 1 and 2 stall",
          "[586:direct][586:refine][accumulator]") {
    // Factor in float with each accumulator, refine the residual in double.
    // The rounding accumulators leave a factor too poor for refinement to
    // contract on Hilbert(9..11); the exact one does not. Measured: 1 and 2
    // stall between 1.2e-4 and 5.3e-4, 3 reaches 1.1e-12, 7.3e-12
    // and 6.8e-13 at n = 9, 10, 11 (mp-ir#12 reproduces them with a quire).
    for (std::size_t n = 9; n <= 11; ++n) {
        const auto A = hilbert<double>(n);
        auto refine = [&](auto tag) {
            using Acc = decltype(tag);
            mtl::vec::dense_vector<double> x(n), b(n, 1.0);
            mtl::sparse::refine_options opt;
            opt.max_iter = 50;
            opt.rel_tol = 1e-14;
            return sf::supernodal_lu_solve_refined<float, Acc>(
                A, x, b, mtl::sparse::ordering::colamd{}, opt).rel_residual;
        };
        const double r1 = refine(float{});
        const double r2 = refine(fma_acc<float>{});
        const double r3 = refine(exact_acc<float>{});
        INFO("Hilbert(" << n << "): plain " << r1 << ", fma " << r2 << ", exact " << r3);
        CHECK(r1 > 1e-6);
        CHECK(r2 > 1e-6);
        CHECK(r3 < 1e-10);
    }
}

// ---------------------------------------------------------------------------
// Krylov solvers.
// ---------------------------------------------------------------------------

namespace {

template <typename Acc>
double cg_residual(const smat& A, int& iters) {
    const fvec b = rhs(A);
    fvec x(A.num_rows(), 0.0f);
    mtl::itl::pc::identity<smat> I(A);
    mtl::itl::basic_iteration<float> it(b, 2000, 1e-6f);
    mtl::itl::cg<Acc>(A, x, b, I, it);
    iters = it.iterations();
    return rel_residual(A, x, b);
}

template <typename Acc>
double gmres_residual(const smat& A, int& iters) {
    const fvec b = rhs(A);
    fvec x(A.num_rows(), 0.0f);
    mtl::itl::pc::identity<smat> I(A);
    mtl::itl::basic_iteration<float> it(b, 2000, 1e-6f);
    mtl::itl::gmres<Acc>(A, x, b, I, it, 20);
    iters = it.iterations();
    return rel_residual(A, x, b);
}

} // namespace

TEST_CASE("[586:krylov] CG and GMRES in float: every configuration converges",
          "[586:krylov][accumulator]") {
    // Measured: no ordering (header). Each run must reach the 1e-6 recurrence
    // tolerance well inside the iteration limit with a true residual of the
    // same order.
    for (double spread : {1.0, 10.0}) {
        const auto A = laplacian_2d(20, spread);
        INFO("CG, spread " << spread);
        int i1 = 0, i2 = 0, i3 = 0;
        CHECK(cg_residual<float>(A, i1) < 4e-6);
        CHECK(cg_residual<fma_acc<float>>(A, i2) < 4e-6);
        CHECK(cg_residual<exact_acc<float>>(A, i3) < 4e-6);
        CHECK(std::max({i1, i2, i3}) < 1000);
    }
    {
        const auto A = convection_diffusion(20, 0.3f);
        int i1 = 0, i2 = 0, i3 = 0;
        CHECK(gmres_residual<float>(A, i1) < 4e-6);
        CHECK(gmres_residual<fma_acc<float>>(A, i2) < 4e-6);
        CHECK(gmres_residual<exact_acc<float>>(A, i3) < 4e-6);
        CHECK(std::max({i1, i2, i3}) < 1000);
    }
}

TEST_CASE("[586:krylov][586:determinism] CG with configuration 3 is independent of the thread count",
          "[586:krylov][586:determinism][accumulator][mt]") {
    // CG's dot products split across the pool when the accumulator can merge.
    // With the exact accumulator the merge is exact, so the whole iteration --
    // every x, r, p -- matches the serial run bit for bit. With plain float the
    // regrouped dot products move the trajectory. The serial run uses the same
    // arithmetic without accumulator_merge. 160000 unknowns: more than two
    // 65536-term chunks, so plain float splits too.
    const unsigned n_threads = mtl::detail::thread_pool::instance().size();
    REQUIRE(n_threads >= 1);
    REQUIRE(n_threads <= 4);
    const auto A = laplacian_2d(400, 1.0);
    const fvec b = rhs(A);
    auto run = [&](auto tag) {
        using Acc = decltype(tag);
        fvec x(A.num_rows(), 0.0f);
        mtl::itl::pc::identity<smat> I(A);
        mtl::itl::basic_iteration<float> it(b, 10, 0.0f);   // exactly 10 iterations
        mtl::itl::cg<Acc>(A, x, b, I, it);
        return x;
    };
    auto same = [](const fvec& u, const fvec& v) {
        return std::memcmp(&u(0), &v(0), u.size() * sizeof(float)) == 0;
    };
    const fvec e_thr = run(exact_acc<float>{}), e_ser = run(serial_ns::exact_float{});
    REQUIRE(same(e_thr, e_ser));
    const fvec f_thr = run(float{}), f_ser = run(serial_ns::plain_float{});
    if (pool_is_threaded())
        CHECK(!same(f_thr, f_ser));
    else
        CHECK(same(f_thr, f_ser));
}

// ---------------------------------------------------------------------------
// Eigensolvers.
// ---------------------------------------------------------------------------

/// A = P^-1 D P with P = I + k e 1^T, e = (1, -1, 1, ...), so P^-1 = I - k e 1^T
/// (1^T e = 0 for even n) and A's eigenvalues are exactly D = diag(2n, 1, 2,
/// ..., n - 1). Every entry is an integer below 2^24, so A is exact in float.
/// The dominant eigenvector P^-1 e_0 = e_0 - k e has entries of size k, and
/// A v = 2n v is a cancellation of n terms of size ~k^3 n.
mtl::mat::dense2D<float> cancelling_similarity(std::size_t n, double k) {
    std::vector<double> d(n), e(n);
    for (std::size_t i = 0; i < n; ++i) {
        d[i] = i ? static_cast<double>(i) : 2.0 * static_cast<double>(n);
        e[i] = i % 2 ? -1.0 : 1.0;
    }
    double de = 0.0;   // 1^T D e
    for (std::size_t i = 0; i < n; ++i) de += d[i] * e[i];
    mtl::mat::dense2D<float> A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            const double a = (i == j ? d[i] : 0.0) + k * d[i] * e[i] - k * e[i] * d[j] - k * k * e[i] * de;
            A(i, j) = static_cast<float>(a);
            REQUIRE(static_cast<double>(A(i, j)) == a);
        }
    return A;
}

TEST_CASE("[586:eigen] power_iteration on a cancelling matvec: exact converges, plain and fma do not",
          "[586:eigen][accumulator]") {
    for (const auto [n, k, bound] : {std::tuple{std::size_t{16}, 16.0, 1e-4},
                                     std::tuple{std::size_t{32}, 32.0, 1e-3}}) {
        const auto A = cancelling_similarity(n, k);
        const double lam = 2.0 * static_cast<double>(n);
        fvec v0(n);
        for (std::size_t i = 0; i < n; ++i)
            v0(i) = static_cast<float>(1 + (i * 7) % 5) * (i % 3 ? 1.0f : -1.0f);
        const auto p1 = mtl::itl::power_iteration<float>(A, v0, 5000, 1e-6f);
        const auto p2 = mtl::itl::power_iteration<fma_acc<float>>(A, v0, 5000, 1e-6f);
        const auto p3 = mtl::itl::power_iteration<exact_acc<float>>(A, v0, 5000, 1e-6f);
        const double e1 = std::fabs(p1.value - lam) / lam;
        const double e2 = std::fabs(p2.value - lam) / lam;
        const double e3 = std::fabs(p3.value - lam) / lam;
        INFO("n = " << n << ", k = " << k << ": plain " << e1 << ", fma " << e2 << ", exact " << e3
             << " (" << p3.iterations << " its)");
        CHECK(p3.converged);
        CHECK(e3 < bound);
        CHECK_FALSE(p1.converged);
        CHECK_FALSE(p2.converged);
        CHECK(e1 > 10.0 * bound);
        // Above the bound, not 10x: part of the float normalization runs on
        // the vendor BLAS, and with BLIS fma lands at 3.7x where OpenBLAS
        // gives more than 10x. Either way it does not reach the bound.
        CHECK(e2 > bound);
    }
}

TEST_CASE("[586:eigen] eigenvalue_symmetric and power_iteration: every configuration works",
          "[586:eigen][accumulator]") {
    // Measured: no ordering (header). lambda_max is accurate to a few float
    // ulps in each configuration; lambda_min of Hilbert is not improved by the
    // accumulator, so it is not asserted.
    for (std::size_t n = 4; n <= 7; ++n) {
        mtl::mat::dense2D<float> Hf(n, n);
        mtl::mat::dense2D<double> Hd(n, n);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j) {
                Hf(i, j) = static_cast<float>(1.0 / static_cast<double>(i + j + 1));
                Hd(i, j) = Hf(i, j);
            }
        auto top = [](const auto& ev) {
            double m = ev(0);
            for (std::size_t i = 1; i < ev.size(); ++i) m = std::max(m, static_cast<double>(ev(i)));
            return m;
        };
        const double ref = top(mtl::eigenvalue_symmetric_generic(Hd));
        INFO("Hilbert(" << n << ")");
        CHECK(std::fabs(top(mtl::eigenvalue_symmetric_generic<float>(Hf)) - ref) <= 1e-6 * ref);
        CHECK(std::fabs(top(mtl::eigenvalue_symmetric_generic<fma_acc<float>>(Hf)) - ref) <= 1e-6 * ref);
        CHECK(std::fabs(top(mtl::eigenvalue_symmetric_generic<exact_acc<float>>(Hf)) - ref) <= 1e-6 * ref);
    }

    // The 1-D Laplacian, lambda_max = 2 - 2 cos(n pi / (n + 1)).
    constexpr std::size_t n = 20;
    mtl::mat::dense2D<float> L(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            L(i, j) = i == j ? 2.0f : ((i + 1 == j || j + 1 == i) ? -1.0f : 0.0f);
    const double lam = 2.0 - 2.0 * std::cos(static_cast<double>(n) * std::numbers::pi /
                                            static_cast<double>(n + 1));
    fvec v0(n);
    for (std::size_t i = 0; i < n; ++i)
        v0(i) = (i % 2 ? -1.0f : 1.0f) * static_cast<float>(1 + i % 3);
    const auto p1 = mtl::itl::power_iteration<float>(L, v0, 20000, 1e-7f);
    const auto p2 = mtl::itl::power_iteration<fma_acc<float>>(L, v0, 20000, 1e-7f);
    const auto p3 = mtl::itl::power_iteration<exact_acc<float>>(L, v0, 20000, 1e-7f);
    CHECK(p1.converged);
    CHECK(p2.converged);
    CHECK(p3.converged);
    CHECK(std::fabs(p1.value - lam) <= 1e-6 * lam);
    CHECK(std::fabs(p2.value - lam) <= 1e-6 * lam);
    CHECK(std::fabs(p3.value - lam) <= 1e-6 * lam);
}
