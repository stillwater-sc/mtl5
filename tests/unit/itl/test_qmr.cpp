#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/operators.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/itl/pc/identity.hpp>
#include <mtl/itl/pc/diagonal.hpp>
#include <mtl/itl/pc/ssor.hpp>
#include <mtl/itl/pc/ilu_0.hpp>
#include <mtl/itl/pc/ic_0.hpp>
#include <mtl/operation/mult.hpp>
#include <mtl/itl/krylov/cg.hpp>
#include <mtl/itl/krylov/bicg.hpp>
#include <mtl/itl/iteration/basic_iteration.hpp>
#include <mtl/itl/krylov/qmr.hpp>

using namespace mtl;

TEST_CASE("QMR: dense 3x3 SPD system", "[itl][krylov][qmr]") {
    mat::dense2D<double> A(3, 3);
    A(0,0) = 4; A(0,1) = 1; A(0,2) = 0;
    A(1,0) = 1; A(1,1) = 4; A(1,2) = 1;
    A(2,0) = 0; A(2,1) = 1; A(2,2) = 4;

    vec::dense_vector<double> b = {1.0, 2.0, 3.0};
    vec::dense_vector<double> x(3, 0.0);

    itl::pc::identity<mat::dense2D<double>> pc(A);
    itl::basic_iteration<double> iter(b, 200, 1e-10);

    int err = itl::qmr(A, x, b, pc, iter);
    REQUIRE(err == 0);

    auto Ax = A * x;
    for (std::size_t i = 0; i < 3; ++i)
        REQUIRE_THAT(Ax(i), Catch::Matchers::WithinAbs(b(i), 1e-8));
}

TEST_CASE("QMR: sparse tridiagonal system", "[itl][krylov][qmr]") {
    const std::size_t n = 20;
    mat::compressed2D<double> A(n, n);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        for (std::size_t i = 0; i < n; ++i) {
            ins[i][i] << 4.0;
            if (i > 0)     ins[i][i-1] << -1.0;
            if (i < n - 1) ins[i][i+1] << -1.0;
        }
    }

    vec::dense_vector<double> b(n, 1.0);
    vec::dense_vector<double> x(n, 0.0);

    itl::pc::identity<mat::compressed2D<double>> pc(A);
    itl::basic_iteration<double> iter(b, 500, 1e-10);

    int err = itl::qmr(A, x, b, pc, iter);
    REQUIRE(err == 0);

    auto Ax = A * x;
    for (std::size_t i = 0; i < n; ++i)
        REQUIRE_THAT(Ax(i), Catch::Matchers::WithinAbs(b(i), 1e-8));
}

// ---------------------------------------------------------------------------
// #393: qmr converged only with pc::identity, the same symptom bicg had (#392)
// but a different cause.
//
// QMR uses a SPLIT preconditioner M = M1*M2 (Barrett et al., Templates Alg 7.3):
//
//     y = M1^-1 v~      z  = M2^-T w~
//     y~ = M2^-1 y      z~ = M1^-T z
//     p = y~ - ...      q  = z~ - ...
//
// The code declared left-only preconditioning -- M1 = M, M2 = I, so y~ = y and
// z~ = M^-T z -- but then applied M^-T when forming z and used z directly for q,
// which is the RIGHT-preconditioned placement. Half of each convention.
//
// The consequence lands on delta = z^T y, which became w~^T M^-1 M^-1 v~ where
// the algorithm needs w~^T M^-1 v~: the preconditioner applied twice, exactly
// the error #392 fixed in bicg. With identity the two coincide.
//
// As in test_bicg.cpp, every case here used pc::identity -- the one
// configuration where the defect is invisible. These use non-identity
// preconditioners deliberately.
// ---------------------------------------------------------------------------

namespace {

mat::compressed2D<double> qmr_laplacian_spd(std::size_t k) {
    const std::size_t n = k * k;
    mat::compressed2D<double> A(n, n);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        for (std::size_t i = 0; i < k; ++i)
            for (std::size_t j = 0; j < k; ++j) {
                const std::size_t r = i * k + j;
                ins[r][r] << 4.0 * (1.0 + static_cast<double>(r % 7) * 0.1);
                if (i)         ins[r][r - k] << -1.0;
                if (i + 1 < k) ins[r][r + k] << -1.0;
                if (j)         ins[r][r - 1] << -1.0;
                if (j + 1 < k) ins[r][r + 1] << -1.0;
            }
    }
    return A;
}

template <typename Mat, typename Prec>
double qmr_solve_err(const Mat& A, Prec& pc, int max_iter, int& iters, int& info) {
    const std::size_t n = A.num_rows();
    vec::dense_vector<double> xt(n);
    for (std::size_t i = 0; i < n; ++i)
        xt[i] = 1.0 + 0.5 * std::sin(static_cast<double>(i));
    vec::dense_vector<double> b(n);
    mtl::mult(A, xt, b);
    vec::dense_vector<double> x(n, 0.0);
    itl::basic_iteration<double> iter(b, max_iter, 1e-12);
    info = itl::qmr(A, x, b, pc, iter);
    iters = static_cast<int>(iter.iterations());
    double e = 0.0, nx = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        e  = std::max(e, std::abs(x(i) - xt(i)));
        nx = std::max(nx, std::abs(xt(i)));
    }
    return e / nx;
}

}  // namespace

TEST_CASE("QMR converges with a non-identity preconditioner (#393)",
          "[itl][qmr][regression]") {
    const auto A = qmr_laplacian_spd(12);
    int iters = 0, info = 0;

    SECTION("diagonal") {
        // Was: 3000 iterations, relative error 2.48e-01.
        itl::pc::diagonal<mat::compressed2D<double>> pc(A);
        const double err = qmr_solve_err(A, pc, 3000, iters, info);
        INFO("iters = " << iters << ", rel err = " << err);
        REQUIRE(info == 0);
        REQUIRE(err < 1e-8);
        REQUIRE(iters < 100);
    }

    SECTION("ssor") {
        itl::pc::ssor<mat::compressed2D<double>> pc(A);
        const double err = qmr_solve_err(A, pc, 3000, iters, info);
        INFO("iters = " << iters << ", rel err = " << err);
        REQUIRE(info == 0);
        REQUIRE(err < 1e-8);
    }

    SECTION("ilu_0") {
        itl::pc::ilu_0<double> pc(A);
        const double err = qmr_solve_err(A, pc, 3000, iters, info);
        INFO("iters = " << iters << ", rel err = " << err);
        REQUIRE(info == 0);
        REQUIRE(err < 1e-8);
    }

    SECTION("ic_0") {
        itl::pc::ic_0<double> pc(A);
        const double err = qmr_solve_err(A, pc, 3000, iters, info);
        INFO("iters = " << iters << ", rel err = " << err);
        REQUIRE(info == 0);
        REQUIRE(err < 1e-8);
    }
}

TEST_CASE("QMR tracks CG and BiCG on an SPD system (#393)",
          "[itl][qmr][regression]") {
    // On an SPD matrix with a symmetric preconditioner, BiCG reduces EXACTLY to
    // CG, so those two agree iteration for iteration. QMR does not: it minimizes
    // a QUASI-residual over the Krylov basis rather than the true residual, so
    // it may stop an iteration either side of CG. Measured here: cg 34, bicg 34,
    // qmr 33.
    //
    // So the bar is "tracks", not "equals" -- an equality assertion here would
    // be asserting a coincidence of stopping criteria rather than anything about
    // the preconditioning. What matters is that qmr is in the same regime and
    // lands on the same solution; before the fix it took 3000 where cg took 34.
    const auto A = qmr_laplacian_spd(12);
    const std::size_t n = A.num_rows();

    vec::dense_vector<double> xt(n);
    for (std::size_t i = 0; i < n; ++i)
        xt[i] = 1.0 + 0.5 * std::sin(static_cast<double>(i));
    vec::dense_vector<double> b(n);
    mtl::mult(A, xt, b);

    itl::pc::diagonal<mat::compressed2D<double>> p1(A), p2(A), p3(A);
    vec::dense_vector<double> x_cg(n, 0.0), x_bicg(n, 0.0), x_qmr(n, 0.0);
    itl::basic_iteration<double> i1(b, 3000, 1e-12), i2(b, 3000, 1e-12), i3(b, 3000, 1e-12);

    REQUIRE(itl::cg  (A, x_cg,   b, p1, i1) == 0);
    REQUIRE(itl::bicg(A, x_bicg, b, p2, i2) == 0);
    REQUIRE(itl::qmr (A, x_qmr,  b, p3, i3) == 0);

    INFO("cg = " << i1.iterations() << ", bicg = " << i2.iterations()
         << ", qmr = " << i3.iterations());
    // bicg == cg exactly: preconditioned BiCG IS preconditioned CG for SPD.
    REQUIRE(i2.iterations() == i1.iterations());
    // qmr within a couple of iterations, for the reason above.
    const int dq = static_cast<int>(i3.iterations()) - static_cast<int>(i1.iterations());
    REQUIRE(std::abs(dq) <= 2);
    REQUIRE(i3.iterations() < 100);          // was hitting the 3000 cap
    for (std::size_t i = 0; i < n; ++i)
        REQUIRE_THAT(x_qmr(i), Catch::Matchers::WithinAbs(x_cg(i), 1e-9));
}

TEST_CASE("QMR with a symmetric preconditioner on a NON-symmetric system (#393)",
          "[itl][qmr][regression]") {
    // As for bicg: the fix is necessary and sufficient when the PRECONDITIONER
    // is symmetric. ssor/ilu_0 of a non-symmetric A are themselves
    // non-symmetric and still fail, through the adjoint_solve stub in #394.
    const std::size_t k = 12, n = k * k;
    mat::compressed2D<double> A(n, n);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        for (std::size_t i = 0; i < k; ++i)
            for (std::size_t j = 0; j < k; ++j) {
                const std::size_t r = i * k + j;
                ins[r][r] << 4.0 * (1.0 + static_cast<double>(r % 7) * 0.1);
                if (i)         ins[r][r - k] << -1.0;
                if (i + 1 < k) ins[r][r + k] << -0.6;
                if (j)         ins[r][r - 1] << -1.0;
                if (j + 1 < k) ins[r][r + 1] << -1.4;
            }
    }
    itl::pc::diagonal<mat::compressed2D<double>> pc(A);
    int iters = 0, info = 0;
    const double err = qmr_solve_err(A, pc, 3000, iters, info);
    INFO("iters = " << iters << ", rel err = " << err);
    REQUIRE(info == 0);
    REQUIRE(err < 1e-8);
}
