#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/operators.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/operation/dot.hpp>
#include <mtl/itl/pc/identity.hpp>
#include <mtl/itl/pc/diagonal.hpp>
#include <mtl/itl/pc/ssor.hpp>
#include <mtl/itl/pc/ilu_0.hpp>
#include <mtl/itl/pc/ic_0.hpp>
#include <mtl/operation/mult.hpp>
#include <mtl/itl/krylov/cg.hpp>
#include <mtl/itl/iteration/basic_iteration.hpp>
#include <mtl/itl/krylov/bicg.hpp>

using namespace mtl;

TEST_CASE("BiCG on non-symmetric 3x3 dense system", "[itl][bicg]") {
    // Non-symmetric matrix
    mat::dense2D<double> A(3, 3);
    A(0,0) = 4; A(0,1) = 1; A(0,2) = 0;
    A(1,0) = 2; A(1,1) = 5; A(1,2) = 1;
    A(2,0) = 0; A(2,1) = 1; A(2,2) = 3;

    vec::dense_vector<double> b = {1.0, 2.0, 3.0};
    vec::dense_vector<double> x(3, 0.0);

    itl::pc::identity<mat::dense2D<double>> pc(A);
    itl::basic_iteration<double> iter(b, 100, 1e-10);

    int err = itl::bicg(A, x, b, pc, iter);

    REQUIRE(err == 0);

    // Verify A*x ~ b
    auto r = A * x;
    for (std::size_t i = 0; i < 3; ++i) {
        REQUIRE_THAT(r(i), Catch::Matchers::WithinAbs(b(i), 1e-8));
    }
}

TEST_CASE("BiCG on sparse tridiagonal system", "[itl][bicg][sparse]") {
    const std::size_t n = 10;
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
    itl::basic_iteration<double> iter(b, 200, 1e-10);

    int err = itl::bicg(A, x, b, pc, iter);

    REQUIRE(err == 0);

    // Verify A*x ~ b
    auto Ax = A * x;
    for (std::size_t i = 0; i < n; ++i) {
        REQUIRE_THAT(Ax(i), Catch::Matchers::WithinAbs(b(i), 1e-8));
    }
}

// ---------------------------------------------------------------------------
// #392: bicg computed rho from the wrong pair of vectors and therefore
// converged ONLY with pc::identity.
//
//   rho = z^T r~     preconditioned residual against the UNpreconditioned
//                    shadow residual (Barrett et al., Templates, 2.3.5)
//
// It used dot(z_tilde, z), which applies the preconditioner twice:
// z~ . z = (M^-T r~) . (M^-1 r) = r~^T M^-1 M^-1 r, where the algorithm needs
// r~^T M^-1 r. With identity, z == r and z~ == r~, so the two coincide.
//
// Which is precisely why it went unnoticed: every test in this file used
// pc::identity, the single configuration in which the defect is invisible.
// test_qmr.cpp does the same and qmr is broken the same way (#393);
// test_bicgstab.cpp exercises pc::diagonal and bicgstab is correct.
//
// So these cases all use a NON-identity preconditioner. That is the point of
// them, not an incidental detail.
// ---------------------------------------------------------------------------

namespace {

/// 2-D Laplacian, SPD. The diagonal is varied so a diagonal preconditioner is
/// not merely a scalar multiple of the identity -- otherwise the bug would stay
/// hidden even with pc::diagonal.
mat::compressed2D<double> laplacian_spd(std::size_t k) {
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

/// Solve A x = b for a known x, and return the relative infinity-norm error.
template <typename Mat, typename Prec>
double solve_err(const Mat& A, Prec& pc, int max_iter, int& iters, int& info) {
    const std::size_t n = A.num_rows();
    vec::dense_vector<double> xt(n);
    for (std::size_t i = 0; i < n; ++i)
        xt[i] = 1.0 + 0.5 * std::sin(static_cast<double>(i));
    vec::dense_vector<double> b(n);
    mtl::mult(A, xt, b);
    vec::dense_vector<double> x(n, 0.0);
    itl::basic_iteration<double> iter(b, max_iter, 1e-12);
    info = itl::bicg(A, x, b, pc, iter);
    iters = static_cast<int>(iter.iterations());
    double e = 0.0, nx = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        e  = std::max(e, std::abs(x(i) - xt(i)));
        nx = std::max(nx, std::abs(xt(i)));
    }
    return e / nx;
}

}  // namespace

TEST_CASE("BiCG converges with a non-identity preconditioner (#392)",
          "[itl][bicg][regression]") {
    const auto A = laplacian_spd(12);
    int iters = 0, info = 0;

    SECTION("diagonal") {
        // Was: 3000 iterations, relative error 2.25e-01.
        itl::pc::diagonal<mat::compressed2D<double>> pc(A);
        const double err = solve_err(A, pc, 3000, iters, info);
        INFO("iters = " << iters << ", rel err = " << err);
        REQUIRE(info == 0);
        REQUIRE(err < 1e-8);
        REQUIRE(iters < 100);          // was hitting the 3000 cap
    }

    SECTION("ssor") {
        itl::pc::ssor<mat::compressed2D<double>> pc(A);
        const double err = solve_err(A, pc, 3000, iters, info);
        INFO("iters = " << iters << ", rel err = " << err);
        REQUIRE(info == 0);
        REQUIRE(err < 1e-8);
    }

    SECTION("ilu_0") {
        itl::pc::ilu_0<double> pc(A);
        const double err = solve_err(A, pc, 3000, iters, info);
        INFO("iters = " << iters << ", rel err = " << err);
        REQUIRE(info == 0);
        REQUIRE(err < 1e-8);
    }

    SECTION("ic_0") {
        itl::pc::ic_0<double> pc(A);
        const double err = solve_err(A, pc, 3000, iters, info);
        INFO("iters = " << iters << ", rel err = " << err);
        REQUIRE(info == 0);
        REQUIRE(err < 1e-8);
    }
}

TEST_CASE("BiCG with a diagonal preconditioner matches CG on an SPD system (#392)",
          "[itl][bicg][regression]") {
    // The sharpest statement of the fix: on an SPD matrix, preconditioned BiCG
    // reduces to preconditioned CG, so it must take the same iteration count.
    // Before the fix bicg took 3000 (the cap) where cg took 33.
    const auto A = laplacian_spd(12);
    const std::size_t n = A.num_rows();

    vec::dense_vector<double> xt(n);
    for (std::size_t i = 0; i < n; ++i)
        xt[i] = 1.0 + 0.5 * std::sin(static_cast<double>(i));
    vec::dense_vector<double> b(n);
    mtl::mult(A, xt, b);

    itl::pc::diagonal<mat::compressed2D<double>> pc_cg(A), pc_bicg(A);

    vec::dense_vector<double> x_cg(n, 0.0), x_bicg(n, 0.0);
    itl::basic_iteration<double> it_cg(b, 3000, 1e-12), it_bicg(b, 3000, 1e-12);
    REQUIRE(itl::cg(A, x_cg, b, pc_cg, it_cg) == 0);
    REQUIRE(itl::bicg(A, x_bicg, b, pc_bicg, it_bicg) == 0);

    INFO("cg = " << it_cg.iterations() << ", bicg = " << it_bicg.iterations());
    REQUIRE(it_bicg.iterations() == it_cg.iterations());
    for (std::size_t i = 0; i < n; ++i)
        REQUIRE_THAT(x_bicg(i), Catch::Matchers::WithinAbs(x_cg(i), 1e-9));
}

TEST_CASE("BiCG with a symmetric preconditioner on a NON-symmetric system (#392)",
          "[itl][bicg][regression]") {
    // The fix is necessary and sufficient when the PRECONDITIONER is symmetric,
    // which is a statement about M and not about A. A diagonal preconditioner is
    // symmetric whatever A is, so this must converge.
    //
    // ssor/ilu_0 of a NON-symmetric A are themselves non-symmetric, and those
    // still fail -- for a different reason, the adjoint_solve stub tracked in
    // #394. They are deliberately not exercised here.
    const std::size_t k = 12, n = k * k;
    mat::compressed2D<double> A(n, n);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        for (std::size_t i = 0; i < k; ++i)
            for (std::size_t j = 0; j < k; ++j) {
                const std::size_t r = i * k + j;
                ins[r][r] << 4.0 * (1.0 + static_cast<double>(r % 7) * 0.1);
                if (i)         ins[r][r - k] << -1.0;
                if (i + 1 < k) ins[r][r + k] << -0.6;    // asymmetric
                if (j)         ins[r][r - 1] << -1.0;
                if (j + 1 < k) ins[r][r + 1] << -1.4;    // asymmetric
            }
    }
    itl::pc::diagonal<mat::compressed2D<double>> pc(A);
    int iters = 0, info = 0;
    const double err = solve_err(A, pc, 3000, iters, info);
    INFO("iters = " << iters << ", rel err = " << err);
    REQUIRE(info == 0);
    REQUIRE(err < 1e-8);
}
