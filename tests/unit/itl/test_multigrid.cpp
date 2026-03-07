#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/operators.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/operation/dot.hpp>
#include <mtl/itl/smoother/gauss_seidel.hpp>
#include <mtl/itl/pc/identity.hpp>
#include <mtl/itl/iteration/basic_iteration.hpp>
#include <mtl/itl/krylov/cg.hpp>
#include <mtl/itl/mg/restriction.hpp>
#include <mtl/itl/mg/prolongation.hpp>
#include <mtl/itl/mg/multigrid.hpp>

using namespace mtl;

/// Build 1D Poisson matrix: -u'' on [0,1] with Dirichlet BCs.
/// Interior points only, so n = number of interior points.
/// A is n x n tridiagonal with 2 on diagonal, -1 on off-diagonals,
/// scaled by (n+1)^2.
static mat::compressed2D<double> make_poisson_1d(std::size_t n) {
    double h2_inv = static_cast<double>((n + 1) * (n + 1));
    mat::compressed2D<double> A(n, n);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        for (std::size_t i = 0; i < n; ++i) {
            ins[i][i] << 2.0 * h2_inv;
            if (i > 0)     ins[i][i-1] << -1.0 * h2_inv;
            if (i < n - 1) ins[i][i+1] << -1.0 * h2_inv;
        }
    }
    return A;
}

TEST_CASE("Restriction: full-weighting stencil", "[itl][mg][restriction]") {
    // Fine grid size 7, coarse grid size 3 = (7-1)/2
    auto R = itl::mg::make_restriction_1d(7);
    REQUIRE(R.num_rows() == 3);
    REQUIRE(R.num_cols() == 7);

    // Test with constant vector (should preserve constants)
    vec::dense_vector<double> fine(7, 1.0);
    auto coarse = itl::mg::restrict(R, fine);
    REQUIRE(coarse.size() == 3);
    for (std::size_t i = 0; i < 3; ++i)
        REQUIRE_THAT(coarse(i), Catch::Matchers::WithinAbs(1.0, 1e-12));

    // Test stencil: e_3 (fine index 3 = center of coarse 1) -> c(1) = 0.5
    vec::dense_vector<double> e3(7, 0.0);
    e3(3) = 1.0;
    auto c3 = itl::mg::restrict(R, e3);
    REQUIRE_THAT(c3(1), Catch::Matchers::WithinAbs(0.5, 1e-12));
    REQUIRE_THAT(c3(0), Catch::Matchers::WithinAbs(0.0, 1e-12));  // outside stencil

    // Test boundary stencil: e_2 is shared by coarse 0 and coarse 1
    vec::dense_vector<double> e2(7, 0.0);
    e2(2) = 1.0;
    auto c2 = itl::mg::restrict(R, e2);
    REQUIRE_THAT(c2(0), Catch::Matchers::WithinAbs(0.25, 1e-12));
    REQUIRE_THAT(c2(1), Catch::Matchers::WithinAbs(0.25, 1e-12));
}

TEST_CASE("Prolongation: linear interpolation", "[itl][mg][prolongation]") {
    // Coarse grid size 3, fine grid size 7 = 2*3+1
    auto P = itl::mg::make_prolongation_1d(3);
    REQUIRE(P.num_rows() == 7);
    REQUIRE(P.num_cols() == 3);

    // Test: constant coarse should give constant fine
    vec::dense_vector<double> coarse(3, 1.0);
    auto fine = itl::mg::prolongate(P, coarse);
    REQUIRE(fine.size() == 7);
    // Interior points get weight 1 or 0.5+0.5 = 1
    for (std::size_t i = 1; i < 6; ++i)
        REQUIRE_THAT(fine(i), Catch::Matchers::WithinAbs(1.0, 1e-12));

    // Edge points get 0.5 (only one neighbor)
    REQUIRE_THAT(fine(0), Catch::Matchers::WithinAbs(0.5, 1e-12));
    REQUIRE_THAT(fine(6), Catch::Matchers::WithinAbs(0.5, 1e-12));
}

TEST_CASE("Prolongation size doubling", "[itl][mg][prolongation]") {
    auto P = itl::mg::make_prolongation_1d(7);
    REQUIRE(P.num_rows() == 15);
    REQUIRE(P.num_cols() == 7);
}

TEST_CASE("V-cycle multigrid on 1D Poisson", "[itl][mg][multigrid]") {
    // 3-level hierarchy: n=15 -> n=7 -> n=3
    const std::size_t n_fine = 15;

    // Build hierarchy
    auto A0 = make_poisson_1d(15);
    auto A1 = make_poisson_1d(7);
    auto A2 = make_poisson_1d(3);

    auto R0 = itl::mg::make_restriction_1d(15);
    auto R1 = itl::mg::make_restriction_1d(7);

    auto P0 = itl::mg::make_prolongation_1d(7);
    auto P1 = itl::mg::make_prolongation_1d(3);

    std::vector<mat::compressed2D<double>> levels = {A0, A1, A2};
    std::vector<mat::compressed2D<double>> restrictors = {R0, R1};
    std::vector<mat::compressed2D<double>> prolongators = {P0, P1};

    // Smoother factory: Gauss-Seidel
    auto smoother_factory = [](const mat::compressed2D<double>& A) {
        return itl::smoother::gauss_seidel<mat::compressed2D<double>>(A);
    };

    // Coarse solver: just do many GS iterations
    auto coarse_solver = [&A2](vec::dense_vector<double>& x, const vec::dense_vector<double>& b) {
        itl::smoother::gauss_seidel<mat::compressed2D<double>> gs(A2);
        for (int i = 0; i < 50; ++i)
            gs(x, b);
    };

    itl::mg::multigrid<double> mg(levels, restrictors, prolongators,
                                   smoother_factory, coarse_solver, 2, 2);

    // RHS
    vec::dense_vector<double> b(n_fine, 1.0);
    vec::dense_vector<double> x(n_fine, 0.0);

    // Apply several V-cycles
    for (int cycle = 0; cycle < 20; ++cycle) {
        mg.vcycle(x, b);
    }

    // Check convergence: ||b - A*x|| / ||b|| should be small
    auto Ax = A0 * x;
    vec::dense_vector<double> r(n_fine);
    for (std::size_t i = 0; i < n_fine; ++i)
        r(i) = b(i) - Ax(i);
    double rel_resid = mtl::two_norm(r) / mtl::two_norm(b);
    REQUIRE(rel_resid < 1e-6);
}

TEST_CASE("Multigrid as preconditioner with CG", "[itl][mg][multigrid]") {
    // 2-level hierarchy: n=7 -> n=3
    auto A0 = make_poisson_1d(7);
    auto A1 = make_poisson_1d(3);

    auto R0 = itl::mg::make_restriction_1d(7);
    auto P0 = itl::mg::make_prolongation_1d(3);

    std::vector<mat::compressed2D<double>> levels = {A0, A1};
    std::vector<mat::compressed2D<double>> restrictors = {R0};
    std::vector<mat::compressed2D<double>> prolongators = {P0};

    auto smoother_factory = [](const mat::compressed2D<double>& A) {
        return itl::smoother::gauss_seidel<mat::compressed2D<double>>(A);
    };

    auto coarse_solver = [&A1](vec::dense_vector<double>& x, const vec::dense_vector<double>& b) {
        itl::smoother::gauss_seidel<mat::compressed2D<double>> gs(A1);
        for (int i = 0; i < 50; ++i)
            gs(x, b);
    };

    itl::mg::multigrid<double> mg(levels, restrictors, prolongators,
                                   smoother_factory, coarse_solver, 2, 2);

    // Use MG as preconditioner for CG
    vec::dense_vector<double> b(7, 1.0);
    vec::dense_vector<double> x(7, 0.0);

    itl::basic_iteration<double> iter(b, 50, 1e-10);
    int err = itl::cg(A0, x, b, mg, iter);
    REQUIRE(err == 0);

    auto Ax = A0 * x;
    for (std::size_t i = 0; i < 7; ++i)
        REQUIRE_THAT(Ax(i), Catch::Matchers::WithinAbs(b(i), 1e-6));
}
