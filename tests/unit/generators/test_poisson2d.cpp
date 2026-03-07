#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/generators/poisson.hpp>
#include <mtl/generators/laplacian.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/operators.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/operation/dot.hpp>
#include <mtl/itl/krylov/cg.hpp>
#include <mtl/itl/pc/identity.hpp>
#include <mtl/itl/iteration/basic_iteration.hpp>

using namespace mtl;
using Catch::Matchers::WithinAbs;

TEST_CASE("poisson2d_dirichlet dimensions", "[generators][poisson]") {
    auto P = generators::poisson2d_dirichlet<double>(3, 3);
    REQUIRE(P.num_rows() == 9);
    REQUIRE(P.num_cols() == 9);
}

TEST_CASE("poisson2d_dirichlet diagonal values", "[generators][poisson]") {
    std::size_t nx = 3, ny = 3;
    auto P = generators::poisson2d_dirichlet<double>(nx, ny);

    double hx = 1.0 / (nx + 1);
    double hy = 1.0 / (ny + 1);
    double expected_diag = 2.0 / (hx * hx) + 2.0 / (hy * hy);

    for (std::size_t i = 0; i < nx * ny; ++i)
        REQUIRE_THAT(P(i, i), WithinAbs(expected_diag, 1e-10));
}

TEST_CASE("poisson2d_dirichlet is symmetric", "[generators][poisson]") {
    auto P = generators::poisson2d_dirichlet<double>(3, 4);
    std::size_t n = 12;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            REQUIRE_THAT(P(i, j), WithinAbs(P(j, i), 1e-10));
}

TEST_CASE("poisson2d_dirichlet SPD: CG converges", "[generators][poisson]") {
    std::size_t nx = 4, ny = 4;
    auto A = generators::poisson2d_dirichlet<double>(nx, ny);
    std::size_t n = nx * ny;

    // RHS: b = ones
    vec::dense_vector<double> b(n, 1.0);
    vec::dense_vector<double> x(n, 0.0);

    itl::pc::identity<decltype(A)> pc(A);
    itl::basic_iteration<double> iter(b, 1000, 1.0e-10);

    int err = itl::cg(A, x, b, pc, iter);
    REQUIRE(err == 0);
    REQUIRE(iter.iterations() > 0);
}

TEST_CASE("poisson2d_dirichlet uniform grid relation to laplacian", "[generators][poisson]") {
    // For uniform grid (nx==ny): poisson = (nx+1)^2 * laplacian_2d
    std::size_t nx = 3;
    auto P = generators::poisson2d_dirichlet<double>(nx, nx);
    auto L = generators::laplacian_2d<double>(nx, nx);
    double scale = static_cast<double>((nx + 1) * (nx + 1));

    std::size_t n = nx * nx;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            REQUIRE_THAT(P(i, j), WithinAbs(scale * L(i, j), 1e-10));
}
