#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/operators.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/operation/dot.hpp>
#include <mtl/itl/pc/identity.hpp>
#include <mtl/itl/pc/diagonal.hpp>
#include <mtl/itl/iteration/basic_iteration.hpp>
#include <mtl/itl/krylov/bicgstab_ell.hpp>

using namespace mtl;

static mat::compressed2D<double> make_tridiagonal(std::size_t n, double diag, double off) {
    mat::compressed2D<double> A(n, n);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        for (std::size_t i = 0; i < n; ++i) {
            ins[i][i] << diag;
            if (i > 0)     ins[i][i-1] << off;
            if (i < n - 1) ins[i][i+1] << off;
        }
    }
    return A;
}

TEST_CASE("BiCGSTAB(2) converges on tridiagonal system", "[itl][bicgstab_ell]") {
    const std::size_t n = 20;
    auto A = make_tridiagonal(n, 4.0, -1.0);

    vec::dense_vector<double> b(n, 1.0);
    vec::dense_vector<double> x(n, 0.0);

    itl::pc::identity<mat::compressed2D<double>> pc(A);
    itl::basic_iteration<double> iter(b, 200, 1e-10);

    int err = itl::bicgstab_ell(A, x, b, pc, iter, 2);
    REQUIRE(err == 0);

    auto Ax = A * x;
    for (std::size_t i = 0; i < n; ++i)
        REQUIRE_THAT(Ax(i), Catch::Matchers::WithinAbs(b(i), 1e-8));
}

TEST_CASE("BiCGSTAB(4) converges in fewer iterations than BiCGSTAB(2)", "[itl][bicgstab_ell]") {
    const std::size_t n = 40;
    auto A = make_tridiagonal(n, 4.0, -1.0);
    vec::dense_vector<double> b(n, 1.0);

    // BiCGSTAB(2)
    vec::dense_vector<double> x1(n, 0.0);
    itl::pc::identity<mat::compressed2D<double>> pc1(A);
    itl::basic_iteration<double> iter1(b, 500, 1e-10);
    itl::bicgstab_ell(A, x1, b, pc1, iter1, 2);
    int iters_2 = iter1.iterations();

    // BiCGSTAB(4)
    vec::dense_vector<double> x2(n, 0.0);
    itl::pc::identity<mat::compressed2D<double>> pc2(A);
    itl::basic_iteration<double> iter2(b, 500, 1e-10);
    itl::bicgstab_ell(A, x2, b, pc2, iter2, 4);
    int iters_4 = iter2.iterations();

    // Higher ell should converge in same or fewer outer iterations
    REQUIRE(iters_4 <= iters_2);
}

TEST_CASE("BiCGSTAB(2) with diagonal PC", "[itl][bicgstab_ell]") {
    const std::size_t n = 20;
    auto A = make_tridiagonal(n, 4.0, -1.0);

    vec::dense_vector<double> b(n, 1.0);
    vec::dense_vector<double> x(n, 0.0);

    itl::pc::diagonal<mat::compressed2D<double>> pc(A);
    itl::basic_iteration<double> iter(b, 200, 1e-10);

    int err = itl::bicgstab_ell(A, x, b, pc, iter, 2);
    REQUIRE(err == 0);

    auto Ax = A * x;
    for (std::size_t i = 0; i < n; ++i)
        REQUIRE_THAT(Ax(i), Catch::Matchers::WithinAbs(b(i), 1e-8));
}
