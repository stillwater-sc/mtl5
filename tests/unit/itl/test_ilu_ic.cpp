#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/operators.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/operation/dot.hpp>
#include <mtl/itl/pc/identity.hpp>
#include <mtl/itl/pc/ilu_0.hpp>
#include <mtl/itl/pc/ic_0.hpp>
#include <mtl/itl/iteration/basic_iteration.hpp>
#include <mtl/itl/krylov/bicgstab.hpp>
#include <mtl/itl/krylov/cg.hpp>

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

TEST_CASE("ILU(0) preconditioned BiCGSTAB on tridiagonal", "[itl][pc][ilu_0]") {
    const std::size_t n = 20;
    auto A = make_tridiagonal(n, 4.0, -1.0);

    vec::dense_vector<double> b(n, 1.0);
    vec::dense_vector<double> x(n, 0.0);

    itl::pc::ilu_0<double> pc(A);
    itl::basic_iteration<double> iter(b, 200, 1e-10);

    int err = itl::bicgstab(A, x, b, pc, iter);
    REQUIRE(err == 0);

    auto Ax = A * x;
    for (std::size_t i = 0; i < n; ++i)
        REQUIRE_THAT(Ax(i), Catch::Matchers::WithinAbs(b(i), 1e-8));
}

TEST_CASE("ILU(0) converges faster than identity PC", "[itl][pc][ilu_0]") {
    const std::size_t n = 30;
    auto A = make_tridiagonal(n, 4.0, -1.0);
    vec::dense_vector<double> b(n, 1.0);

    // With identity PC
    vec::dense_vector<double> x1(n, 0.0);
    itl::pc::identity<mat::compressed2D<double>> id_pc(A);
    itl::basic_iteration<double> iter1(b, 500, 1e-10);
    itl::bicgstab(A, x1, b, id_pc, iter1);
    int iters_identity = iter1.iterations();

    // With ILU(0) PC
    vec::dense_vector<double> x2(n, 0.0);
    itl::pc::ilu_0<double> ilu_pc(A);
    itl::basic_iteration<double> iter2(b, 500, 1e-10);
    itl::bicgstab(A, x2, b, ilu_pc, iter2);
    int iters_ilu = iter2.iterations();

    REQUIRE(iters_ilu <= iters_identity);
}

TEST_CASE("IC(0) preconditioned CG on SPD tridiagonal", "[itl][pc][ic_0]") {
    const std::size_t n = 20;
    auto A = make_tridiagonal(n, 4.0, -1.0);

    vec::dense_vector<double> b(n, 1.0);
    vec::dense_vector<double> x(n, 0.0);

    itl::pc::ic_0<double> pc(A);
    itl::basic_iteration<double> iter(b, 200, 1e-10);

    int err = itl::cg(A, x, b, pc, iter);
    REQUIRE(err == 0);

    auto Ax = A * x;
    for (std::size_t i = 0; i < n; ++i)
        REQUIRE_THAT(Ax(i), Catch::Matchers::WithinAbs(b(i), 1e-8));
}
