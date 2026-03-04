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
#include <mtl/itl/iteration/basic_iteration.hpp>
#include <mtl/itl/krylov/gmres.hpp>

using namespace mtl;

TEST_CASE("GMRES on 3x3 non-symmetric system", "[itl][gmres]") {
    mat::dense2D<double> A(3, 3);
    A(0,0) = 4; A(0,1) = 1; A(0,2) = 0;
    A(1,0) = 2; A(1,1) = 5; A(1,2) = 1;
    A(2,0) = 0; A(2,1) = 1; A(2,2) = 3;

    vec::dense_vector<double> b = {1.0, 2.0, 3.0};
    vec::dense_vector<double> x(3, 0.0);

    itl::pc::identity<mat::dense2D<double>> pc(A);
    itl::basic_iteration<double> iter(b, 100, 1e-10);

    int err = itl::gmres(A, x, b, pc, iter, 30);

    REQUIRE(err == 0);

    auto r = A * x;
    for (std::size_t i = 0; i < 3; ++i) {
        REQUIRE_THAT(r(i), Catch::Matchers::WithinAbs(b(i), 1e-8));
    }
}

TEST_CASE("GMRES with restart on 10x10 system", "[itl][gmres][restart]") {
    const std::size_t n = 10;
    mat::dense2D<double> A(n, n);
    // Zero out
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A(i, j) = 0.0;
    // Diagonally dominant non-symmetric
    for (std::size_t i = 0; i < n; ++i) {
        A(i, i) = 5.0;
        if (i > 0)     A(i, i-1) = -1.0;
        if (i < n - 1) A(i, i+1) = -2.0;  // non-symmetric
    }

    vec::dense_vector<double> b(n, 1.0);
    vec::dense_vector<double> x(n, 0.0);

    itl::pc::identity<mat::dense2D<double>> pc(A);
    itl::basic_iteration<double> iter(b, 200, 1e-10);

    // Use restart=3 to force multiple restarts
    int err = itl::gmres(A, x, b, pc, iter, 3);

    REQUIRE(err == 0);

    auto Ax = A * x;
    for (std::size_t i = 0; i < n; ++i) {
        REQUIRE_THAT(Ax(i), Catch::Matchers::WithinAbs(b(i), 1e-7));
    }
}

TEST_CASE("GMRES with diagonal preconditioner", "[itl][gmres][pc]") {
    const std::size_t n = 5;
    mat::dense2D<double> A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A(i, j) = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        A(i, i) = 10.0;
        if (i > 0)     A(i, i-1) = -1.0;
        if (i < n - 1) A(i, i+1) = -2.0;
    }

    vec::dense_vector<double> b(n, 1.0);
    vec::dense_vector<double> x(n, 0.0);

    itl::pc::diagonal<mat::dense2D<double>> pc(A);
    itl::basic_iteration<double> iter(b, 100, 1e-10);

    int err = itl::gmres(A, x, b, pc, iter);

    REQUIRE(err == 0);

    auto Ax = A * x;
    for (std::size_t i = 0; i < n; ++i) {
        REQUIRE_THAT(Ax(i), Catch::Matchers::WithinAbs(b(i), 1e-8));
    }
}
