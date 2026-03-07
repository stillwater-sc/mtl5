#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/vec/dense_vector.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/operation/operators.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/operation/dot.hpp>
#include <mtl/itl/pc/identity.hpp>
#include <mtl/itl/pc/diagonal.hpp>
#include <mtl/itl/iteration/basic_iteration.hpp>
#include <mtl/itl/krylov/bicgstab.hpp>

using namespace mtl;

TEST_CASE("BiCGSTAB converges on non-symmetric 3x3 with identity PC", "[itl][bicgstab]") {
    // Non-symmetric matrix
    mat::dense2D<double> A(3, 3);
    A(0,0) = 4; A(0,1) = 1; A(0,2) = 0;
    A(1,0) = 2; A(1,1) = 5; A(1,2) = 1;
    A(2,0) = 0; A(2,1) = 1; A(2,2) = 3;

    vec::dense_vector<double> b = {1.0, 2.0, 3.0};
    vec::dense_vector<double> x(3, 0.0);

    itl::pc::identity<mat::dense2D<double>> pc(A);
    itl::basic_iteration<double> iter(b, 100, 1e-10);

    int err = itl::bicgstab(A, x, b, pc, iter);

    REQUIRE(err == 0);

    // Verify A*x ~= b
    auto Ax = A * x;
    for (std::size_t i = 0; i < 3; ++i) {
        REQUIRE_THAT(Ax(i), Catch::Matchers::WithinAbs(b(i), 1e-8));
    }
}

TEST_CASE("BiCGSTAB converges with diagonal PC", "[itl][bicgstab]") {
    // Diagonally dominant non-symmetric system
    const std::size_t n = 8;
    mat::dense2D<double> A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A(i, j) = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        A(i, i) = 4.0;
        if (i > 0)     A(i, i-1) = -1.0;
        if (i < n - 1) A(i, i+1) = -0.5; // asymmetric off-diagonals
    }

    vec::dense_vector<double> b(n, 1.0);
    vec::dense_vector<double> x(n, 0.0);

    itl::pc::diagonal<mat::dense2D<double>> pc(A);
    itl::basic_iteration<double> iter(b, 200, 1e-10);

    int err = itl::bicgstab(A, x, b, pc, iter);

    REQUIRE(err == 0);

    // Verify A*x ~= b
    auto Ax = A * x;
    for (std::size_t i = 0; i < n; ++i) {
        REQUIRE_THAT(Ax(i), Catch::Matchers::WithinAbs(b(i), 1e-8));
    }
}

TEST_CASE("BiCGSTAB converges on SPD system too", "[itl][bicgstab]") {
    // BiCGSTAB should also work on SPD (though CG is preferred)
    mat::dense2D<double> A(3, 3);
    A(0,0) = 4; A(0,1) = 1; A(0,2) = 0;
    A(1,0) = 1; A(1,1) = 3; A(1,2) = 1;
    A(2,0) = 0; A(2,1) = 1; A(2,2) = 2;

    vec::dense_vector<double> b = {5.0, 5.0, 5.0};
    vec::dense_vector<double> x(3, 0.0);

    itl::pc::identity<mat::dense2D<double>> pc(A);
    itl::basic_iteration<double> iter(b, 100, 1e-10);

    int err = itl::bicgstab(A, x, b, pc, iter);

    REQUIRE(err == 0);

    auto Ax = A * x;
    for (std::size_t i = 0; i < 3; ++i) {
        REQUIRE_THAT(Ax(i), Catch::Matchers::WithinAbs(b(i), 1e-8));
    }
}
