#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <sstream>

#include <mtl/vec/dense_vector.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/operation/operators.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/operation/dot.hpp>
#include <mtl/itl/pc/identity.hpp>
#include <mtl/itl/pc/diagonal.hpp>
#include <mtl/itl/iteration/basic_iteration.hpp>
#include <mtl/itl/iteration/noisy_iteration.hpp>
#include <mtl/itl/krylov/cg.hpp>

using namespace mtl;

TEST_CASE("CG converges on 3x3 SPD system with identity PC", "[itl][cg]") {
    // A = {{4,1,0},{1,3,1},{0,1,2}} -- symmetric positive definite
    mat::dense2D<double> A(3, 3);
    A(0,0) = 4; A(0,1) = 1; A(0,2) = 0;
    A(1,0) = 1; A(1,1) = 3; A(1,2) = 1;
    A(2,0) = 0; A(2,1) = 1; A(2,2) = 2;

    // b = {1, 2, 3}
    vec::dense_vector<double> b = {1.0, 2.0, 3.0};

    // x = 0 (initial guess)
    vec::dense_vector<double> x(3, 0.0);

    itl::pc::identity<mat::dense2D<double>> pc(A);
    itl::basic_iteration<double> iter(b, 100, 1e-10);

    int err = itl::cg(A, x, b, pc, iter);

    REQUIRE(err == 0);
    REQUIRE(iter.iterations() <= 3); // CG converges in at most n steps for nxn

    // Verify A*x ~= b
    auto r = A * x;
    for (std::size_t i = 0; i < 3; ++i) {
        REQUIRE_THAT(r(i), Catch::Matchers::WithinAbs(b(i), 1e-8));
    }
}

TEST_CASE("CG converges with diagonal (Jacobi) PC on tridiagonal", "[itl][cg]") {
    // 1D Laplacian: tridiagonal with 2 on diagonal, -1 on off-diagonals
    const std::size_t n = 10;
    mat::dense2D<double> A(n, n);
    // Zero out the matrix first
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A(i, j) = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        A(i, i) = 2.0;
        if (i > 0)     A(i, i-1) = -1.0;
        if (i < n - 1) A(i, i+1) = -1.0;
    }

    // b = 1
    vec::dense_vector<double> b(n, 1.0);
    vec::dense_vector<double> x(n, 0.0);

    itl::pc::diagonal<mat::dense2D<double>> pc(A);
    itl::basic_iteration<double> iter(b, 200, 1e-10);

    int err = itl::cg(A, x, b, pc, iter);

    REQUIRE(err == 0);

    // Verify A*x ~= b
    auto Ax = A * x;
    for (std::size_t i = 0; i < n; ++i) {
        REQUIRE_THAT(Ax(i), Catch::Matchers::WithinAbs(b(i), 1e-8));
    }
}

TEST_CASE("basic_iteration reports max_iter exceeded", "[itl][iteration]") {
    vec::dense_vector<double> r0 = {1.0, 1.0};
    itl::basic_iteration<double> iter(r0, 2, 1e-15); // max 2 iterations, very tight tol

    // Drive with a non-converging residual
    int count = 0;
    double fake_resid = 1.0;
    while (!iter.finished(fake_resid)) {
        ++iter;
        fake_resid *= 0.5; // halve each time, but won't reach 1e-15 in 2 steps
        ++count;
    }

    REQUIRE(iter.error_code() == 1); // max_iter exceeded
    REQUIRE(count == 2);
}

TEST_CASE("noisy_iteration prints to ostream", "[itl][iteration]") {
    std::ostringstream oss;
    vec::dense_vector<double> r0 = {3.0, 4.0}; // norm = 5
    itl::noisy_iteration<double> iter(r0, 5, 1e-15, 0.0, oss);

    // Run a few iterations with fake residuals
    iter.finished(5.0); // initial check
    ++iter;
    iter.finished(2.5);
    ++iter;
    iter.finished(1.0);

    std::string output = oss.str();
    REQUIRE(!output.empty());
    // Should contain "iteration" and "resid"
    REQUIRE(output.find("iteration") != std::string::npos);
    REQUIRE(output.find("resid") != std::string::npos);
}
