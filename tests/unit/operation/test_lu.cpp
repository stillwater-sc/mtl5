#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/lu.hpp>
#include <mtl/operation/inv.hpp>
#include <mtl/operation/operators.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/generators/frank.hpp>
#include <mtl/generators/moler.hpp>
#include <mtl/generators/hilbert.hpp>
#include <mtl/generators/pascal.hpp>

using namespace mtl;

TEST_CASE("LU factorization and solve", "[operation][lu]") {
    mat::dense2D<double> A(3, 3);
    A(0,0) = 2; A(0,1) = 1; A(0,2) = 1;
    A(1,0) = 4; A(1,1) = 3; A(1,2) = 3;
    A(2,0) = 8; A(2,1) = 7; A(2,2) = 9;

    vec::dense_vector<double> b = {4.0, 10.0, 24.0};

    // Copy A for factoring (it gets modified)
    mat::dense2D<double> LU(3, 3);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            LU(i, j) = A(i, j);

    std::vector<std::size_t> pivot;
    int info = lu_factor(LU, pivot);
    REQUIRE(info == 0);

    vec::dense_vector<double> x(3);
    lu_solve(LU, pivot, x, b);

    // Verify A*x = b
    auto r = A * x;
    for (std::size_t i = 0; i < 3; ++i)
        REQUIRE_THAT(r(i), Catch::Matchers::WithinAbs(b(i), 1e-10));
}

TEST_CASE("LU convenience function lu_apply", "[operation][lu]") {
    mat::dense2D<double> A(3, 3);
    A(0,0) = 1; A(0,1) = 2; A(0,2) = 3;
    A(1,0) = 4; A(1,1) = 5; A(1,2) = 6;
    A(2,0) = 7; A(2,1) = 8; A(2,2) = 10;

    // Save original
    mat::dense2D<double> Aorig(3, 3);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            Aorig(i, j) = A(i, j);

    vec::dense_vector<double> b = {1.0, 2.0, 3.0};
    vec::dense_vector<double> x(3);

    int info = lu_apply(A, x, b);
    REQUIRE(info == 0);

    auto r = Aorig * x;
    for (std::size_t i = 0; i < 3; ++i)
        REQUIRE_THAT(r(i), Catch::Matchers::WithinAbs(b(i), 1e-10));
}

TEST_CASE("Matrix inverse via LU", "[operation][inv]") {
    mat::dense2D<double> A(3, 3);
    A(0,0) = 4; A(0,1) = 7; A(0,2) = 2;
    A(1,0) = 3; A(1,1) = 6; A(1,2) = 1;
    A(2,0) = 2; A(2,1) = 5; A(2,2) = 3;

    auto Ainv = inv(A);

    // A * Ainv should be approximately I
    auto I_approx = A * Ainv;
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            REQUIRE_THAT(I_approx(i, j), Catch::Matchers::WithinAbs(expected, 1e-10));
        }
}

// -- Generator-based LU tests -----------------------------------------

TEST_CASE("LU solve on 8x8 Frank matrix", "[operation][lu][generator]") {
    constexpr std::size_t n = 8;
    auto A = generators::frank<double>(n);

    mat::dense2D<double> Aorig(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            Aorig(i, j) = A(i, j);

    // Known solution
    vec::dense_vector<double> x_true(n);
    for (std::size_t i = 0; i < n; ++i)
        x_true(i) = static_cast<double>(i + 1);

    auto b = Aorig * x_true;

    vec::dense_vector<double> x(n);
    int info = lu_apply(A, x, b);
    REQUIRE(info == 0);

    // Verify backward error
    auto residual = Aorig * x - b;
    double rel_residual = two_norm(residual) / two_norm(b);
    REQUIRE(rel_residual < 1e-10);
}

TEST_CASE("LU solve on Moler matrix", "[operation][lu][generator]") {
    constexpr std::size_t n = 6;
    auto A = generators::moler<double>(n);

    mat::dense2D<double> Aorig(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            Aorig(i, j) = A(i, j);

    vec::dense_vector<double> x_true(n);
    for (std::size_t i = 0; i < n; ++i)
        x_true(i) = static_cast<double>(i + 1);

    auto b = Aorig * x_true;

    vec::dense_vector<double> x(n);
    int info = lu_apply(A, x, b);
    REQUIRE(info == 0);

    auto residual = Aorig * x - b;
    double rel_residual = two_norm(residual) / two_norm(b);
    REQUIRE(rel_residual < 1e-8);
}

TEST_CASE("LU on ill-conditioned Hilbert 6x6", "[operation][lu][generator]") {
    constexpr std::size_t n = 6;
    generators::hilbert<double> H_gen(n);
    mat::dense2D<double> A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A(i, j) = H_gen(i, j);

    mat::dense2D<double> Aorig(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            Aorig(i, j) = A(i, j);

    vec::dense_vector<double> x_true(n);
    for (std::size_t i = 0; i < n; ++i)
        x_true(i) = 1.0;

    auto b = Aorig * x_true;

    vec::dense_vector<double> x(n);
    int info = lu_apply(A, x, b);
    REQUIRE(info == 0);

    // Hilbert is ill-conditioned: check backward error, not forward error
    auto residual = Aorig * x - b;
    double rel_residual = two_norm(residual) / two_norm(b);
    REQUIRE(rel_residual < 1e-4);
}

TEST_CASE("LU inverse of Pascal matrix", "[operation][lu][generator]") {
    constexpr std::size_t n = 5;
    auto A = generators::pascal<double>(n);

    auto Ainv = inv(A);

    // Pascal has det=1, well-conditioned: A * A^{-1} should be I
    auto I_approx = A * Ainv;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            REQUIRE_THAT(I_approx(i, j), Catch::Matchers::WithinAbs(expected, 1e-10));
        }
}
