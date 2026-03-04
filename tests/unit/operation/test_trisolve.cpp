#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/lower_trisolve.hpp>
#include <mtl/operation/upper_trisolve.hpp>
#include <mtl/operation/operators.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/operation/trace.hpp>

using namespace mtl;

TEST_CASE("Lower triangular solve", "[operation][trisolve]") {
    // L = {{2,0,0},{1,3,0},{4,2,5}}
    mat::dense2D<double> L(3, 3);
    L(0,0) = 2; L(0,1) = 0; L(0,2) = 0;
    L(1,0) = 1; L(1,1) = 3; L(1,2) = 0;
    L(2,0) = 4; L(2,1) = 2; L(2,2) = 5;

    vec::dense_vector<double> b = {2.0, 7.0, 26.0};
    vec::dense_vector<double> x(3);

    lower_trisolve(L, x, b);

    // Verify L*x = b
    auto r = L * x;
    for (std::size_t i = 0; i < 3; ++i)
        REQUIRE_THAT(r(i), Catch::Matchers::WithinAbs(b(i), 1e-10));
}

TEST_CASE("Upper triangular solve", "[operation][trisolve]") {
    // U = {{3,1,2},{0,4,1},{0,0,2}}
    mat::dense2D<double> U(3, 3);
    U(0,0) = 3; U(0,1) = 1; U(0,2) = 2;
    U(1,0) = 0; U(1,1) = 4; U(1,2) = 1;
    U(2,0) = 0; U(2,1) = 0; U(2,2) = 2;

    vec::dense_vector<double> b = {10.0, 9.0, 4.0};
    vec::dense_vector<double> x(3);

    upper_trisolve(U, x, b);

    auto r = U * x;
    for (std::size_t i = 0; i < 3; ++i)
        REQUIRE_THAT(r(i), Catch::Matchers::WithinAbs(b(i), 1e-10));
}

TEST_CASE("Lower triangular solve with unit diagonal", "[operation][trisolve]") {
    // L = {{1,0,0},{3,1,0},{2,4,1}} (unit diagonal)
    mat::dense2D<double> L(3, 3);
    L(0,0) = 1; L(0,1) = 0; L(0,2) = 0;
    L(1,0) = 3; L(1,1) = 1; L(1,2) = 0;
    L(2,0) = 2; L(2,1) = 4; L(2,2) = 1;

    vec::dense_vector<double> b = {1.0, 5.0, 14.0};
    vec::dense_vector<double> x(3);

    lower_trisolve(L, x, b, true);

    auto r = L * x;
    for (std::size_t i = 0; i < 3; ++i)
        REQUIRE_THAT(r(i), Catch::Matchers::WithinAbs(b(i), 1e-10));
}

TEST_CASE("Trace of a matrix", "[operation][trace]") {
    mat::dense2D<double> A(3, 3);
    A(0,0) = 1; A(0,1) = 2; A(0,2) = 3;
    A(1,0) = 4; A(1,1) = 5; A(1,2) = 6;
    A(2,0) = 7; A(2,1) = 8; A(2,2) = 9;

    REQUIRE_THAT(trace(A), Catch::Matchers::WithinAbs(15.0, 1e-12));
}
