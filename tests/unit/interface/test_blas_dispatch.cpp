// MTL5 Phase 14 -- Tests for BLAS dispatch paths
// These tests verify that the dispatch logic in mult.hpp and norms.hpp produces
// correct results. When MTL5_HAS_BLAS is defined, the BLAS-accelerated paths are
// exercised; otherwise the generic C++ fallback paths are tested.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/mult.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/math/identity.hpp>

using namespace mtl;
using Catch::Matchers::WithinAbs;

// -- GEMV dispatch -----------------------------------------------------------

TEST_CASE("mult mat*vec produces correct result", "[interface][blas]") {
    // A = [1 2; 3 4], x = [1; 1], y should be [3; 7]
    mat::dense2D<double> A(2, 2);
    A(0,0) = 1.0; A(0,1) = 2.0;
    A(1,0) = 3.0; A(1,1) = 4.0;

    vec::dense_vector<double> x = {1.0, 1.0};
    vec::dense_vector<double> y(2, 0.0);

    mult(A, x, y);

    REQUIRE_THAT(y(0), WithinAbs(3.0, 1e-12));
    REQUIRE_THAT(y(1), WithinAbs(7.0, 1e-12));
}

TEST_CASE("mult mat*vec with float", "[interface][blas]") {
    mat::dense2D<float> A(2, 2);
    A(0,0) = 1.0f; A(0,1) = 2.0f;
    A(1,0) = 3.0f; A(1,1) = 4.0f;

    vec::dense_vector<float> x = {1.0f, 1.0f};
    vec::dense_vector<float> y(2, 0.0f);

    mult(A, x, y);

    REQUIRE_THAT(y(0), WithinAbs(3.0f, 1e-5f));
    REQUIRE_THAT(y(1), WithinAbs(7.0f, 1e-5f));
}

TEST_CASE("mult mat*vec identity", "[interface][blas]") {
    constexpr int n = 4;
    mat::dense2D<double> I(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            I(i, j) = (i == j) ? 1.0 : 0.0;

    vec::dense_vector<double> x = {1.0, 2.0, 3.0, 4.0};
    vec::dense_vector<double> y(n, 0.0);

    mult(I, x, y);

    for (int i = 0; i < n; ++i)
        REQUIRE_THAT(y(i), WithinAbs(x(i), 1e-12));
}

// -- GEMM dispatch -----------------------------------------------------------

TEST_CASE("mult mat*mat produces correct result", "[interface][blas]") {
    // A = [1 2; 3 4], B = [5 6; 7 8], C = A*B = [19 22; 43 50]
    mat::dense2D<double> A(2, 2), B(2, 2), C(2, 2);
    A(0,0) = 1.0; A(0,1) = 2.0;
    A(1,0) = 3.0; A(1,1) = 4.0;
    B(0,0) = 5.0; B(0,1) = 6.0;
    B(1,0) = 7.0; B(1,1) = 8.0;

    mult(A, B, C);

    REQUIRE_THAT(C(0,0), WithinAbs(19.0, 1e-12));
    REQUIRE_THAT(C(0,1), WithinAbs(22.0, 1e-12));
    REQUIRE_THAT(C(1,0), WithinAbs(43.0, 1e-12));
    REQUIRE_THAT(C(1,1), WithinAbs(50.0, 1e-12));
}

TEST_CASE("mult mat*mat identity", "[interface][blas]") {
    constexpr int n = 3;
    mat::dense2D<double> A(n, n), I(n, n), C(n, n);

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            A(i, j) = static_cast<double>(i * n + j + 1);
            I(i, j) = (i == j) ? 1.0 : 0.0;
        }

    mult(A, I, C);

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            REQUIRE_THAT(C(i, j), WithinAbs(A(i, j), 1e-12));
}

TEST_CASE("mult mat*mat rectangular", "[interface][blas]") {
    // A(2x3) * B(3x2) = C(2x2)
    mat::dense2D<double> A(2, 3), B(3, 2), C(2, 2);
    A(0,0) = 1; A(0,1) = 2; A(0,2) = 3;
    A(1,0) = 4; A(1,1) = 5; A(1,2) = 6;
    B(0,0) = 7; B(0,1) = 8;
    B(1,0) = 9; B(1,1) = 10;
    B(2,0) = 11; B(2,1) = 12;

    mult(A, B, C);

    // C = [1*7+2*9+3*11, 1*8+2*10+3*12; 4*7+5*9+6*11, 4*8+5*10+6*12]
    //   = [58, 64; 139, 154]
    REQUIRE_THAT(C(0,0), WithinAbs(58.0, 1e-12));
    REQUIRE_THAT(C(0,1), WithinAbs(64.0, 1e-12));
    REQUIRE_THAT(C(1,0), WithinAbs(139.0, 1e-12));
    REQUIRE_THAT(C(1,1), WithinAbs(154.0, 1e-12));
}

// -- Norm dispatch -----------------------------------------------------------

TEST_CASE("two_norm dispatch produces correct result", "[interface][blas]") {
    vec::dense_vector<double> v = {3.0, 4.0};
    REQUIRE_THAT(two_norm(v), WithinAbs(5.0, 1e-12));
}

TEST_CASE("two_norm float dispatch", "[interface][blas]") {
    vec::dense_vector<float> v = {3.0f, 4.0f};
    REQUIRE_THAT(two_norm(v), WithinAbs(5.0f, 1e-5f));
}

// -- Dispatch trait checks ---------------------------------------------------

TEST_CASE("dispatch traits correctly identify BLAS-eligible types", "[interface][blas]") {
    STATIC_REQUIRE(interface::is_blas_scalar_v<float>);
    STATIC_REQUIRE(interface::is_blas_scalar_v<double>);
    STATIC_REQUIRE_FALSE(interface::is_blas_scalar_v<int>);
    STATIC_REQUIRE_FALSE(interface::is_blas_scalar_v<long double>);

    STATIC_REQUIRE(interface::BlasDenseMatrix<mat::dense2D<double>>);
    STATIC_REQUIRE(interface::BlasDenseMatrix<mat::dense2D<float>>);
    STATIC_REQUIRE_FALSE(interface::BlasDenseMatrix<mat::dense2D<int>>);

    STATIC_REQUIRE(interface::BlasDenseVector<vec::dense_vector<double>>);
    STATIC_REQUIRE(interface::BlasDenseVector<vec::dense_vector<float>>);
    STATIC_REQUIRE_FALSE(interface::BlasDenseVector<vec::dense_vector<int>>);
}
