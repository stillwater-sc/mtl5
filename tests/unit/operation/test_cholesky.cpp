#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/cholesky.hpp>
#include <mtl/operation/operators.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/operation/trans.hpp>
#include <mtl/generators/randspd.hpp>
#include <mtl/generators/pascal.hpp>
#include <mtl/generators/moler.hpp>
#include <mtl/generators/lehmer.hpp>

using namespace mtl;

TEST_CASE("Cholesky factorization: L*L^T reproduces A", "[operation][cholesky]") {
    // SPD matrix: A = {{4,2,1},{2,5,3},{1,3,6}}
    mat::dense2D<double> A(3, 3);
    A(0,0) = 4; A(0,1) = 2; A(0,2) = 1;
    A(1,0) = 2; A(1,1) = 5; A(1,2) = 3;
    A(2,0) = 1; A(2,1) = 3; A(2,2) = 6;

    mat::dense2D<double> Aorig(3, 3);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            Aorig(i, j) = A(i, j);

    int info = cholesky_factor(A);
    REQUIRE(info == 0);

    // Extract L from lower triangle of A
    mat::dense2D<double> L(3, 3);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            L(i, j) = (j <= i) ? A(i, j) : 0.0;

    // L * L^T should equal Aorig
    auto LLt = L * trans(L);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            REQUIRE_THAT(LLt(i, j), Catch::Matchers::WithinAbs(Aorig(i, j), 1e-10));
}

TEST_CASE("Cholesky solve", "[operation][cholesky]") {
    mat::dense2D<double> A(3, 3);
    A(0,0) = 4; A(0,1) = 2; A(0,2) = 1;
    A(1,0) = 2; A(1,1) = 5; A(1,2) = 3;
    A(2,0) = 1; A(2,1) = 3; A(2,2) = 6;

    mat::dense2D<double> Aorig(3, 3);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            Aorig(i, j) = A(i, j);

    vec::dense_vector<double> b = {1.0, 2.0, 3.0};
    vec::dense_vector<double> x(3);

    int info = cholesky_factor(A);
    REQUIRE(info == 0);
    cholesky_solve(A, x, b);

    // Verify Aorig * x = b
    auto r = Aorig * x;
    for (std::size_t i = 0; i < 3; ++i)
        REQUIRE_THAT(r(i), Catch::Matchers::WithinAbs(b(i), 1e-10));
}

TEST_CASE("Cholesky detects non-SPD matrix", "[operation][cholesky]") {
    // Not positive definite
    mat::dense2D<double> A(2, 2);
    A(0,0) = 1;  A(0,1) = 3;
    A(1,0) = 3;  A(1,1) = 1;

    int info = cholesky_factor(A);
    REQUIRE(info != 0);
}

// -- Generator-based Cholesky tests ------------------------------------

TEST_CASE("Cholesky on randspd with known eigenvalues", "[operation][cholesky][generator]") {
    constexpr std::size_t n = 5;
    auto A = generators::randspd<double>(n, {8.0, 4.0, 2.0, 1.0, 0.5});

    mat::dense2D<double> Aorig(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            Aorig(i, j) = A(i, j);

    int info = cholesky_factor(A);
    REQUIRE(info == 0);

    // Extract L and verify L*L^T = A
    mat::dense2D<double> L(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            L(i, j) = (j <= i) ? A(i, j) : 0.0;

    auto LLt = L * trans(L);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            REQUIRE_THAT(LLt(i, j), Catch::Matchers::WithinAbs(Aorig(i, j), 1e-10));
}

TEST_CASE("Cholesky on Pascal matrix", "[operation][cholesky][generator]") {
    constexpr std::size_t n = 6;
    auto A = generators::pascal<double>(n);

    mat::dense2D<double> Aorig(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            Aorig(i, j) = A(i, j);

    int info = cholesky_factor(A);
    REQUIRE(info == 0);

    mat::dense2D<double> L(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            L(i, j) = (j <= i) ? A(i, j) : 0.0;

    auto LLt = L * trans(L);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            REQUIRE_THAT(LLt(i, j), Catch::Matchers::WithinAbs(Aorig(i, j), 1e-10));
}

TEST_CASE("Cholesky on Moler matrix", "[operation][cholesky][generator]") {
    constexpr std::size_t n = 6;
    auto A = generators::moler<double>(n);

    mat::dense2D<double> Aorig(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            Aorig(i, j) = A(i, j);

    int info = cholesky_factor(A);
    REQUIRE(info == 0);

    mat::dense2D<double> L(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            L(i, j) = (j <= i) ? A(i, j) : 0.0;

    auto LLt = L * trans(L);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            REQUIRE_THAT(LLt(i, j), Catch::Matchers::WithinAbs(Aorig(i, j), 1e-8));
}

TEST_CASE("Cholesky on Lehmer matrix", "[operation][cholesky][generator]") {
    constexpr std::size_t n = 6;
    generators::lehmer<double> L_gen(n);
    mat::dense2D<double> A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A(i, j) = L_gen(i, j);

    mat::dense2D<double> Aorig(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            Aorig(i, j) = A(i, j);

    int info = cholesky_factor(A);
    REQUIRE(info == 0);

    mat::dense2D<double> L(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            L(i, j) = (j <= i) ? A(i, j) : 0.0;

    auto LLt = L * trans(L);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            REQUIRE_THAT(LLt(i, j), Catch::Matchers::WithinAbs(Aorig(i, j), 1e-8));
}
