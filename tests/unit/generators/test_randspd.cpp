#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/generators/randspd.hpp>
#include <mtl/operation/eigenvalue_symmetric.hpp>
#include <mtl/operation/cholesky.hpp>
#include <mtl/operation/norms.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

using namespace mtl;
using Catch::Matchers::WithinAbs;

TEST_CASE("randspd dimensions", "[generators][randspd]") {
    std::vector<double> eigs = {5.0, 3.0, 1.0};
    auto A = generators::randspd<double>(3, eigs);
    REQUIRE(A.num_rows() == 3);
    REQUIRE(A.num_cols() == 3);
}

TEST_CASE("randspd is symmetric", "[generators][randspd]") {
    std::vector<double> eigs = {4.0, 2.0, 1.0};
    auto A = generators::randspd<double>(3, eigs);

    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            REQUIRE_THAT(A(i, j), WithinAbs(A(j, i), 1e-10));
}

TEST_CASE("randspd eigenvalues are positive and match prescribed", "[generators][randspd]") {
    std::vector<double> prescribed = {8.0, 4.0, 2.0, 1.0};
    auto A = generators::randspd<double>(4, prescribed);

    auto computed = eigenvalue_symmetric(A, 1e-12);

    std::vector<double> expected = prescribed;
    std::sort(expected.begin(), expected.end());

    REQUIRE(computed.size() == 4);
    for (std::size_t i = 0; i < 4; ++i) {
        REQUIRE(double(computed(i)) > 0.0);  // all positive
        REQUIRE_THAT(double(computed(i)), WithinAbs(expected[i], 0.1));
    }
}

TEST_CASE("randspd Cholesky succeeds", "[generators][randspd]") {
    std::vector<double> eigs = {5.0, 3.0, 1.0, 0.5};
    auto A = generators::randspd<double>(4, eigs);

    // Cholesky should succeed for SPD matrix
    mat::dense2D<double> L(4, 4);
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            L(i, j) = A(i, j);

    int result = cholesky_factor(L);
    REQUIRE(result == 0);
}

TEST_CASE("randspd with kappa and mode", "[generators][randspd]") {
    auto A = generators::randspd<double>(5, 100.0, 3);
    REQUIRE(A.num_rows() == 5);
    REQUIRE(A.num_cols() == 5);

    // Should be symmetric
    for (std::size_t i = 0; i < 5; ++i)
        for (std::size_t j = 0; j < 5; ++j)
            REQUIRE_THAT(A(i, j), WithinAbs(A(j, i), 1e-10));

    // Cholesky should succeed
    mat::dense2D<double> L(5, 5);
    for (std::size_t i = 0; i < 5; ++i)
        for (std::size_t j = 0; j < 5; ++j)
            L(i, j) = A(i, j);
    REQUIRE(cholesky_factor(L) == 0);
}

TEST_CASE("randspd with kappa eigenvalues in correct range", "[generators][randspd]") {
    double kappa = 50.0;
    auto A = generators::randspd<double>(4, kappa, 4);

    auto computed = eigenvalue_symmetric(A, 1e-12);

    // All eigenvalues should be in [1/kappa, 1]
    for (std::size_t i = 0; i < 4; ++i) {
        double ev = double(computed(i));
        REQUIRE(ev > 0.0);
        REQUIRE(ev >= 1.0 / kappa - 0.01);
        REQUIRE(ev <= 1.0 + 0.01);
    }
}
