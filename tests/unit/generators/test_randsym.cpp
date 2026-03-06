#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/generators/randsym.hpp>
#include <mtl/operation/eigenvalue_symmetric.hpp>
#include <mtl/operation/norms.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

using namespace mtl;
using Catch::Matchers::WithinAbs;

TEST_CASE("randsym dimensions", "[generators][randsym]") {
    std::vector<double> eigs = {5.0, 3.0, 1.0, 0.5};
    auto A = generators::randsym<double>(4, eigs);
    REQUIRE(A.num_rows() == 4);
    REQUIRE(A.num_cols() == 4);
}

TEST_CASE("randsym is symmetric", "[generators][randsym]") {
    std::vector<double> eigs = {4.0, 2.0, 1.0};
    auto A = generators::randsym<double>(3, eigs);

    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            REQUIRE_THAT(A(i, j), WithinAbs(A(j, i), 1e-10));
}

TEST_CASE("randsym recovers prescribed eigenvalues", "[generators][randsym]") {
    std::vector<double> prescribed = {10.0, 5.0, 2.0, 1.0};
    auto A = generators::randsym<double>(4, prescribed);

    auto computed = eigenvalue_symmetric(A, 1e-12);

    // Sort both for comparison
    std::vector<double> expected = prescribed;
    std::sort(expected.begin(), expected.end());

    REQUIRE(computed.size() == 4);
    for (std::size_t i = 0; i < 4; ++i) {
        REQUIRE_THAT(double(computed(i)), WithinAbs(expected[i], 0.1));
    }
}

TEST_CASE("randsym negative eigenvalues", "[generators][randsym]") {
    std::vector<double> prescribed = {-3.0, -1.0, 2.0, 5.0};
    auto A = generators::randsym<double>(4, prescribed);

    // Should still be symmetric
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            REQUIRE_THAT(A(i, j), WithinAbs(A(j, i), 1e-10));

    auto computed = eigenvalue_symmetric(A, 1e-12);
    std::vector<double> expected = prescribed;
    std::sort(expected.begin(), expected.end());

    for (std::size_t i = 0; i < 4; ++i) {
        REQUIRE_THAT(double(computed(i)), WithinAbs(expected[i], 0.1));
    }
}

TEST_CASE("randsym with kappa and mode", "[generators][randsym]") {
    auto A = generators::randsym<double>(5, 50.0, 3);
    REQUIRE(A.num_rows() == 5);
    REQUIRE(A.num_cols() == 5);

    // Should be symmetric
    for (std::size_t i = 0; i < 5; ++i)
        for (std::size_t j = 0; j < 5; ++j)
            REQUIRE_THAT(A(i, j), WithinAbs(A(j, i), 1e-10));
}
