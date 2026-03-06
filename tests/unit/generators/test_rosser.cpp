#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/generators/rosser.hpp>
#include <mtl/operation/norms.hpp>

#include <cmath>

using namespace mtl;
using Catch::Matchers::WithinAbs;

TEST_CASE("rosser dimensions", "[generators][rosser]") {
    auto R = generators::rosser<double>();
    REQUIRE(R.num_rows() == 8);
    REQUIRE(R.num_cols() == 8);
}

TEST_CASE("rosser is symmetric", "[generators][rosser]") {
    auto R = generators::rosser<double>();
    for (std::size_t i = 0; i < 8; ++i)
        for (std::size_t j = 0; j < 8; ++j)
            REQUIRE_THAT(R(i, j), WithinAbs(R(j, i), 1e-15));
}

TEST_CASE("rosser known corner values", "[generators][rosser]") {
    auto R = generators::rosser<double>();
    REQUIRE_THAT(R(0, 0), WithinAbs(611.0, 1e-15));
    REQUIRE_THAT(R(0, 1), WithinAbs(196.0, 1e-15));
    REQUIRE_THAT(R(1, 1), WithinAbs(899.0, 1e-15));
    REQUIRE_THAT(R(7, 7), WithinAbs(99.0, 1e-15));
    REQUIRE_THAT(R(6, 7), WithinAbs(-911.0, 1e-15));
    REQUIRE_THAT(R(4, 5), WithinAbs(-599.0, 1e-15));
}

TEST_CASE("rosser trace equals sum of eigenvalues", "[generators][rosser]") {
    auto R = generators::rosser<double>();

    // trace(R) = sum of diagonal = 611+899+899+611+411+411+99+99 = 4040
    double trace = 0.0;
    for (std::size_t i = 0; i < 8; ++i)
        trace += R(i, i);

    // Known eigenvalues sum: 0 + 1000 + 1000 + 1020 + 1020
    //   + (-10*sqrt(10405)) + (10*sqrt(10405)) + (510+100*sqrt(26))
    //   = 3040 + 510 + 100*sqrt(26) ≈ 3040 + 510 + 509.902 ≈ 4059.902
    // But trace should equal sum of eigenvalues exactly.
    // trace = 611+899+899+611+411+411+99+99 = 4040
    double expected_trace = 4040.0;
    REQUIRE_THAT(trace, WithinAbs(expected_trace, 1e-10));
}

TEST_CASE("rosser Frobenius norm", "[generators][rosser]") {
    auto R = generators::rosser<double>();

    // Frobenius norm is a fixed deterministic property of the hardcoded matrix
    double fnorm = frobenius_norm(R);
    REQUIRE_THAT(fnorm, WithinAbs(2482.257, 0.01));
}
