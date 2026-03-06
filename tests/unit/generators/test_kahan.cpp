#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/generators/kahan.hpp>
#include <cmath>

using namespace mtl;
using Catch::Matchers::WithinAbs;

TEST_CASE("kahan is upper triangular", "[generators][kahan]") {
    auto K = generators::kahan<double>(5);

    REQUIRE(K.num_rows() == 5);
    REQUIRE(K.num_cols() == 5);

    // Below diagonal must be zero
    for (std::size_t i = 1; i < 5; ++i)
        for (std::size_t j = 0; j < i; ++j)
            REQUIRE_THAT(K(i, j), WithinAbs(0.0, 1e-15));
}

TEST_CASE("kahan diagonal values", "[generators][kahan]") {
    double theta = 1.2;
    double zeta = 25.0;
    auto K = generators::kahan<double>(4, theta, zeta);

    double s = std::sin(theta);
    // K(i,i) = s^i * zeta
    REQUIRE_THAT(K(0, 0), WithinAbs(zeta, 1e-12));
    REQUIRE_THAT(K(1, 1), WithinAbs(s * zeta, 1e-12));
    REQUIRE_THAT(K(2, 2), WithinAbs(s * s * zeta, 1e-12));
    REQUIRE_THAT(K(3, 3), WithinAbs(s * s * s * zeta, 1e-12));
}

TEST_CASE("kahan superdiagonal values", "[generators][kahan]") {
    double theta = 1.2;
    auto K = generators::kahan<double>(3, theta);

    double s = std::sin(theta);
    double c = std::cos(theta);

    // K(0,1) = -c * s^0 = -c
    REQUIRE_THAT(K(0, 1), WithinAbs(-c, 1e-12));
    // K(0,2) = -c * s^0 = -c
    REQUIRE_THAT(K(0, 2), WithinAbs(-c, 1e-12));
    // K(1,2) = -c * s^1 = -c*s
    REQUIRE_THAT(K(1, 2), WithinAbs(-c * s, 1e-12));
}

TEST_CASE("kahan default parameters", "[generators][kahan]") {
    auto K = generators::kahan<double>(3);
    REQUIRE(K.num_rows() == 3);
    REQUIRE(K.num_cols() == 3);
    // Just verify it's upper triangular with defaults
    REQUIRE_THAT(K(1, 0), WithinAbs(0.0, 1e-15));
    REQUIRE_THAT(K(2, 0), WithinAbs(0.0, 1e-15));
    REQUIRE_THAT(K(2, 1), WithinAbs(0.0, 1e-15));
}
