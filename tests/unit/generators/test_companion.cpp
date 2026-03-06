#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/generators/companion.hpp>
#include <vector>

using namespace mtl;
using Catch::Matchers::WithinAbs;

TEST_CASE("companion dimensions", "[generators][companion]") {
    std::vector<double> coeffs = {1.0, -2.0, 3.0};
    auto C = generators::companion(coeffs);
    REQUIRE(C.num_rows() == 3);
    REQUIRE(C.num_cols() == 3);
}

TEST_CASE("companion structure", "[generators][companion]") {
    std::vector<double> coeffs = {6.0, -11.0, 6.0};
    auto C = generators::companion(coeffs);

    // Sub-diagonal should be 1s
    for (std::size_t i = 1; i < 3; ++i)
        REQUIRE_THAT(C(i, i - 1), WithinAbs(1.0, 1e-15));

    // Last column should be -coeffs
    REQUIRE_THAT(C(0, 2), WithinAbs(-6.0, 1e-15));
    REQUIRE_THAT(C(1, 2), WithinAbs(11.0, 1e-15));
    REQUIRE_THAT(C(2, 2), WithinAbs(-6.0, 1e-15));

    // Rest should be zero
    REQUIRE_THAT(C(0, 0), WithinAbs(0.0, 1e-15));
    REQUIRE_THAT(C(0, 1), WithinAbs(0.0, 1e-15));
    REQUIRE_THAT(C(1, 0), WithinAbs(1.0, 1e-15));
    REQUIRE_THAT(C(2, 0), WithinAbs(0.0, 1e-15));
    REQUIRE_THAT(C(2, 1), WithinAbs(1.0, 1e-15));
}

TEST_CASE("companion 2x2 for quadratic", "[generators][companion]") {
    // p(x) = x^2 - 5x + 6 = (x-2)(x-3), roots: 2, 3
    // coeffs = [6, -5] (constant term, x coefficient)
    std::vector<double> coeffs = {6.0, -5.0};
    auto C = generators::companion(coeffs);

    REQUIRE_THAT(C(0, 1), WithinAbs(-6.0, 1e-15));
    REQUIRE_THAT(C(1, 0), WithinAbs(1.0, 1e-15));
    REQUIRE_THAT(C(1, 1), WithinAbs(5.0, 1e-15));
    REQUIRE_THAT(C(0, 0), WithinAbs(0.0, 1e-15));
}

TEST_CASE("companion 1x1", "[generators][companion]") {
    std::vector<double> coeffs = {3.0};
    auto C = generators::companion(coeffs);
    REQUIRE(C.num_rows() == 1);
    REQUIRE(C.num_cols() == 1);
    REQUIRE_THAT(C(0, 0), WithinAbs(-3.0, 1e-15));
}
