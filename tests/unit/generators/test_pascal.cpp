#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/generators/pascal.hpp>

using namespace mtl;
using Catch::Matchers::WithinAbs;

TEST_CASE("pascal dimensions", "[generators][pascal]") {
    auto P = generators::pascal<double>(5);
    REQUIRE(P.num_rows() == 5);
    REQUIRE(P.num_cols() == 5);
}

TEST_CASE("pascal is symmetric", "[generators][pascal]") {
    auto P = generators::pascal<double>(5);
    for (std::size_t i = 0; i < 5; ++i)
        for (std::size_t j = 0; j < 5; ++j)
            REQUIRE_THAT(P(i, j), WithinAbs(P(j, i), 1e-12));
}

TEST_CASE("pascal 5x5 known values", "[generators][pascal]") {
    auto P = generators::pascal<double>(5);
    // P(i,j) = C(i+j, i) = binomial coefficient
    // Row 0: 1 1 1  1  1
    // Row 1: 1 2 3  4  5
    // Row 2: 1 3 6 10 15
    // Row 3: 1 4 10 20 35
    // Row 4: 1 5 15 35 70
    REQUIRE_THAT(P(0, 0), WithinAbs(1.0, 1e-12));
    REQUIRE_THAT(P(0, 4), WithinAbs(1.0, 1e-12));
    REQUIRE_THAT(P(1, 1), WithinAbs(2.0, 1e-12));
    REQUIRE_THAT(P(1, 3), WithinAbs(4.0, 1e-12));
    REQUIRE_THAT(P(2, 2), WithinAbs(6.0, 1e-12));
    REQUIRE_THAT(P(2, 4), WithinAbs(15.0, 1e-12));
    REQUIRE_THAT(P(3, 3), WithinAbs(20.0, 1e-12));
    REQUIRE_THAT(P(4, 4), WithinAbs(70.0, 1e-12));
}

TEST_CASE("pascal first row and column are all ones", "[generators][pascal]") {
    auto P = generators::pascal<double>(6);
    for (std::size_t i = 0; i < 6; ++i) {
        REQUIRE_THAT(P(i, 0), WithinAbs(1.0, 1e-12));
        REQUIRE_THAT(P(0, i), WithinAbs(1.0, 1e-12));
    }
}

TEST_CASE("pascal recurrence relation", "[generators][pascal]") {
    auto P = generators::pascal<double>(6);
    // P(i,j) = P(i-1,j) + P(i,j-1) for i,j >= 1
    for (std::size_t i = 1; i < 6; ++i)
        for (std::size_t j = 1; j < 6; ++j)
            REQUIRE_THAT(P(i, j), WithinAbs(P(i - 1, j) + P(i, j - 1), 1e-12));
}
