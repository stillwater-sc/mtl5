#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/generators/clement.hpp>
#include <cmath>

using namespace mtl;
using Catch::Matchers::WithinAbs;

TEST_CASE("clement dimensions", "[generators][clement]") {
    auto C = generators::clement<double>(5);
    REQUIRE(C.num_rows() == 5);
    REQUIRE(C.num_cols() == 5);
}

TEST_CASE("clement is tridiagonal", "[generators][clement]") {
    auto C = generators::clement<double>(6);

    // All entries outside the tridiagonal band must be zero
    for (std::size_t i = 0; i < 6; ++i)
        for (std::size_t j = 0; j < 6; ++j)
            if (i != j && i + 1 != j && j + 1 != i)
                REQUIRE_THAT(C(i, j), WithinAbs(0.0, 1e-15));

    // Diagonal must be zero
    for (std::size_t i = 0; i < 6; ++i)
        REQUIRE_THAT(C(i, i), WithinAbs(0.0, 1e-15));
}

TEST_CASE("clement is symmetric", "[generators][clement]") {
    auto C = generators::clement<double>(6);
    for (std::size_t i = 0; i < 6; ++i)
        for (std::size_t j = 0; j < 6; ++j)
            REQUIRE_THAT(C(i, j), WithinAbs(C(j, i), 1e-15));
}

TEST_CASE("clement sub/superdiagonal values", "[generators][clement]") {
    std::size_t n = 5;
    auto C = generators::clement<double>(n);

    // C(i,i+1) = sqrt((i+1)*(n-1-i))
    for (std::size_t i = 0; i + 1 < n; ++i) {
        double expected = std::sqrt(double((i + 1) * (n - 1 - i)));
        REQUIRE_THAT(C(i, i + 1), WithinAbs(expected, 1e-12));
        REQUIRE_THAT(C(i + 1, i), WithinAbs(expected, 1e-12));
    }
}

TEST_CASE("clement 4x4 known values", "[generators][clement]") {
    auto C = generators::clement<double>(4);
    // C(0,1) = sqrt(1*3) = sqrt(3)
    // C(1,2) = sqrt(2*2) = 2
    // C(2,3) = sqrt(3*1) = sqrt(3)
    REQUIRE_THAT(C(0, 1), WithinAbs(std::sqrt(3.0), 1e-12));
    REQUIRE_THAT(C(1, 2), WithinAbs(2.0, 1e-12));
    REQUIRE_THAT(C(2, 3), WithinAbs(std::sqrt(3.0), 1e-12));
}
