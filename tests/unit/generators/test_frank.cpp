#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/generators/frank.hpp>

using namespace mtl;
using Catch::Matchers::WithinAbs;

TEST_CASE("frank dimensions", "[generators][frank]") {
    auto F = generators::frank<double>(5);
    REQUIRE(F.num_rows() == 5);
    REQUIRE(F.num_cols() == 5);
}

TEST_CASE("frank is upper Hessenberg", "[generators][frank]") {
    auto F = generators::frank<double>(5);

    // Zeros below sub-diagonal: F(i,j) = 0 for j < i-1
    for (std::size_t i = 2; i < 5; ++i)
        for (std::size_t j = 0; j + 1 < i; ++j)
            REQUIRE_THAT(F(i, j), WithinAbs(0.0, 1e-15));
}

TEST_CASE("frank 4x4 known values", "[generators][frank]") {
    auto F = generators::frank<double>(4);

    // F(i,j) = n+1-max(i+1,j+1) for j >= i-1
    // n=4:
    // [4 3 2 1]
    // [3 3 2 1]
    // [0 2 2 1]
    // [0 0 1 1]
    REQUIRE_THAT(F(0, 0), WithinAbs(4.0, 1e-15));
    REQUIRE_THAT(F(0, 1), WithinAbs(3.0, 1e-15));
    REQUIRE_THAT(F(0, 2), WithinAbs(2.0, 1e-15));
    REQUIRE_THAT(F(0, 3), WithinAbs(1.0, 1e-15));
    REQUIRE_THAT(F(1, 0), WithinAbs(3.0, 1e-15));
    REQUIRE_THAT(F(1, 1), WithinAbs(3.0, 1e-15));
    REQUIRE_THAT(F(1, 2), WithinAbs(2.0, 1e-15));
    REQUIRE_THAT(F(1, 3), WithinAbs(1.0, 1e-15));
    REQUIRE_THAT(F(2, 0), WithinAbs(0.0, 1e-15));
    REQUIRE_THAT(F(2, 1), WithinAbs(2.0, 1e-15));
    REQUIRE_THAT(F(2, 2), WithinAbs(2.0, 1e-15));
    REQUIRE_THAT(F(2, 3), WithinAbs(1.0, 1e-15));
    REQUIRE_THAT(F(3, 0), WithinAbs(0.0, 1e-15));
    REQUIRE_THAT(F(3, 1), WithinAbs(0.0, 1e-15));
    REQUIRE_THAT(F(3, 2), WithinAbs(1.0, 1e-15));
    REQUIRE_THAT(F(3, 3), WithinAbs(1.0, 1e-15));
}

TEST_CASE("frank sub-diagonal values", "[generators][frank]") {
    auto F = generators::frank<double>(5);
    // Sub-diagonal F(i, i-1) = n+1-(i+1) = n-i
    for (std::size_t i = 1; i < 5; ++i)
        REQUIRE_THAT(F(i, i - 1), WithinAbs(double(5 - i), 1e-15));
}
