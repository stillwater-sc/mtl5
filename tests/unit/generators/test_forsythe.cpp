#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/generators/forsythe.hpp>

using namespace mtl;
using Catch::Matchers::WithinAbs;

TEST_CASE("forsythe dimensions", "[generators][forsythe]") {
    auto F = generators::forsythe<double>(5);
    REQUIRE(F.num_rows() == 5);
    REQUIRE(F.num_cols() == 5);
}

TEST_CASE("forsythe Jordan block structure", "[generators][forsythe]") {
    double alpha = 1e-10;
    double lambda = 2.0;
    auto F = generators::forsythe<double>(4, alpha, lambda);

    // Diagonal = lambda
    for (std::size_t i = 0; i < 4; ++i)
        REQUIRE_THAT(F(i, i), WithinAbs(lambda, 1e-15));

    // Superdiagonal = 1
    for (std::size_t i = 0; i < 3; ++i)
        REQUIRE_THAT(F(i, i + 1), WithinAbs(1.0, 1e-15));

    // Corner perturbation
    REQUIRE_THAT(F(3, 0), WithinAbs(alpha, 1e-25));

    // Everything else is zero
    REQUIRE_THAT(F(0, 2), WithinAbs(0.0, 1e-15));
    REQUIRE_THAT(F(0, 3), WithinAbs(0.0, 1e-15));
    REQUIRE_THAT(F(1, 0), WithinAbs(0.0, 1e-15));
    REQUIRE_THAT(F(1, 3), WithinAbs(0.0, 1e-15));
    REQUIRE_THAT(F(2, 0), WithinAbs(0.0, 1e-15));
    REQUIRE_THAT(F(2, 1), WithinAbs(0.0, 1e-15));
    REQUIRE_THAT(F(3, 1), WithinAbs(0.0, 1e-15));
    REQUIRE_THAT(F(3, 2), WithinAbs(0.0, 1e-15));
}

TEST_CASE("forsythe with zero alpha is Jordan block", "[generators][forsythe]") {
    auto F = generators::forsythe<double>(3, 0.0, 5.0);

    // Pure Jordan block with eigenvalue 5
    REQUIRE_THAT(F(0, 0), WithinAbs(5.0, 1e-15));
    REQUIRE_THAT(F(1, 1), WithinAbs(5.0, 1e-15));
    REQUIRE_THAT(F(2, 2), WithinAbs(5.0, 1e-15));
    REQUIRE_THAT(F(0, 1), WithinAbs(1.0, 1e-15));
    REQUIRE_THAT(F(1, 2), WithinAbs(1.0, 1e-15));
    REQUIRE_THAT(F(2, 0), WithinAbs(0.0, 1e-15)); // no perturbation
}

TEST_CASE("forsythe default parameters", "[generators][forsythe]") {
    auto F = generators::forsythe<double>(3);
    // Default: alpha=1e-10, lambda=0
    REQUIRE_THAT(F(0, 0), WithinAbs(0.0, 1e-15));
    REQUIRE_THAT(F(0, 1), WithinAbs(1.0, 1e-15));
    REQUIRE_THAT(F(2, 0), WithinAbs(1e-10, 1e-25));
}
