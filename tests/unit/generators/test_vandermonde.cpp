#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/generators/vandermonde.hpp>
#include <vector>

using namespace mtl;
using Catch::Matchers::WithinAbs;

TEST_CASE("vandermonde dimensions", "[generators][vandermonde]") {
    std::vector<double> nodes = {1.0, 2.0, 3.0};
    auto V = generators::vandermonde(nodes);
    REQUIRE(V.num_rows() == 3);
    REQUIRE(V.num_cols() == 3);
}

TEST_CASE("vandermonde V(i,j) = x_i^j", "[generators][vandermonde]") {
    std::vector<double> nodes = {1.0, 2.0, 3.0, 4.0};
    auto V = generators::vandermonde(nodes);

    for (std::size_t i = 0; i < 4; ++i) {
        double xpow = 1.0;
        for (std::size_t j = 0; j < 4; ++j) {
            REQUIRE_THAT(V(i, j), WithinAbs(xpow, 1e-10));
            xpow *= nodes[i];
        }
    }
}

TEST_CASE("vandermonde first column is all ones", "[generators][vandermonde]") {
    std::vector<double> nodes = {0.5, 1.5, 2.5, 3.5};
    auto V = generators::vandermonde(nodes);
    for (std::size_t i = 0; i < 4; ++i)
        REQUIRE_THAT(V(i, 0), WithinAbs(1.0, 1e-15));
}

TEST_CASE("vandermonde second column equals nodes", "[generators][vandermonde]") {
    std::vector<double> nodes = {0.5, 1.5, 2.5, 3.5};
    auto V = generators::vandermonde(nodes);
    for (std::size_t i = 0; i < 4; ++i)
        REQUIRE_THAT(V(i, 1), WithinAbs(nodes[i], 1e-15));
}

TEST_CASE("vandermonde 2x2 known values", "[generators][vandermonde]") {
    std::vector<double> nodes = {2.0, 5.0};
    auto V = generators::vandermonde(nodes);
    // V = [1  2]
    //     [1  5]
    REQUIRE_THAT(V(0, 0), WithinAbs(1.0, 1e-15));
    REQUIRE_THAT(V(0, 1), WithinAbs(2.0, 1e-15));
    REQUIRE_THAT(V(1, 0), WithinAbs(1.0, 1e-15));
    REQUIRE_THAT(V(1, 1), WithinAbs(5.0, 1e-15));
}
