#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/generators/wilkinson.hpp>

using namespace mtl;
using Catch::Matchers::WithinAbs;

TEST_CASE("wilkinson dimensions", "[generators][wilkinson]") {
    auto W = generators::wilkinson<double>(7);
    REQUIRE(W.num_rows() == 7);
    REQUIRE(W.num_cols() == 7);
}

TEST_CASE("wilkinson is symmetric", "[generators][wilkinson]") {
    auto W = generators::wilkinson<double>(7);
    for (std::size_t i = 0; i < 7; ++i)
        for (std::size_t j = 0; j < 7; ++j)
            REQUIRE_THAT(W(i, j), WithinAbs(W(j, i), 1e-15));
}

TEST_CASE("wilkinson is tridiagonal", "[generators][wilkinson]") {
    auto W = generators::wilkinson<double>(7);

    // Off-tridiagonal entries must be zero
    for (std::size_t i = 0; i < 7; ++i)
        for (std::size_t j = 0; j < 7; ++j)
            if (i != j && i + 1 != j && j + 1 != i)
                REQUIRE_THAT(W(i, j), WithinAbs(0.0, 1e-15));
}

TEST_CASE("wilkinson diagonal values", "[generators][wilkinson]") {
    // n=7, m=3: diagonal = [3, 2, 1, 0, 1, 2, 3]
    auto W = generators::wilkinson<double>(7);
    REQUIRE_THAT(W(0, 0), WithinAbs(3.0, 1e-15));
    REQUIRE_THAT(W(1, 1), WithinAbs(2.0, 1e-15));
    REQUIRE_THAT(W(2, 2), WithinAbs(1.0, 1e-15));
    REQUIRE_THAT(W(3, 3), WithinAbs(0.0, 1e-15));
    REQUIRE_THAT(W(4, 4), WithinAbs(1.0, 1e-15));
    REQUIRE_THAT(W(5, 5), WithinAbs(2.0, 1e-15));
    REQUIRE_THAT(W(6, 6), WithinAbs(3.0, 1e-15));
}

TEST_CASE("wilkinson sub/superdiagonal all ones", "[generators][wilkinson]") {
    auto W = generators::wilkinson<double>(7);
    for (std::size_t i = 0; i + 1 < 7; ++i) {
        REQUIRE_THAT(W(i, i + 1), WithinAbs(1.0, 1e-15));
        REQUIRE_THAT(W(i + 1, i), WithinAbs(1.0, 1e-15));
    }
}

TEST_CASE("wilkinson 5x5 known values", "[generators][wilkinson]") {
    // n=5, m=2: diagonal = [2, 1, 0, 1, 2]
    auto W = generators::wilkinson<double>(5);

    // Full 5x5:
    // [2 1 0 0 0]
    // [1 1 1 0 0]
    // [0 1 0 1 0]
    // [0 0 1 1 1]
    // [0 0 0 1 2]
    REQUIRE_THAT(W(0, 0), WithinAbs(2.0, 1e-15));
    REQUIRE_THAT(W(0, 1), WithinAbs(1.0, 1e-15));
    REQUIRE_THAT(W(0, 2), WithinAbs(0.0, 1e-15));
    REQUIRE_THAT(W(1, 1), WithinAbs(1.0, 1e-15));
    REQUIRE_THAT(W(2, 2), WithinAbs(0.0, 1e-15));
    REQUIRE_THAT(W(4, 4), WithinAbs(2.0, 1e-15));
}
