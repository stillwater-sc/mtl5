#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/generators/wilkinson.hpp>

#include <stdexcept>

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

TEST_CASE("wilkinson of even order has half-integer diagonal, as MATLAB", "[generators][wilkinson]") {
    // MATLAB's wilkinson(n) is abs(-(n-1)/2 : (n-1)/2) on the diagonal for any
    // n. Under NDEBUG an even n used to give 1, 0, 1, 2 (#844).
    auto W = generators::wilkinson<double>(6);
    const double d[] = {2.5, 1.5, 0.5, 0.5, 1.5, 2.5};
    for (std::size_t i = 0; i < 6; ++i) REQUIRE(W(i, i) == d[i]);
    for (std::size_t i = 0; i + 1 < 6; ++i) {
        REQUIRE(W(i, i + 1) == 1.0);
        REQUIRE(W(i + 1, i) == 1.0);
    }
}

TEST_CASE("wilkinson rejects order 0", "[generators][wilkinson]") {
    // (n - 1) / 2 wrapped for n = 0 under NDEBUG (#844)
    REQUIRE_THROWS_AS(generators::wilkinson<double>(0), std::invalid_argument);
}
