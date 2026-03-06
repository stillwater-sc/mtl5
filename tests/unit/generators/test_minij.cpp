#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/generators/minij.hpp>
#include <mtl/concepts/matrix.hpp>

using namespace mtl;
using Catch::Matchers::WithinAbs;

TEST_CASE("minij satisfies Matrix concept", "[generators][minij]") {
    STATIC_REQUIRE(Matrix<generators::minij<double>>);
}

TEST_CASE("minij element values", "[generators][minij]") {
    generators::minij<double> M(4);

    REQUIRE(M.num_rows() == 4);
    REQUIRE(M.num_cols() == 4);

    // M(i,j) = min(i+1, j+1)
    REQUIRE_THAT(M(0, 0), WithinAbs(1.0, 1e-15));
    REQUIRE_THAT(M(0, 3), WithinAbs(1.0, 1e-15));
    REQUIRE_THAT(M(1, 1), WithinAbs(2.0, 1e-15));
    REQUIRE_THAT(M(2, 1), WithinAbs(2.0, 1e-15));
    REQUIRE_THAT(M(3, 3), WithinAbs(4.0, 1e-15));
    REQUIRE_THAT(M(2, 3), WithinAbs(3.0, 1e-15));
}

TEST_CASE("minij is symmetric", "[generators][minij]") {
    generators::minij<double> M(5);
    for (std::size_t i = 0; i < 5; ++i)
        for (std::size_t j = 0; j < 5; ++j)
            REQUIRE_THAT(M(i, j), WithinAbs(M(j, i), 1e-15));
}

TEST_CASE("minij 3x3 known matrix", "[generators][minij]") {
    generators::minij<double> M(3);
    // Expected:
    // [1 1 1]
    // [1 2 2]
    // [1 2 3]
    REQUIRE_THAT(M(0, 0), WithinAbs(1.0, 1e-15));
    REQUIRE_THAT(M(0, 1), WithinAbs(1.0, 1e-15));
    REQUIRE_THAT(M(0, 2), WithinAbs(1.0, 1e-15));
    REQUIRE_THAT(M(1, 0), WithinAbs(1.0, 1e-15));
    REQUIRE_THAT(M(1, 1), WithinAbs(2.0, 1e-15));
    REQUIRE_THAT(M(1, 2), WithinAbs(2.0, 1e-15));
    REQUIRE_THAT(M(2, 0), WithinAbs(1.0, 1e-15));
    REQUIRE_THAT(M(2, 1), WithinAbs(2.0, 1e-15));
    REQUIRE_THAT(M(2, 2), WithinAbs(3.0, 1e-15));
}
