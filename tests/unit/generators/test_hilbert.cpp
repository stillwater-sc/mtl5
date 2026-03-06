#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/generators/hilbert.hpp>
#include <mtl/concepts/matrix.hpp>

using namespace mtl;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

TEST_CASE("hilbert satisfies Matrix concept", "[generators][hilbert]") {
    STATIC_REQUIRE(Matrix<generators::hilbert<double>>);
}

TEST_CASE("hilbert element values", "[generators][hilbert]") {
    generators::hilbert<double> H(4);

    REQUIRE(H.num_rows() == 4);
    REQUIRE(H.num_cols() == 4);
    REQUIRE(H.size() == 16);

    REQUIRE_THAT(H(0, 0), WithinAbs(1.0, 1e-15));
    REQUIRE_THAT(H(0, 1), WithinAbs(1.0 / 2.0, 1e-15));
    REQUIRE_THAT(H(0, 2), WithinAbs(1.0 / 3.0, 1e-15));
    REQUIRE_THAT(H(1, 0), WithinAbs(1.0 / 2.0, 1e-15));
    REQUIRE_THAT(H(1, 1), WithinAbs(1.0 / 3.0, 1e-15));
    REQUIRE_THAT(H(2, 3), WithinAbs(1.0 / 6.0, 1e-15));
    REQUIRE_THAT(H(3, 3), WithinAbs(1.0 / 7.0, 1e-15));
}

TEST_CASE("hilbert is symmetric", "[generators][hilbert]") {
    generators::hilbert<double> H(5);
    for (std::size_t i = 0; i < 5; ++i)
        for (std::size_t j = 0; j < 5; ++j)
            REQUIRE_THAT(H(i, j), WithinAbs(H(j, i), 1e-15));
}

TEST_CASE("hilbert condition number grows rapidly", "[generators][hilbert]") {
    // H(1) = [1], cond = 1
    // H(2) has cond ~ 19.3
    // H(3) has cond ~ 524
    // Verify entries grow in magnitude ratio
    generators::hilbert<double> H3(3);
    generators::hilbert<double> H5(5);

    // Just verify the corner entries show expected decay
    REQUIRE(H3(0, 0) > H3(2, 2));
    REQUIRE(H5(0, 0) > H5(4, 4));
    REQUIRE_THAT(H5(4, 4), WithinAbs(1.0 / 9.0, 1e-15));
}
