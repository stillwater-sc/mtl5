#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/generators/ones.hpp>
#include <mtl/concepts/matrix.hpp>

using namespace mtl;
using Catch::Matchers::WithinAbs;

TEST_CASE("ones satisfies Matrix concept", "[generators][ones]") {
    STATIC_REQUIRE(Matrix<generators::ones<double>>);
}

TEST_CASE("ones square matrix", "[generators][ones]") {
    generators::ones<double> O(4);
    REQUIRE(O.num_rows() == 4);
    REQUIRE(O.num_cols() == 4);
    REQUIRE(O.size() == 16);

    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            REQUIRE_THAT(O(i, j), WithinAbs(1.0, 1e-15));
}

TEST_CASE("ones rectangular matrix", "[generators][ones]") {
    generators::ones<double> O(3, 5);
    REQUIRE(O.num_rows() == 3);
    REQUIRE(O.num_cols() == 5);
    REQUIRE(O.size() == 15);

    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 5; ++j)
            REQUIRE_THAT(O(i, j), WithinAbs(1.0, 1e-15));
}

TEST_CASE("ones with integer type", "[generators][ones]") {
    generators::ones<int> O(3);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            REQUIRE(O(i, j) == 1);
}
