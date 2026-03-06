#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/generators/lehmer.hpp>
#include <mtl/concepts/matrix.hpp>

using namespace mtl;
using Catch::Matchers::WithinAbs;

TEST_CASE("lehmer satisfies Matrix concept", "[generators][lehmer]") {
    STATIC_REQUIRE(Matrix<generators::lehmer<double>>);
}

TEST_CASE("lehmer element values", "[generators][lehmer]") {
    generators::lehmer<double> L(4);

    REQUIRE(L.num_rows() == 4);
    REQUIRE(L.num_cols() == 4);

    // L(i,j) = (min(i,j)+1)/(max(i,j)+1)
    REQUIRE_THAT(L(0, 0), WithinAbs(1.0, 1e-15));          // 1/1
    REQUIRE_THAT(L(0, 1), WithinAbs(1.0 / 2.0, 1e-15));    // 1/2
    REQUIRE_THAT(L(0, 3), WithinAbs(1.0 / 4.0, 1e-15));    // 1/4
    REQUIRE_THAT(L(1, 2), WithinAbs(2.0 / 3.0, 1e-15));    // 2/3
    REQUIRE_THAT(L(2, 3), WithinAbs(3.0 / 4.0, 1e-15));    // 3/4
    REQUIRE_THAT(L(3, 3), WithinAbs(1.0, 1e-15));           // 4/4
}

TEST_CASE("lehmer is symmetric", "[generators][lehmer]") {
    generators::lehmer<double> L(5);
    for (std::size_t i = 0; i < 5; ++i)
        for (std::size_t j = 0; j < 5; ++j)
            REQUIRE_THAT(L(i, j), WithinAbs(L(j, i), 1e-15));
}

TEST_CASE("lehmer diagonal is all ones", "[generators][lehmer]") {
    generators::lehmer<double> L(6);
    for (std::size_t i = 0; i < 6; ++i)
        REQUIRE_THAT(L(i, i), WithinAbs(1.0, 1e-15));
}

TEST_CASE("lehmer entries are positive", "[generators][lehmer]") {
    generators::lehmer<double> L(5);
    for (std::size_t i = 0; i < 5; ++i)
        for (std::size_t j = 0; j < 5; ++j)
            REQUIRE(L(i, j) > 0.0);
}
