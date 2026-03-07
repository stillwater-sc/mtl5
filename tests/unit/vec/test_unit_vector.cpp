#include <catch2/catch_test_macros.hpp>
#include <mtl/vec/unit_vector.hpp>

using namespace mtl;

TEST_CASE("unit_vector has correct size", "[vec][unit_vector]") {
    auto v = unit_vector(5, 2);
    REQUIRE(v.size() == 5);
}

TEST_CASE("unit_vector element at k is 1, others are 0", "[vec][unit_vector]") {
    auto v = unit_vector(4, 1);
    REQUIRE(v(0) == 0.0);
    REQUIRE(v(1) == 1.0);
    REQUIRE(v(2) == 0.0);
    REQUIRE(v(3) == 0.0);
}

TEST_CASE("unit_vector k=0 (first position)", "[vec][unit_vector]") {
    auto v = unit_vector(3, 0);
    REQUIRE(v(0) == 1.0);
    REQUIRE(v(1) == 0.0);
    REQUIRE(v(2) == 0.0);
}

TEST_CASE("unit_vector k=n-1 (last position)", "[vec][unit_vector]") {
    auto v = unit_vector(3, 2);
    REQUIRE(v(0) == 0.0);
    REQUIRE(v(1) == 0.0);
    REQUIRE(v(2) == 1.0);
}

TEST_CASE("unit_vector with float type", "[vec][unit_vector]") {
    auto v = unit_vector<float>(4, 2);
    REQUIRE(v(0) == 0.0f);
    REQUIRE(v(1) == 0.0f);
    REQUIRE(v(2) == 1.0f);
    REQUIRE(v(3) == 0.0f);
}

TEST_CASE("unit_vector with int type", "[vec][unit_vector]") {
    auto v = unit_vector<int>(3, 1);
    REQUIRE(v(0) == 0);
    REQUIRE(v(1) == 1);
    REQUIRE(v(2) == 0);
}
