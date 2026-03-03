#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/math/identity.hpp>

TEST_CASE("math::zero returns additive identity", "[math][identity]") {
    SECTION("integer zero") {
        REQUIRE(mtl::math::zero<int>() == 0);
    }
    SECTION("float zero") {
        REQUIRE(mtl::math::zero<float>() == 0.0f);
    }
    SECTION("double zero") {
        REQUIRE(mtl::math::zero<double>() == 0.0);
    }
}

TEST_CASE("math::one returns multiplicative identity", "[math][identity]") {
    SECTION("integer one") {
        REQUIRE(mtl::math::one<int>() == 1);
    }
    SECTION("float one") {
        REQUIRE(mtl::math::one<float>() == 1.0f);
    }
    SECTION("double one") {
        REQUIRE(mtl::math::one<double>() == 1.0);
    }
}

TEST_CASE("math::zero and math::one are constexpr", "[math][identity]") {
    constexpr auto z = mtl::math::zero<double>();
    constexpr auto o = mtl::math::one<double>();
    STATIC_REQUIRE(z == 0.0);
    STATIC_REQUIRE(o == 1.0);
}

TEST_CASE("identity_t for max_op is lowest", "[math][identity]") {
    auto v = mtl::math::identity_t<mtl::math::max_op<double>, double>{}();
    REQUIRE(v == std::numeric_limits<double>::lowest());
}

TEST_CASE("identity_t for min_op is max", "[math][identity]") {
    auto v = mtl::math::identity_t<mtl::math::min_op<double>, double>{}();
    REQUIRE(v == std::numeric_limits<double>::max());
}

TEST_CASE("operation functors produce correct results", "[math][operations]") {
    mtl::math::add<int> a;
    mtl::math::mult<int> m;
    REQUIRE(a(3, 4) == 7);
    REQUIRE(m(3, 4) == 12);

    mtl::math::max_op<int> mx;
    mtl::math::min_op<int> mn;
    REQUIRE(mx(3, 7) == 7);
    REQUIRE(mn(3, 7) == 3);
}
