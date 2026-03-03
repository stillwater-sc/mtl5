#include <catch2/catch_test_macros.hpp>

// Stub test — will be filled when dense2D is ported
#include <mtl/mat/dimension.hpp>
#include <mtl/mat/parameter.hpp>

TEST_CASE("Fixed matrix dimensions", "[mat][dimension]") {
    mtl::mat::fixed::dimensions<3, 4> d;
    REQUIRE(d.num_rows() == 3);
    REQUIRE(d.num_cols() == 4);
    REQUIRE(d.size() == 12);
    STATIC_REQUIRE(d.is_fixed);
}

TEST_CASE("Non-fixed matrix dimensions", "[mat][dimension]") {
    mtl::mat::non_fixed::dimensions d(5, 6);
    REQUIRE(d.num_rows() == 5);
    REQUIRE(d.num_cols() == 6);
    REQUIRE(d.size() == 30);
    STATIC_REQUIRE_FALSE(d.is_fixed);

    d.set_dimensions(10, 20);
    REQUIRE(d.num_rows() == 10);
    REQUIRE(d.num_cols() == 20);
}

TEST_CASE("Matrix parameter bundle compiles", "[mat][parameter]") {
    using default_params = mtl::mat::parameters<>;
    STATIC_REQUIRE(std::is_same_v<default_params::orientation, mtl::tag::row_major>);
    STATIC_REQUIRE(std::is_same_v<default_params::storage, mtl::tag::on_heap>);
}
