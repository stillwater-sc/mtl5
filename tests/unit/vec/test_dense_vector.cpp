#include <catch2/catch_test_macros.hpp>

// Stub test — will be filled when dense_vector is ported
#include <mtl/vec/dimension.hpp>
#include <mtl/vec/parameter.hpp>

TEST_CASE("Fixed vector dimension", "[vec][dimension]") {
    mtl::vec::fixed::dimension<5> d;
    REQUIRE(d.size() == 5);
    STATIC_REQUIRE(d.is_fixed);
}

TEST_CASE("Non-fixed vector dimension", "[vec][dimension]") {
    mtl::vec::non_fixed::dimension d(10);
    REQUIRE(d.size() == 10);
    STATIC_REQUIRE_FALSE(d.is_fixed);

    d.set_size(20);
    REQUIRE(d.size() == 20);
}

TEST_CASE("Vector parameter bundle compiles", "[vec][parameter]") {
    using default_params = mtl::vec::parameters<>;
    STATIC_REQUIRE(std::is_same_v<default_params::orientation, mtl::tag::col_major>);
    STATIC_REQUIRE(std::is_same_v<default_params::storage, mtl::tag::on_heap>);
}
