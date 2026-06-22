// MTL5 -- element-wise tensor convert/cast (issue #164).
// Non-fused re-quantization of a stored dense vector/matrix between number types.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cstddef>
#include <type_traits>

#include <mtl/vec/dense_vector.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/operation/convert.hpp>

using namespace mtl;
using Catch::Matchers::WithinRel;

TEST_CASE("convert vector out-of-place to a new element type", "[operation][convert]") {
    vec::dense_vector<double> v = {1.0, 2.5, -3.0, 4.25};
    auto f = convert<float>(v);
    static_assert(std::is_same_v<decltype(f)::value_type, float>);
    REQUIRE(f.size() == v.size());
    for (std::size_t i = 0; i < v.size(); ++i)
        REQUIRE(f(static_cast<int>(i)) == static_cast<float>(v(static_cast<int>(i))));
}

TEST_CASE("convert vector in-place into a preallocated destination", "[operation][convert]") {
    vec::dense_vector<float> src = {1.5f, 2.5f, 3.5f};
    vec::dense_vector<double> dst(3, 0.0);
    convert(src, dst);
    REQUIRE_THAT(dst(0), WithinRel(1.5, 1e-12));
    REQUIRE_THAT(dst(2), WithinRel(3.5, 1e-12));
}

TEST_CASE("convert matrix out-of-place and in-place", "[operation][convert]") {
    mat::dense2D<double> A(2, 2);
    A(0,0)=1.0; A(0,1)=2.0; A(1,0)=3.0; A(1,1)=4.0;

    auto Af = convert<float>(A);
    static_assert(std::is_same_v<decltype(Af)::value_type, float>);
    REQUIRE(Af.num_rows() == 2);
    REQUIRE(Af(1,1) == 4.0f);

    mat::dense2D<double> B(2, 2);
    convert(Af, B);                 // float -> double, in-place
    REQUIRE_THAT(B(0,1), WithinRel(2.0, 1e-12));
    REQUIRE_THAT(B(1,0), WithinRel(3.0, 1e-12));
}

TEST_CASE("convert round-trip through a narrower type loses precision as expected",
          "[operation][convert]") {
    // A value not representable in float: round-tripping double->float->double
    // changes it, confirming convert performs a real element cast (not a bit-copy).
    vec::dense_vector<double> v = {1.0 + 1e-12};
    auto back = convert<double>(convert<float>(v));
    REQUIRE(back(0) != v(0));
    REQUIRE_THAT(back(0), WithinRel(1.0, 1e-6));
}
