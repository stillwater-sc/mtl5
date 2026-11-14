// The identities of max and min (#838): -inf and +inf where the type has
// infinities, lowest() and max() where it does not.
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>

#include <mtl/math/identity.hpp>

using namespace mtl::math;

TEST_CASE("max and min identities", "[math][identity][838]") {
    constexpr double inf = std::numeric_limits<double>::infinity();
    STATIC_REQUIRE(identity_t<max_op<double>, double>{}() == -inf);   // was lowest()
    STATIC_REQUIRE(identity_t<min_op<double>, double>{}() == inf);    // was max()
    STATIC_REQUIRE(identity_t<max_op<float>, float>{}() == -std::numeric_limits<float>::infinity());
    // No infinities: the extreme values.
    STATIC_REQUIRE(identity_t<max_op<int>, int>{}() == std::numeric_limits<int>::lowest());
    STATIC_REQUIRE(identity_t<min_op<int>, int>{}() == std::numeric_limits<int>::max());
    STATIC_REQUIRE(identity_t<max_op<std::uint8_t>, std::uint8_t>{}() == 0);

    // An identity is neutral, so a max over values that are all -inf is -inf.
    const double all_neg_inf[] = {-inf, -inf};
    double m = identity_t<max_op<double>, double>{}();
    for (double v : all_neg_inf) m = std::max(m, v);
    CHECK(m == -inf);   // lowest() before #838
}
