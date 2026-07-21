// Tests for the vector property predicates (#244, batch 1):
// is_zero, is_finite, has_nan, has_inf, is_normalized/is_unit, is_orthogonal_to.
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <limits>

#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/vector_properties.hpp>

using namespace mtl;
using vec_t = vec::dense_vector<double>;

TEST_CASE("is_zero", "[operation][properties][vector]") {
    REQUIRE(is_zero(vec_t{0.0, 0.0, 0.0}));
    REQUIRE_FALSE(is_zero(vec_t{0.0, 1e-12, 0.0}));
    // Within tol.
    REQUIRE(is_zero(vec_t{0.0, 1e-12, 0.0}, 1e-9));
    // Empty vector is (vacuously) zero.
    REQUIRE(is_zero(vec_t(0)));
}

TEST_CASE("is_finite / has_nan / has_inf", "[operation][properties][vector]") {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();

    REQUIRE(is_finite(vec_t{1.0, -2.0, 3.5}));
    REQUIRE_FALSE(has_nan(vec_t{1.0, -2.0, 3.5}));
    REQUIRE_FALSE(has_inf(vec_t{1.0, -2.0, 3.5}));

    REQUIRE_FALSE(is_finite(vec_t{1.0, nan, 3.0}));
    REQUIRE(has_nan(vec_t{1.0, nan, 3.0}));
    REQUIRE_FALSE(has_inf(vec_t{1.0, nan, 3.0}));

    REQUIRE_FALSE(is_finite(vec_t{1.0, inf, 3.0}));
    REQUIRE(has_inf(vec_t{1.0, inf, 3.0}));
    REQUIRE_FALSE(has_nan(vec_t{1.0, inf, 3.0}));
}

TEST_CASE("is_normalized / is_unit", "[operation][properties][vector]") {
    // 3-4-5 normalized.
    REQUIRE(is_normalized(vec_t{0.6, 0.8}));
    REQUIRE(is_unit(vec_t{1.0, 0.0, 0.0}));
    REQUIRE_FALSE(is_normalized(vec_t{3.0, 4.0}));  // norm 5
    // Slightly off unit norm: rejected at default tol, accepted with a loose tol.
    vec_t nearly{1.0 + 1e-3, 0.0};
    REQUIRE_FALSE(is_normalized(nearly));
    REQUIRE(is_normalized(nearly, 1e-2));
}

TEST_CASE("is_orthogonal_to", "[operation][properties][vector]") {
    REQUIRE(is_orthogonal_to(vec_t{1.0, 0.0}, vec_t{0.0, 1.0}));
    REQUIRE(is_orthogonal_to(vec_t{1.0, 1.0}, vec_t{1.0, -1.0}));
    REQUIRE_FALSE(is_orthogonal_to(vec_t{1.0, 0.0}, vec_t{1.0, 1.0}));
    // Scale invariance of the relative test.
    REQUIRE(is_orthogonal_to(vec_t{1e6, 0.0}, vec_t{0.0, 1e-6}));
    // A zero vector is orthogonal to anything (threshold is 0, dot is 0).
    REQUIRE(is_orthogonal_to(vec_t{0.0, 0.0}, vec_t{3.0, 4.0}));
}
