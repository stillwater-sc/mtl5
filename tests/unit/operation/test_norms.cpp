#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/operation/norms.hpp>
#include <complex>
#include <cmath>

using namespace mtl;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

// ── Vector norms ────────────────────────────────────────────────────────

TEST_CASE("one_norm of vector", "[operation][norms]") {
    dense_vector<double> v = {1.0, -2.0, 3.0};
    REQUIRE(one_norm(v) == 6.0);
}

TEST_CASE("two_norm of vector", "[operation][norms]") {
    dense_vector<double> v = {3.0, 4.0};
    REQUIRE_THAT(two_norm(v), WithinAbs(5.0, 1e-10));
}

TEST_CASE("two_norm of unit vector", "[operation][norms]") {
    dense_vector<double> v = {1.0, 0.0, 0.0};
    REQUIRE_THAT(two_norm(v), WithinAbs(1.0, 1e-10));
}

TEST_CASE("infinity_norm of vector", "[operation][norms]") {
    dense_vector<double> v = {1.0, -5.0, 3.0};
    REQUIRE(infinity_norm(v) == 5.0);
}

TEST_CASE("norms of zero vector", "[operation][norms]") {
    dense_vector<double> v(3, 0.0);
    REQUIRE(one_norm(v) == 0.0);
    REQUIRE(two_norm(v) == 0.0);
    REQUIRE(infinity_norm(v) == 0.0);
}

TEST_CASE("norms of complex vector", "[operation][norms]") {
    using cd = std::complex<double>;
    dense_vector<cd> v = {cd(3.0, 4.0)};  // |3+4i| = 5
    REQUIRE_THAT(one_norm(v), WithinAbs(5.0, 1e-10));
    REQUIRE_THAT(two_norm(v), WithinAbs(5.0, 1e-10));
    REQUIRE_THAT(infinity_norm(v), WithinAbs(5.0, 1e-10));
}

// ── Matrix norms ────────────────────────────────────────────────────────

TEST_CASE("frobenius_norm of matrix", "[operation][norms]") {
    dense2D<double> m = {{1.0, 2.0}, {3.0, 4.0}};
    // sqrt(1+4+9+16) = sqrt(30)
    REQUIRE_THAT(frobenius_norm(m), WithinAbs(std::sqrt(30.0), 1e-10));
}

TEST_CASE("one_norm of matrix (max col sum)", "[operation][norms]") {
    dense2D<double> m = {{1.0, -2.0}, {3.0, 4.0}};
    // col0: |1|+|3| = 4, col1: |-2|+|4| = 6
    REQUIRE(one_norm(m) == 6.0);
}

TEST_CASE("infinity_norm of matrix (max row sum)", "[operation][norms]") {
    dense2D<double> m = {{1.0, -2.0}, {3.0, 4.0}};
    // row0: |1|+|-2| = 3, row1: |3|+|4| = 7
    REQUIRE(infinity_norm(m) == 7.0);
}

TEST_CASE("frobenius_norm of identity matrix", "[operation][norms]") {
    dense2D<double> m = {{1.0, 0.0}, {0.0, 1.0}};
    REQUIRE_THAT(frobenius_norm(m), WithinAbs(std::sqrt(2.0), 1e-10));
}
