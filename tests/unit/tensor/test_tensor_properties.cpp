// Tests for the rank-2 tensor property predicates (#244, batch 4):
// is_symmetric, is_antisymmetric.
#include <catch2/catch_test_macros.hpp>

#include <limits>

#include <mtl/tensor/tensor.hpp>
#include <mtl/tensor/properties.hpp>

using namespace mtl::tensor;

TEST_CASE("tensor is_symmetric", "[tensor][properties]") {
    tensor<double, 2, 3> S;
    S(0, 0) = 1; S(0, 1) = 2; S(0, 2) = 3;
    S(1, 0) = 2; S(1, 1) = 4; S(1, 2) = 5;
    S(2, 0) = 3; S(2, 1) = 5; S(2, 2) = 6;
    REQUIRE(is_symmetric(S));
    REQUIRE_FALSE(is_antisymmetric(S));

    // Break symmetry in one off-diagonal pair.
    tensor<double, 2, 3> N = S;
    N(0, 2) = 3.5;
    REQUIRE_FALSE(is_symmetric(N));
    REQUIRE(is_symmetric(N, 1.0));   // within a loose tolerance

    // Diagonal tensor is symmetric.
    tensor<double, 2, 2> D;
    D(0, 0) = 7; D(1, 1) = -2;
    REQUIRE(is_symmetric(D));
}

TEST_CASE("tensor is_antisymmetric", "[tensor][properties]") {
    // Skew tensor: t(i,j) = -t(j,i), zero diagonal.
    tensor<double, 2, 3> A;
    A(0, 1) = 2;  A(1, 0) = -2;
    A(0, 2) = -3; A(2, 0) = 3;
    A(1, 2) = 1;  A(2, 1) = -1;
    REQUIRE(is_antisymmetric(A));
    REQUIRE_FALSE(is_symmetric(A));

    // A non-zero diagonal breaks antisymmetry even if off-diagonals are skew.
    tensor<double, 2, 2> B;
    B(0, 0) = 1;              // must be zero for skew
    B(0, 1) = 5; B(1, 0) = -5;
    REQUIRE_FALSE(is_antisymmetric(B));

    // The zero tensor is both symmetric and antisymmetric.
    tensor<double, 2, 3> Z;
    REQUIRE(is_symmetric(Z));
    REQUIRE(is_antisymmetric(Z));
}

TEST_CASE("tensor predicates are NaN-safe", "[tensor][properties]") {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    // NaN off-diagonal makes the deviation unordered; must fail at any tolerance.
    tensor<double, 2, 2> S;
    S(0, 0) = 1; S(0, 1) = nan; S(1, 0) = 2; S(1, 1) = 1;
    REQUIRE_FALSE(is_symmetric(S));
    REQUIRE_FALSE(is_symmetric(S, 1e6));

    tensor<double, 2, 2> A;
    A(0, 1) = nan; A(1, 0) = -1;
    REQUIRE_FALSE(is_antisymmetric(A));
}

TEST_CASE("tensor 1D degenerate case", "[tensor][properties]") {
    // A 1x1 rank-2 tensor: symmetric always; antisymmetric iff the sole
    // (diagonal) component is zero.
    tensor<double, 2, 1> t;
    t(0, 0) = 5;
    REQUIRE(is_symmetric(t));
    REQUIRE_FALSE(is_antisymmetric(t));
    t(0, 0) = 0;
    REQUIRE(is_antisymmetric(t));
}
