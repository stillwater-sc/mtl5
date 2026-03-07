#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/generators/randorth.hpp>
#include <mtl/generators/ones.hpp>
#include <mtl/generators/wilkinson.hpp>
#include <complex>
#include <cmath>

using namespace mtl;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

// -- Vector norms --------------------------------------------------------

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

// -- Matrix norms --------------------------------------------------------

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

// -- Generator-based norm tests ----------------------------------------

TEST_CASE("frobenius_norm of orthogonal matrix equals sqrt(n)", "[operation][norms][generator]") {
    constexpr std::size_t n = 6;
    auto Q = generators::randorth<double>(n);
    // ||Q||_F = sqrt(n) for any n x n orthogonal matrix
    REQUIRE_THAT(frobenius_norm(Q), WithinAbs(std::sqrt(static_cast<double>(n)), 1e-10));
}

TEST_CASE("one_norm of ones matrix equals n", "[operation][norms][generator]") {
    constexpr std::size_t n = 5;
    generators::ones<double> J_gen(n);
    mat::dense2D<double> J(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            J(i, j) = J_gen(i, j);
    // ||J||_1 = max col sum = n
    REQUIRE(one_norm(J) == static_cast<double>(n));
}

TEST_CASE("infinity_norm of Wilkinson matrix", "[operation][norms][generator]") {
    // Wilkinson W+ with n=7 (m=3): diag = [3,2,1,0,1,2,3], sub/super = 1
    // Row sums: row0 = |3|+|1| = 4, row1 = |1|+|2|+|1| = 4,
    //           row2 = |1|+|1|+|1| = 3, row3 = |1|+|0|+|1| = 2,
    //           row4 = |1|+|1|+|1| = 3, row5 = |1|+|2|+|1| = 4, row6 = |1|+|3| = 4
    // ||W||_inf = max row sum = 4 = m + 1 where m = (n-1)/2 = 3
    constexpr std::size_t n = 7;
    auto W = generators::wilkinson<double>(n);
    std::size_t m = (n - 1) / 2;
    // For interior rows: |1| + |diag| + |1|, max is at row 1 or row 5: 1+2+1=4
    // For boundary rows: |diag| + |1| = m+1
    // The maximum row sum is m+1 for the first/last row
    REQUIRE_THAT(infinity_norm(W), WithinAbs(static_cast<double>(m + 1), 1e-10));
}
