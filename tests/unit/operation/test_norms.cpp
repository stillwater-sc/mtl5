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
#include <limits>

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

// -- Extreme magnitudes and NaN (#690) -----------------------------------
//
// The default two_norm and frobenius_norm used to sum unscaled squares, so a
// norm whose squares overflow came out inf and one whose squares underflow came
// out 0. They now rescan with a power-of-two scale when that happens. The
// max-based norms used to skip a NaN element; they now return NaN.

namespace {
template <typename T>
constexpr T nan_v = std::numeric_limits<T>::quiet_NaN();
}

TEST_CASE("two_norm does not overflow near the float and double limits", "[operation][norms][690]") {
    dense_vector<float> f = {1e20f, 1e20f};                 // squares overflow float
    REQUIRE_THAT(two_norm(f), WithinRel(1e20f * std::sqrt(2.0f), 1e-6f));
    dense_vector<float> fmax = {std::numeric_limits<float>::max() / 2, std::numeric_limits<float>::max() / 2};
    REQUIRE_THAT(two_norm(fmax), WithinRel(std::numeric_limits<float>::max() / std::sqrt(2.0f), 1e-6f));
    dense_vector<double> d = {3e200, 4e200};
    REQUIRE_THAT(two_norm(d), WithinRel(5e200, 1e-15));
    // A genuinely overflowing norm is still inf.
    dense_vector<double> big = {std::numeric_limits<double>::max(), std::numeric_limits<double>::max()};
    REQUIRE(std::isinf(two_norm(big)));
}

TEST_CASE("two_norm does not underflow near the float and double limits", "[operation][norms][690]") {
    dense_vector<double> d = {1e-200, 1e-200};             // squares underflow to 0
    REQUIRE_THAT(two_norm(d), WithinRel(1e-200 * std::sqrt(2.0), 1e-15));
    dense_vector<double> d34 = {3e-170, 4e-170};           // squares subnormal
    REQUIRE_THAT(two_norm(d34), WithinRel(5e-170, 1e-15));
    dense_vector<float> f = {3e-30f, 4e-30f};
    REQUIRE_THAT(two_norm(f), WithinRel(5e-30f, 1e-6f));
    const double tiny = std::numeric_limits<double>::denorm_min();
    dense_vector<double> sub = {tiny, 0.0};
    REQUIRE(two_norm(sub) == tiny);
    dense_vector<double> zero = {0.0, 0.0, 0.0};
    REQUIRE(two_norm(zero) == 0.0);
}

TEST_CASE("two_norm rescan keeps small elements beside large ones", "[operation][norms][690]") {
    // 1e-170 is lost against 1 either way; against 1e-160 it is not.
    dense_vector<double> v = {1e-160, 1e-160, 1e-160, 1e-160};
    REQUIRE_THAT(two_norm(v), WithinRel(2e-160, 1e-15));
    dense_vector<double> inf_v = {1.0, std::numeric_limits<double>::infinity()};
    REQUIRE(std::isinf(two_norm(inf_v)));
}

TEST_CASE("two_norm of an ordinary vector is unchanged by the rescan", "[operation][norms][690]") {
    // Ordinary inputs take the plain sum, with no rescan.
    dense_vector<double> v = {2.0, 3.0, 6.0};
    REQUIRE(two_norm(v) == 7.0);
    dense_vector<float> f = {1.0f, 2.0f, 2.0f};
    REQUIRE(two_norm(f) == 3.0f);
    dense2D<double> m = {{2.0, 3.0}, {6.0, 0.0}};
    REQUIRE(frobenius_norm(m) == 7.0);
}

TEST_CASE("two_norm generic path rescales too", "[operation][norms][690]") {
    // complex<double> is not a BLAS/SIMD scalar, so this takes the generic loop.
    dense_vector<std::complex<double>> big = {{3e200, 0.0}, {0.0, 4e200}};
    REQUIRE_THAT(two_norm(big), WithinRel(5e200, 1e-15));
    dense_vector<std::complex<double>> small = {{3e-200, 0.0}, {0.0, 4e-200}};
    REQUIRE_THAT(two_norm(small), WithinRel(5e-200, 1e-15));
}

TEST_CASE("frobenius_norm does not overflow or underflow", "[operation][norms][690]") {
    dense2D<float> fbig = {{1e20f, 1e20f}, {1e20f, 1e20f}};
    REQUIRE_THAT(frobenius_norm(fbig), WithinRel(2e20f, 1e-6f));
    dense2D<double> dbig = {{3e200, 0.0}, {0.0, 4e200}};
    REQUIRE_THAT(frobenius_norm(dbig), WithinRel(5e200, 1e-15));
    dense2D<double> dsmall = {{1e-200, 1e-200}, {1e-200, 1e-200}};
    REQUIRE_THAT(frobenius_norm(dsmall), WithinRel(2e-200, 1e-15));
    dense2D<double> zero = {{0.0, 0.0}, {0.0, 0.0}};
    REQUIRE(frobenius_norm(zero) == 0.0);
}

TEST_CASE("vector norms propagate NaN in every position", "[operation][norms][690]") {
    for (std::size_t pos = 0; pos < 3; ++pos) {
        dense_vector<double> v = {1.0, -2.0, 3.0};
        v(pos) = nan_v<double>;
        CAPTURE(pos);
        REQUIRE(std::isnan(one_norm(v)));
        REQUIRE(std::isnan(two_norm(v)));
        REQUIRE(std::isnan(infinity_norm(v)));
        dense_vector<float> f = {1.0f, -2.0f, 3.0f};
        f(pos) = nan_v<float>;
        REQUIRE(std::isnan(infinity_norm(f)));
        REQUIRE(std::isnan(two_norm(f)));
    }
    // NaN beside an element whose square overflows: still NaN, not inf.
    dense_vector<double> big = {1e200, nan_v<double>};
    REQUIRE(std::isnan(two_norm(big)));
}

TEST_CASE("matrix norms propagate NaN in every position", "[operation][norms][690]") {
    for (std::size_t r = 0; r < 2; ++r) {
        for (std::size_t c = 0; c < 2; ++c) {
            dense2D<double> m = {{1.0, -2.0}, {3.0, 4.0}};
            m(r, c) = nan_v<double>;
            CAPTURE(r, c);
            REQUIRE(std::isnan(one_norm(m)));
            REQUIRE(std::isnan(infinity_norm(m)));
            REQUIRE(std::isnan(frobenius_norm(m)));
        }
    }
}

TEST_CASE("integer infinity_norm is unchanged by NaN propagation", "[operation][norms][690]") {
    dense_vector<int> v = {1, -7, 3};
    REQUIRE(infinity_norm(v) == 7);
    dense2D<int> m = {{1, -2}, {3, 4}};
    REQUIRE(one_norm(m) == 6);
    REQUIRE(infinity_norm(m) == 7);
}
