#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/generators/moler.hpp>

using namespace mtl;
using Catch::Matchers::WithinAbs;

TEST_CASE("moler dimensions", "[generators][moler]") {
    auto M = generators::moler<double>(5);
    REQUIRE(M.num_rows() == 5);
    REQUIRE(M.num_cols() == 5);
}

TEST_CASE("moler is symmetric", "[generators][moler]") {
    auto M = generators::moler<double>(5);
    for (std::size_t i = 0; i < 5; ++i)
        for (std::size_t j = 0; j < 5; ++j)
            REQUIRE_THAT(M(i, j), WithinAbs(M(j, i), 1e-15));
}

TEST_CASE("moler 3x3 known values", "[generators][moler]") {
    auto M = generators::moler<double>(3);
    // M = L*L^T where L = [[1,0,0],[-1,1,0],[-1,-1,1]]
    // Result:
    // [ 1 -1 -1]
    // [-1  2  0]
    // [-1  0  3]
    REQUIRE_THAT(M(0, 0), WithinAbs(1.0, 1e-15));
    REQUIRE_THAT(M(0, 1), WithinAbs(-1.0, 1e-15));
    REQUIRE_THAT(M(0, 2), WithinAbs(-1.0, 1e-15));
    REQUIRE_THAT(M(1, 0), WithinAbs(-1.0, 1e-15));
    REQUIRE_THAT(M(1, 1), WithinAbs(2.0, 1e-15));
    REQUIRE_THAT(M(1, 2), WithinAbs(0.0, 1e-15));
    REQUIRE_THAT(M(2, 0), WithinAbs(-1.0, 1e-15));
    REQUIRE_THAT(M(2, 1), WithinAbs(0.0, 1e-15));
    REQUIRE_THAT(M(2, 2), WithinAbs(3.0, 1e-15));
}

TEST_CASE("moler equals L*L^T with unit lower triangular L", "[generators][moler]") {
    std::size_t n = 4;
    double alpha = -1.0;
    auto M = generators::moler<double>(n, alpha);

    // Build L manually: L(i,i)=1, L(i,j)=alpha for i>j, 0 for i<j
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            double llt = 0.0;
            for (std::size_t k = 0; k < n; ++k) {
                double l_ik = (i == k) ? 1.0 : ((i > k) ? alpha : 0.0);
                double l_jk = (j == k) ? 1.0 : ((j > k) ? alpha : 0.0);
                llt += l_ik * l_jk;
            }
            REQUIRE_THAT(M(i, j), WithinAbs(llt, 1e-12));
        }
    }
}

TEST_CASE("moler with custom alpha", "[generators][moler]") {
    double alpha = -0.5;
    auto M = generators::moler<double>(3, alpha);
    // M(i,i) = i*alpha^2 + 1
    // M(i,j) = min(i,j)*alpha^2 + alpha for i != j
    double a2 = alpha * alpha; // 0.25
    REQUIRE_THAT(M(0, 0), WithinAbs(1.0, 1e-15));          // 0*0.25 + 1
    REQUIRE_THAT(M(1, 1), WithinAbs(1.25, 1e-15));         // 1*0.25 + 1
    REQUIRE_THAT(M(2, 2), WithinAbs(1.5, 1e-15));          // 2*0.25 + 1
    REQUIRE_THAT(M(0, 1), WithinAbs(-0.5, 1e-15));         // 0*0.25 + (-0.5)
    REQUIRE_THAT(M(1, 2), WithinAbs(alpha + a2, 1e-15));   // 1*0.25 + (-0.5) = -0.25
}
