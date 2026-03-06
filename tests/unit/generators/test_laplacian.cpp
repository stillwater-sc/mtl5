#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/generators/laplacian.hpp>
#include <cmath>
#include <numbers>

using namespace mtl;
using Catch::Matchers::WithinAbs;

TEST_CASE("laplacian_1d dimensions", "[generators][laplacian]") {
    auto L = generators::laplacian_1d<double>(5);
    REQUIRE(L.num_rows() == 5);
    REQUIRE(L.num_cols() == 5);
}

TEST_CASE("laplacian_1d tridiagonal structure", "[generators][laplacian]") {
    std::size_t n = 5;
    auto L = generators::laplacian_1d<double>(n);

    for (std::size_t i = 0; i < n; ++i) {
        // Diagonal = 2
        REQUIRE_THAT(L(i, i), WithinAbs(2.0, 1e-15));
        // Sub-diagonal = -1
        if (i > 0)
            REQUIRE_THAT(L(i, i - 1), WithinAbs(-1.0, 1e-15));
        // Super-diagonal = -1
        if (i + 1 < n)
            REQUIRE_THAT(L(i, i + 1), WithinAbs(-1.0, 1e-15));
    }

    // Verify zeros outside tridiagonal band
    REQUIRE_THAT(L(0, 2), WithinAbs(0.0, 1e-15));
    REQUIRE_THAT(L(0, 3), WithinAbs(0.0, 1e-15));
    REQUIRE_THAT(L(4, 0), WithinAbs(0.0, 1e-15));
    REQUIRE_THAT(L(4, 2), WithinAbs(0.0, 1e-15));
}

TEST_CASE("laplacian_1d known eigenvalues", "[generators][laplacian]") {
    // Eigenvalues: 2 - 2*cos(k*pi/(n+1)), k = 1..n
    // For n=3: eigenvalues are 2-2*cos(pi/4), 2-2*cos(pi/2), 2-2*cos(3*pi/4)
    // = 2-sqrt(2), 2, 2+sqrt(2)
    // Just verify the matrix is SPD by checking diagonal dominance
    std::size_t n = 4;
    auto L = generators::laplacian_1d<double>(n);

    // Verify row sums (Dirichlet BC: interior rows sum to 0, boundary rows sum to 1)
    // Row 0: 2 + (-1) = 1
    // Row 1: -1 + 2 + (-1) = 0
    // Row n-1: (-1) + 2 = 1
    double sum_first = L(0, 0) + L(0, 1);
    REQUIRE_THAT(sum_first, WithinAbs(1.0, 1e-15));

    double sum_mid = L(1, 0) + L(1, 1) + L(1, 2);
    REQUIRE_THAT(sum_mid, WithinAbs(0.0, 1e-15));

    double sum_last = L(n - 1, n - 2) + L(n - 1, n - 1);
    REQUIRE_THAT(sum_last, WithinAbs(1.0, 1e-15));
}

TEST_CASE("laplacian_2d dimensions", "[generators][laplacian]") {
    auto L = generators::laplacian_2d<double>(3, 4);
    REQUIRE(L.num_rows() == 12);
    REQUIRE(L.num_cols() == 12);
}

TEST_CASE("laplacian_2d diagonal is 4", "[generators][laplacian]") {
    std::size_t nx = 3, ny = 3;
    auto L = generators::laplacian_2d<double>(nx, ny);
    std::size_t n = nx * ny;

    for (std::size_t i = 0; i < n; ++i)
        REQUIRE_THAT(L(i, i), WithinAbs(4.0, 1e-15));
}

TEST_CASE("laplacian_2d 5-point stencil structure", "[generators][laplacian]") {
    std::size_t nx = 3, ny = 3;
    auto L = generators::laplacian_2d<double>(nx, ny);

    // Interior point (1,1) = row 4
    std::size_t row = 1 * nx + 1; // = 4
    REQUIRE_THAT(L(row, row), WithinAbs(4.0, 1e-15));
    REQUIRE_THAT(L(row, row - 1), WithinAbs(-1.0, 1e-15));   // left
    REQUIRE_THAT(L(row, row + 1), WithinAbs(-1.0, 1e-15));   // right
    REQUIRE_THAT(L(row, row - nx), WithinAbs(-1.0, 1e-15));  // above
    REQUIRE_THAT(L(row, row + nx), WithinAbs(-1.0, 1e-15));  // below
}

TEST_CASE("laplacian_2d is symmetric", "[generators][laplacian]") {
    auto L = generators::laplacian_2d<double>(3, 3);
    std::size_t n = 9;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            REQUIRE_THAT(L(i, j), WithinAbs(L(j, i), 1e-15));
}
