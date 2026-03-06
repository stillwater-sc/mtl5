#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/generators/randsvd.hpp>
#include <mtl/operation/svd.hpp>
#include <mtl/operation/norms.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

using namespace mtl;
using Catch::Matchers::WithinAbs;

TEST_CASE("randsvd explicit sigma dimensions", "[generators][randsvd]") {
    std::vector<double> sigma = {3.0, 2.0, 1.0};
    auto A = generators::randsvd<double>(3, 3, sigma);
    REQUIRE(A.num_rows() == 3);
    REQUIRE(A.num_cols() == 3);
}

TEST_CASE("randsvd explicit sigma recovers singular values", "[generators][randsvd]") {
    std::vector<double> sigma = {5.0, 3.0, 1.0, 0.5};
    auto A = generators::randsvd<double>(4, 4, sigma);

    // Compute SVD of result
    mat::dense2D<double> U, S, V;
    svd(A, U, S, V, 1e-12);

    // Extract diagonal of S and sort descending
    std::vector<double> recovered(4);
    for (std::size_t i = 0; i < 4; ++i)
        recovered[i] = S(i, i);
    std::sort(recovered.begin(), recovered.end(), std::greater<double>());

    // Compare with prescribed sigma (sorted descending)
    std::vector<double> expected = sigma;
    std::sort(expected.begin(), expected.end(), std::greater<double>());

    for (std::size_t i = 0; i < 4; ++i) {
        REQUIRE_THAT(recovered[i], WithinAbs(expected[i], 0.1));
    }
}

TEST_CASE("randsvd condition number mode 1", "[generators][randsvd]") {
    double kappa = 100.0;
    auto A = generators::randsvd<double>(5, kappa, 1);

    mat::dense2D<double> U, S, V;
    svd(A, U, S, V, 1e-12);

    // Find max and min singular values
    double smax = 0, smin = 1e30;
    for (std::size_t i = 0; i < 5; ++i) {
        if (S(i, i) > smax) smax = S(i, i);
        if (S(i, i) < smin) smin = S(i, i);
    }
    double cond = smax / smin;
    REQUIRE_THAT(cond, WithinAbs(kappa, 5.0));
}

TEST_CASE("randsvd condition number mode 3 geometric", "[generators][randsvd]") {
    double kappa = 50.0;
    auto A = generators::randsvd<double>(4, kappa, 3);

    mat::dense2D<double> U, S, V;
    svd(A, U, S, V, 1e-12);

    double smax = 0, smin = 1e30;
    for (std::size_t i = 0; i < 4; ++i) {
        if (S(i, i) > smax) smax = S(i, i);
        if (S(i, i) < smin) smin = S(i, i);
    }
    double cond = smax / smin;
    REQUIRE_THAT(cond, WithinAbs(kappa, 5.0));
}

TEST_CASE("randsvd rectangular", "[generators][randsvd]") {
    std::vector<double> sigma = {4.0, 2.0, 1.0};
    auto A = generators::randsvd<double>(5, 3, sigma);
    REQUIRE(A.num_rows() == 5);
    REQUIRE(A.num_cols() == 3);
}

TEST_CASE("randsvd rectangular with kappa", "[generators][randsvd]") {
    double kappa = 10.0;
    auto A = generators::randsvd<double>(6, 4, kappa, 4);
    REQUIRE(A.num_rows() == 6);
    REQUIRE(A.num_cols() == 4);
}

TEST_CASE("randsvd make_sigma modes produce correct endpoints", "[generators][randsvd]") {
    double kappa = 100.0;
    std::size_t p = 5;

    for (int mode = 1; mode <= 5; ++mode) {
        auto sigma = generators::detail::make_sigma(p, kappa, mode);
        REQUIRE(sigma.size() == p);

        // Sort descending to find max/min
        auto sorted = sigma;
        std::sort(sorted.begin(), sorted.end(), std::greater<double>());

        // sigma_max should be 1, sigma_min should be 1/kappa
        // (mode 5 forces exact endpoints)
        REQUIRE_THAT(sorted.front(), WithinAbs(1.0, 1e-12));
        REQUIRE_THAT(sorted.back(), WithinAbs(1.0 / kappa, 1e-12));
    }
}
