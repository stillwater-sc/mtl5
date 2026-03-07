#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/mat/permutation_matrix.hpp>
#include <mtl/mat/dense2D.hpp>

using namespace mtl;

TEST_CASE("permutation_matrix: identity permutation", "[mat][permutation]") {
    permutation_matrix<double> P(4);

    REQUIRE(P.num_rows() == 4);
    REQUIRE(P.num_cols() == 4);

    // Should be identity
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            REQUIRE_THAT(P(i, j), Catch::Matchers::WithinAbs(expected, 1e-10));
        }
}

TEST_CASE("permutation_matrix: explicit permutation", "[mat][permutation]") {
    // perm = [2, 0, 3, 1] means row 0 has 1 in col 2, row 1 has 1 in col 0, etc.
    permutation_matrix<double> P(std::vector<std::size_t>{2, 0, 3, 1});

    REQUIRE_THAT(P(0, 2), Catch::Matchers::WithinAbs(1.0, 1e-10));
    REQUIRE_THAT(P(1, 0), Catch::Matchers::WithinAbs(1.0, 1e-10));
    REQUIRE_THAT(P(2, 3), Catch::Matchers::WithinAbs(1.0, 1e-10));
    REQUIRE_THAT(P(3, 1), Catch::Matchers::WithinAbs(1.0, 1e-10));

    // Off entries should be zero
    REQUIRE_THAT(P(0, 0), Catch::Matchers::WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(P(0, 1), Catch::Matchers::WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(P(1, 1), Catch::Matchers::WithinAbs(0.0, 1e-10));
}

TEST_CASE("permutation_matrix: swap_rows", "[mat][permutation]") {
    permutation_matrix<double> P(3);
    P.swap_rows(0, 2); // swap rows 0 and 2

    REQUIRE_THAT(P(0, 2), Catch::Matchers::WithinAbs(1.0, 1e-10));
    REQUIRE_THAT(P(1, 1), Catch::Matchers::WithinAbs(1.0, 1e-10));
    REQUIRE_THAT(P(2, 0), Catch::Matchers::WithinAbs(1.0, 1e-10));
}

TEST_CASE("permutation_matrix: inverse", "[mat][permutation]") {
    permutation_matrix<double> P(std::vector<std::size_t>{2, 0, 3, 1});
    auto Pinv = P.inverse();

    // P * Pinv should be identity
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j) {
            double sum = 0.0;
            for (std::size_t k = 0; k < 4; ++k)
                sum += P(i, k) * Pinv(k, j);
            double expected = (i == j) ? 1.0 : 0.0;
            REQUIRE_THAT(sum, Catch::Matchers::WithinAbs(expected, 1e-10));
        }
}

TEST_CASE("permutation_matrix: efficient P*x matvec", "[mat][permutation]") {
    // perm = [2, 0, 1] -> y[0]=x[2], y[1]=x[0], y[2]=x[1]
    permutation_matrix<double> P(std::vector<std::size_t>{2, 0, 1});

    vec::dense_vector<double> x(3);
    x(0) = 10.0; x(1) = 20.0; x(2) = 30.0;

    auto y = P * x;

    REQUIRE_THAT(y(0), Catch::Matchers::WithinAbs(30.0, 1e-10));
    REQUIRE_THAT(y(1), Catch::Matchers::WithinAbs(10.0, 1e-10));
    REQUIRE_THAT(y(2), Catch::Matchers::WithinAbs(20.0, 1e-10));
}

TEST_CASE("permutation_matrix: P^{-1} * P * x = x", "[mat][permutation]") {
    permutation_matrix<double> P(std::vector<std::size_t>{3, 1, 0, 2});
    auto Pinv = P.inverse();

    vec::dense_vector<double> x(4);
    x(0) = 1.0; x(1) = 2.0; x(2) = 3.0; x(3) = 4.0;

    auto y = Pinv * (P * x);

    for (std::size_t i = 0; i < 4; ++i)
        REQUIRE_THAT(y(i), Catch::Matchers::WithinAbs(x(i), 1e-10));
}
