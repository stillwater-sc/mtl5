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

// -- #826: braced construction, validation, P*A ---------------------------

#include <mtl/operation/reorder.hpp>
#include <stdexcept>

TEST_CASE("permutation_matrix: a braced list is the permutation", "[mat][permutation][826]") {
    // {0} used to pick the size constructor: a 0x0 matrix.
    permutation_matrix<double> P1({0});
    REQUIRE(P1.num_rows() == 1);
    REQUIRE(P1(0, 0) == 1.0);
    permutation_matrix<double> P3{2, 0, 1};
    REQUIRE(P3.num_rows() == 3);
    REQUIRE(P3(0, 2) == 1.0);
    REQUIRE(P3(1, 0) == 1.0);
    // The identity of size n is P(n), with parentheses.
    permutation_matrix<double> I(3);
    REQUIRE(I.num_rows() == 3);
    REQUIRE(I(2, 2) == 1.0);
}

TEST_CASE("permutation_matrix: empty", "[mat][permutation][826]") {
    permutation_matrix<double> E(std::vector<std::size_t>{});
    REQUIRE(E.num_rows() == 0);
    REQUIRE(E.inverse().num_rows() == 0);
    permutation_matrix<double> Z(std::size_t{0});
    REQUIRE(Z.num_rows() == 0);
}

TEST_CASE("permutation_matrix: an invalid permutation is rejected", "[mat][permutation][826]") {
    // {2} used to be the 2x2 identity; a duplicate or out-of-range entry made
    // inverse() and P * x index out of bounds.
    REQUIRE_THROWS_AS(permutation_matrix<double>({2}), std::invalid_argument);
    REQUIRE_THROWS_AS(permutation_matrix<double>(std::vector<std::size_t>{0, 0}), std::invalid_argument);
    REQUIRE_THROWS_AS(permutation_matrix<double>(std::vector<std::size_t>{0, 3, 1}), std::invalid_argument);
    REQUIRE_THROWS_AS((permutation_matrix<double>{1, 1, 0}), std::invalid_argument);
}

TEST_CASE("permutation_matrix: P * A permutes the rows", "[mat][permutation][826]") {
    permutation_matrix<double> P{2, 0, 1};
    mat::dense2D<double> A{{1, 2}, {3, 4}, {5, 6}};
    const auto B = P * A;   // operation/reorder.hpp
    REQUIRE(B.num_rows() == 3);
    REQUIRE(B(0, 0) == 5.0);
    REQUIRE(B(1, 1) == 2.0);
    REQUIRE(B(2, 0) == 3.0);
}
