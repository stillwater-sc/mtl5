#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/generators/randorth.hpp>
#include <mtl/mat/operators.hpp>
#include <mtl/operation/trans.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/math/identity.hpp>

using namespace mtl;
using Catch::Matchers::WithinAbs;

TEST_CASE("randorth dimensions", "[generators][randorth]") {
    auto Q = generators::randorth<double>(5);
    REQUIRE(Q.num_rows() == 5);
    REQUIRE(Q.num_cols() == 5);
}

TEST_CASE("randorth is orthogonal (QtQ = I)", "[generators][randorth]") {
    auto Q = generators::randorth<double>(6);

    // Compute Q^T * Q
    mat::dense2D<double> QtQ = trans(Q) * Q;

    // Q^T * Q should be identity
    for (std::size_t i = 0; i < 6; ++i) {
        for (std::size_t j = 0; j < 6; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            REQUIRE_THAT(QtQ(i, j), WithinAbs(expected, 1e-10));
        }
    }
}

TEST_CASE("randorth QQt = I", "[generators][randorth]") {
    auto Q = generators::randorth<double>(5);

    // Compute Q * Q^T (should also be identity)
    mat::dense2D<double> QQt = Q * trans(Q);

    for (std::size_t i = 0; i < 5; ++i) {
        for (std::size_t j = 0; j < 5; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            REQUIRE_THAT(QQt(i, j), WithinAbs(expected, 1e-10));
        }
    }
}

TEST_CASE("randorth Frobenius norm of QtQ - I is small", "[generators][randorth]") {
    auto Q = generators::randorth<double>(8);

    mat::dense2D<double> QtQ = trans(Q) * Q;

    // Subtract identity
    for (std::size_t i = 0; i < 8; ++i)
        QtQ(i, i) -= 1.0;

    double err = frobenius_norm(QtQ);
    REQUIRE(err < 1e-10);
}

TEST_CASE("randorth different calls produce different matrices", "[generators][randorth]") {
    auto Q1 = generators::randorth<double>(4);
    auto Q2 = generators::randorth<double>(4);

    // At least one entry should differ (overwhelmingly likely)
    bool differ = false;
    for (std::size_t i = 0; i < 4 && !differ; ++i)
        for (std::size_t j = 0; j < 4 && !differ; ++j)
            if (std::abs(Q1(i, j) - Q2(i, j)) > 1e-14)
                differ = true;
    REQUIRE(differ);
}
