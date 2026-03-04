#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/svd.hpp>
#include <mtl/operation/operators.hpp>
#include <mtl/operation/trans.hpp>

#include <cmath>

using namespace mtl;

TEST_CASE("SVD: U*S*V^T reproduces A", "[operation][svd]") {
    mat::dense2D<double> A(3, 3);
    A(0,0) = 1; A(0,1) = 2; A(0,2) = 3;
    A(1,0) = 4; A(1,1) = 5; A(1,2) = 6;
    A(2,0) = 7; A(2,1) = 8; A(2,2) = 10;

    mat::dense2D<double> U, S, V;
    svd(A, U, S, V, 1e-12);

    // Reconstruct: A_approx = U * S * V^T
    auto SV = S * trans(V);
    auto A_approx = U * SV;

    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            REQUIRE_THAT(A_approx(i, j), Catch::Matchers::WithinAbs(A(i, j), 1e-8));
}

TEST_CASE("SVD: U and V are orthogonal", "[operation][svd]") {
    mat::dense2D<double> A(3, 3);
    A(0,0) = 1; A(0,1) = 0; A(0,2) = 0;
    A(1,0) = 0; A(1,1) = 2; A(1,2) = 0;
    A(2,0) = 0; A(2,1) = 0; A(2,2) = 3;

    mat::dense2D<double> U, S, V;
    svd(A, U, S, V, 1e-12);

    // U^T * U = I
    auto UtU = trans(U) * U;
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            REQUIRE_THAT(UtU(i, j), Catch::Matchers::WithinAbs(expected, 1e-8));
        }

    // V^T * V = I
    auto VtV = trans(V) * V;
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            REQUIRE_THAT(VtV(i, j), Catch::Matchers::WithinAbs(expected, 1e-8));
        }
}

TEST_CASE("SVD: singular values of diagonal matrix", "[operation][svd]") {
    mat::dense2D<double> A(3, 3);
    A(0,0) = 5; A(0,1) = 0; A(0,2) = 0;
    A(1,0) = 0; A(1,1) = 3; A(1,2) = 0;
    A(2,0) = 0; A(2,1) = 0; A(2,2) = 1;

    mat::dense2D<double> U, S, V;
    svd(A, U, S, V, 1e-12);

    // Singular values should be 5, 3, 1 (in S diagonal)
    std::vector<double> sv = {S(0,0), S(1,1), S(2,2)};
    std::sort(sv.begin(), sv.end());

    REQUIRE_THAT(sv[0], Catch::Matchers::WithinAbs(1.0, 1e-8));
    REQUIRE_THAT(sv[1], Catch::Matchers::WithinAbs(3.0, 1e-8));
    REQUIRE_THAT(sv[2], Catch::Matchers::WithinAbs(5.0, 1e-8));
}

TEST_CASE("SVD: tuple return form", "[operation][svd]") {
    mat::dense2D<double> A(2, 2);
    A(0,0) = 3; A(0,1) = 0;
    A(1,0) = 0; A(1,1) = 4;

    auto [U, S, V] = svd(A, 1e-12);

    // Reconstruct
    auto A_approx = U * S * trans(V);
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 2; ++j)
            REQUIRE_THAT(A_approx(i, j), Catch::Matchers::WithinAbs(A(i, j), 1e-8));
}
