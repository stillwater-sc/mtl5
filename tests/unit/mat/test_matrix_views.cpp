#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/view/banded_view.hpp>
#include <mtl/mat/view/hermitian_view.hpp>
#include <mtl/mat/view/map_view.hpp>

using namespace mtl;

TEST_CASE("banded_view: tridiagonal band", "[mat][view][banded]") {
    mat::dense2D<double> A(4, 4);
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            A(i, j) = static_cast<double>(i * 4 + j + 1);

    auto B = banded(A, 1, 1); // tridiagonal

    REQUIRE(B.num_rows() == 4);
    REQUIRE(B.num_cols() == 4);

    // Diagonal
    REQUIRE_THAT(B(0, 0), Catch::Matchers::WithinAbs(A(0, 0), 1e-10));
    REQUIRE_THAT(B(1, 1), Catch::Matchers::WithinAbs(A(1, 1), 1e-10));

    // Sub- and super-diagonal
    REQUIRE_THAT(B(1, 0), Catch::Matchers::WithinAbs(A(1, 0), 1e-10));
    REQUIRE_THAT(B(0, 1), Catch::Matchers::WithinAbs(A(0, 1), 1e-10));

    // Outside band should be zero
    REQUIRE_THAT(B(0, 2), Catch::Matchers::WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(B(0, 3), Catch::Matchers::WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(B(3, 0), Catch::Matchers::WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(B(3, 1), Catch::Matchers::WithinAbs(0.0, 1e-10));
}

TEST_CASE("banded_view: diagonal only", "[mat][view][banded]") {
    mat::dense2D<double> A(3, 3);
    A(0,0) = 1; A(0,1) = 2; A(0,2) = 3;
    A(1,0) = 4; A(1,1) = 5; A(1,2) = 6;
    A(2,0) = 7; A(2,1) = 8; A(2,2) = 9;

    auto D = banded(A, 0, 0); // diagonal only

    REQUIRE_THAT(D(0, 0), Catch::Matchers::WithinAbs(1.0, 1e-10));
    REQUIRE_THAT(D(1, 1), Catch::Matchers::WithinAbs(5.0, 1e-10));
    REQUIRE_THAT(D(2, 2), Catch::Matchers::WithinAbs(9.0, 1e-10));
    REQUIRE_THAT(D(0, 1), Catch::Matchers::WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(D(1, 0), Catch::Matchers::WithinAbs(0.0, 1e-10));
}

TEST_CASE("hermitian_view: symmetric real matrix", "[mat][view][hermitian]") {
    // Store only upper triangle
    mat::dense2D<double> A(3, 3);
    A(0,0) = 4; A(0,1) = 2; A(0,2) = 1;
    A(1,0) = 0; A(1,1) = 5; A(1,2) = 3; // lower not stored
    A(2,0) = 0; A(2,1) = 0; A(2,2) = 6;

    auto H = hermitian(A);

    // Upper triangle unchanged
    REQUIRE_THAT(H(0, 1), Catch::Matchers::WithinAbs(2.0, 1e-10));
    REQUIRE_THAT(H(0, 2), Catch::Matchers::WithinAbs(1.0, 1e-10));

    // Lower triangle mirrors upper (for real, conj is identity)
    REQUIRE_THAT(H(1, 0), Catch::Matchers::WithinAbs(2.0, 1e-10));
    REQUIRE_THAT(H(2, 0), Catch::Matchers::WithinAbs(1.0, 1e-10));
    REQUIRE_THAT(H(2, 1), Catch::Matchers::WithinAbs(3.0, 1e-10));

    // Diagonal
    REQUIRE_THAT(H(0, 0), Catch::Matchers::WithinAbs(4.0, 1e-10));
    REQUIRE_THAT(H(1, 1), Catch::Matchers::WithinAbs(5.0, 1e-10));
    REQUIRE_THAT(H(2, 2), Catch::Matchers::WithinAbs(6.0, 1e-10));
}

TEST_CASE("map_view: row/column permutation", "[mat][view][map]") {
    mat::dense2D<double> A(3, 3);
    A(0,0) = 1; A(0,1) = 2; A(0,2) = 3;
    A(1,0) = 4; A(1,1) = 5; A(1,2) = 6;
    A(2,0) = 7; A(2,1) = 8; A(2,2) = 9;

    // Reverse row order
    std::vector<std::size_t> row_map = {2, 1, 0};
    std::vector<std::size_t> col_map = {0, 1, 2}; // identity

    auto M = mapped(A, row_map, col_map);

    REQUIRE(M.num_rows() == 3);
    REQUIRE(M.num_cols() == 3);

    // Row 0 of view = row 2 of A
    REQUIRE_THAT(M(0, 0), Catch::Matchers::WithinAbs(7.0, 1e-10));
    REQUIRE_THAT(M(0, 1), Catch::Matchers::WithinAbs(8.0, 1e-10));
    REQUIRE_THAT(M(0, 2), Catch::Matchers::WithinAbs(9.0, 1e-10));

    // Row 2 of view = row 0 of A
    REQUIRE_THAT(M(2, 0), Catch::Matchers::WithinAbs(1.0, 1e-10));
    REQUIRE_THAT(M(2, 1), Catch::Matchers::WithinAbs(2.0, 1e-10));
}

TEST_CASE("map_view: submatrix extraction", "[mat][view][map]") {
    mat::dense2D<double> A(4, 4);
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            A(i, j) = static_cast<double>(i * 4 + j);

    // Extract 2x2 submatrix: rows {1,3}, cols {0,2}
    std::vector<std::size_t> row_map = {1, 3};
    std::vector<std::size_t> col_map = {0, 2};

    auto S = mapped(A, row_map, col_map);

    REQUIRE(S.num_rows() == 2);
    REQUIRE(S.num_cols() == 2);
    REQUIRE_THAT(S(0, 0), Catch::Matchers::WithinAbs(4.0, 1e-10));  // A(1,0)
    REQUIRE_THAT(S(0, 1), Catch::Matchers::WithinAbs(6.0, 1e-10));  // A(1,2)
    REQUIRE_THAT(S(1, 0), Catch::Matchers::WithinAbs(12.0, 1e-10)); // A(3,0)
    REQUIRE_THAT(S(1, 1), Catch::Matchers::WithinAbs(14.0, 1e-10)); // A(3,2)
}

// -- #826: complex, non-square and invalid maps ----------------------------

#include <complex>
#include <stdexcept>

TEST_CASE("hermitian_view: complex", "[mat][view][hermitian][826]") {
    using C = std::complex<double>;
    mat::dense2D<C> A(2, 2);
    A(0, 0) = C(1, 0); A(0, 1) = C(2, 3);
    A(1, 0) = C(9, 9); A(1, 1) = C(4, 0);   // the lower triangle is not read
    auto H = mtl::hermitian(A);
    REQUIRE(H(0, 1) == C(2, 3));
    REQUIRE(H(1, 0) == C(2, -3));
    REQUIRE(H(0, 0) == C(1, 0));
    REQUIRE(H(1, 1) == C(4, 0));
}

TEST_CASE("hermitian_view: a non-square matrix is rejected", "[mat][view][hermitian][826]") {
    // For a 3x2 A, H(2,0) read A(0,2).
    mat::dense2D<double> A(3, 2);
    REQUIRE_THROWS_AS(mtl::hermitian(A), std::invalid_argument);
}

TEST_CASE("banded_view: negative bandwidths and non-square", "[mat][view][banded][826]") {
    mat::dense2D<double> A{{1, 2, 3, 4}, {5, 6, 7, 8}};   // 2x4
    // lower = -1: the band starts one above the diagonal, c - r in [1, 2].
    auto B = mtl::banded(A, -1, 2);
    REQUIRE(B.num_rows() == 2);
    REQUIRE(B.num_cols() == 4);
    REQUIRE(B(0, 0) == 0.0);
    REQUIRE(B(0, 1) == 2.0);
    REQUIRE(B(0, 2) == 3.0);
    REQUIRE(B(0, 3) == 0.0);
    REQUIRE(B(1, 3) == 8.0);
    // upper < -lower: an empty band.
    auto Z = mtl::banded(A, 0, -1);
    for (std::size_t r = 0; r < 2; ++r)
        for (std::size_t c = 0; c < 4; ++c) REQUIRE(Z(r, c) == 0.0);
}

TEST_CASE("map_view: empty and repeated-index maps", "[mat][view][map][826]") {
    mat::dense2D<double> A{{1, 2}, {3, 4}};
    auto E = mtl::mapped(A, std::vector<std::size_t>{}, std::vector<std::size_t>{0, 1});
    REQUIRE(E.num_rows() == 0);
    REQUIRE(E.num_cols() == 2);
    auto R = mtl::mapped(A, std::vector<std::size_t>{1, 1, 0}, std::vector<std::size_t>{0, 0});
    REQUIRE(R.num_rows() == 3);
    REQUIRE(R(0, 0) == 3.0);
    REQUIRE(R(1, 1) == 3.0);
    REQUIRE(R(2, 1) == 1.0);
}

TEST_CASE("map_view: a map entry out of range is rejected", "[mat][view][map][826]") {
    mat::dense2D<double> A{{1, 2}, {3, 4}};
    REQUIRE_THROWS_AS(mtl::mapped(A, std::vector<std::size_t>{0, 2}, std::vector<std::size_t>{0}),
                      std::out_of_range);
    REQUIRE_THROWS_AS(mtl::mapped(A, std::vector<std::size_t>{0}, std::vector<std::size_t>{5}),
                      std::out_of_range);
}
