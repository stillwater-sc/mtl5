#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/mat/block_diagonal2D.hpp>

using namespace mtl;

static mat::dense2D<double> make_block(std::size_t rows, std::size_t cols,
                                       double start) {
    mat::dense2D<double> B(rows, cols);
    double val = start;
    for (std::size_t i = 0; i < rows; ++i)
        for (std::size_t j = 0; j < cols; ++j)
            B(i, j) = val++;
    return B;
}

TEST_CASE("block_diagonal2D: dimensions", "[mat][block_diagonal]") {
    auto B1 = make_block(2, 2, 1.0);
    auto B2 = make_block(3, 3, 5.0);

    block_diagonal2D<double> BD({B1, B2});

    REQUIRE(BD.num_rows() == 5);
    REQUIRE(BD.num_cols() == 5);
    REQUIRE(BD.num_blocks() == 2);
}

TEST_CASE("block_diagonal2D: element access", "[mat][block_diagonal]") {
    // Block 0: 2x2 [[1,2],[3,4]]
    // Block 1: 2x2 [[5,6],[7,8]]
    auto B1 = make_block(2, 2, 1.0);
    auto B2 = make_block(2, 2, 5.0);

    block_diagonal2D<double> BD({B1, B2});

    // Block 0 region (rows 0-1, cols 0-1)
    REQUIRE_THAT(BD(0, 0), Catch::Matchers::WithinAbs(1.0, 1e-10));
    REQUIRE_THAT(BD(0, 1), Catch::Matchers::WithinAbs(2.0, 1e-10));
    REQUIRE_THAT(BD(1, 0), Catch::Matchers::WithinAbs(3.0, 1e-10));
    REQUIRE_THAT(BD(1, 1), Catch::Matchers::WithinAbs(4.0, 1e-10));

    // Block 1 region (rows 2-3, cols 2-3)
    REQUIRE_THAT(BD(2, 2), Catch::Matchers::WithinAbs(5.0, 1e-10));
    REQUIRE_THAT(BD(2, 3), Catch::Matchers::WithinAbs(6.0, 1e-10));
    REQUIRE_THAT(BD(3, 2), Catch::Matchers::WithinAbs(7.0, 1e-10));
    REQUIRE_THAT(BD(3, 3), Catch::Matchers::WithinAbs(8.0, 1e-10));

    // Off-diagonal blocks should be zero
    REQUIRE_THAT(BD(0, 2), Catch::Matchers::WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(BD(0, 3), Catch::Matchers::WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(BD(1, 2), Catch::Matchers::WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(BD(2, 0), Catch::Matchers::WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(BD(3, 0), Catch::Matchers::WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(BD(3, 1), Catch::Matchers::WithinAbs(0.0, 1e-10));
}

TEST_CASE("block_diagonal2D: add_block", "[mat][block_diagonal]") {
    block_diagonal2D<double> BD;
    BD.add_block(make_block(2, 2, 1.0));
    REQUIRE(BD.num_rows() == 2);
    REQUIRE(BD.num_cols() == 2);
    REQUIRE(BD.num_blocks() == 1);

    BD.add_block(make_block(3, 3, 10.0));
    REQUIRE(BD.num_rows() == 5);
    REQUIRE(BD.num_cols() == 5);
    REQUIRE(BD.num_blocks() == 2);
}

TEST_CASE("block_diagonal2D: efficient matvec", "[mat][block_diagonal]") {
    // B1 = [[1,0],[0,2]], B2 = [[3]]
    mat::dense2D<double> B1(2, 2);
    B1(0, 0) = 1.0; B1(0, 1) = 0.0;
    B1(1, 0) = 0.0; B1(1, 1) = 2.0;

    mat::dense2D<double> B2(1, 1);
    B2(0, 0) = 3.0;

    block_diagonal2D<double> BD({B1, B2});

    vec::dense_vector<double> x(3);
    x(0) = 10.0; x(1) = 20.0; x(2) = 30.0;

    auto y = BD * x;

    REQUIRE(y.size() == 3);
    REQUIRE_THAT(y(0), Catch::Matchers::WithinAbs(10.0, 1e-10));  // 1*10 + 0*20
    REQUIRE_THAT(y(1), Catch::Matchers::WithinAbs(40.0, 1e-10));  // 0*10 + 2*20
    REQUIRE_THAT(y(2), Catch::Matchers::WithinAbs(90.0, 1e-10));  // 3*30
}

TEST_CASE("block_diagonal2D: three blocks matvec", "[mat][block_diagonal]") {
    // B1 = [[2]], B2 = [[1,1],[1,1]], B3 = [[3]]
    mat::dense2D<double> B1(1, 1);
    B1(0, 0) = 2.0;

    mat::dense2D<double> B2(2, 2);
    B2(0, 0) = 1.0; B2(0, 1) = 1.0;
    B2(1, 0) = 1.0; B2(1, 1) = 1.0;

    mat::dense2D<double> B3(1, 1);
    B3(0, 0) = 3.0;

    block_diagonal2D<double> BD({B1, B2, B3});

    REQUIRE(BD.num_rows() == 4);
    REQUIRE(BD.num_cols() == 4);

    vec::dense_vector<double> x(4);
    x(0) = 1.0; x(1) = 2.0; x(2) = 3.0; x(3) = 4.0;

    auto y = BD * x;

    REQUIRE_THAT(y(0), Catch::Matchers::WithinAbs(2.0, 1e-10));   // 2*1
    REQUIRE_THAT(y(1), Catch::Matchers::WithinAbs(5.0, 1e-10));   // 1*2 + 1*3
    REQUIRE_THAT(y(2), Catch::Matchers::WithinAbs(5.0, 1e-10));   // 1*2 + 1*3
    REQUIRE_THAT(y(3), Catch::Matchers::WithinAbs(12.0, 1e-10));  // 3*4
}

TEST_CASE("block_diagonal2D: block access", "[mat][block_diagonal]") {
    auto B1 = make_block(2, 2, 1.0);
    auto B2 = make_block(3, 3, 5.0);

    block_diagonal2D<double> BD({B1, B2});

    REQUIRE(BD.block(0).num_rows() == 2);
    REQUIRE(BD.block(0).num_cols() == 2);
    REQUIRE(BD.block(1).num_rows() == 3);
    REQUIRE(BD.block(1).num_cols() == 3);

    REQUIRE_THAT(BD.block(0)(0, 0), Catch::Matchers::WithinAbs(1.0, 1e-10));
    REQUIRE_THAT(BD.block(1)(0, 0), Catch::Matchers::WithinAbs(5.0, 1e-10));
}

// -- #826: shapes, mutation, lookup ----------------------------------------

#include <mtl/vec/dense_vector.hpp>
#include <type_traits>

TEST_CASE("block_diagonal2D: non-square blocks", "[mat][block_diagonal][826]") {
    block_diagonal2D<double> BD{mat::dense2D<double>{{1, 2, 3}},          // 1x3
                                mat::dense2D<double>{{4}, {5}}};           // 2x1
    REQUIRE(BD.num_rows() == 3);
    REQUIRE(BD.num_cols() == 4);
    REQUIRE(BD(0, 2) == 3.0);
    REQUIRE(BD(1, 3) == 4.0);
    REQUIRE(BD(2, 3) == 5.0);
    REQUIRE(BD(1, 0) == 0.0);
    REQUIRE(BD(0, 3) == 0.0);
    vec::dense_vector<double> x(4, 1.0);
    const auto y = BD * x;
    REQUIRE(y.size() == 3);
    REQUIRE(y(0) == 6.0);
    REQUIRE(y(2) == 5.0);
}

TEST_CASE("block_diagonal2D: empty, and empty blocks", "[mat][block_diagonal][826]") {
    block_diagonal2D<double> E;
    REQUIRE(E.num_rows() == 0);
    REQUIRE(E.num_blocks() == 0);
    block_diagonal2D<double> L(std::vector<mat::dense2D<double>>{});
    REQUIRE(L.num_cols() == 0);
    block_diagonal2D<double> Z{mat::dense2D<double>{{1}}, mat::dense2D<double>(0, 0),
                               mat::dense2D<double>{{2}}};
    REQUIRE(Z.num_rows() == 2);
    REQUIRE(Z(0, 0) == 1.0);
    REQUIRE(Z(1, 1) == 2.0);
    REQUIRE(Z(1, 0) == 0.0);
}

TEST_CASE("block_diagonal2D: mixed value types in the matvec", "[mat][block_diagonal][826]") {
    block_diagonal2D<float> BD{mat::dense2D<float>{{1, 2}, {3, 4}}};
    vec::dense_vector<double> x(2);
    x(0) = 0.5; x(1) = 0.25;
    const auto y = BD * x;
    STATIC_REQUIRE(std::is_same_v<std::remove_cvref_t<decltype(y(0))>, double>);
    REQUIRE(y(0) == 1.0);
    REQUIRE(y(1) == 2.5);
}

TEST_CASE("block_diagonal2D: writing a block, and resizing what block() returns", "[mat][block_diagonal][826]") {
    block_diagonal2D<double> BD{mat::dense2D<double>{{1, 2}, {3, 4}}, mat::dense2D<double>{{5}}};
    auto B0 = BD.block(0);
    B0(0, 1) = 20.0;                 // writes the block
    REQUIRE(BD(0, 1) == 20.0);
    B0.change_dim(5, 5);             // detaches; the block and the offsets stand
    REQUIRE(BD.num_rows() == 3);
    REQUIRE(BD.block(0).num_rows() == 2);
    REQUIRE(BD(2, 2) == 5.0);
    BD.set_block(0, mat::dense2D<double>{{7}});   // a new shape, offsets updated
    REQUIRE(BD.num_rows() == 2);
    REQUIRE(BD(0, 0) == 7.0);
    REQUIRE(BD(1, 1) == 5.0);
}

TEST_CASE("block_diagonal2D: add_block keeps the offsets right", "[mat][block_diagonal][826]") {
    block_diagonal2D<double> BD;
    for (int k = 0; k < 50; ++k) BD.add_block(mat::dense2D<double>{{double(k)}, {double(k) + 0.5}});
    REQUIRE(BD.num_rows() == 100);
    REQUIRE(BD.num_cols() == 50);
    for (std::size_t k = 0; k < 50; ++k) {
        REQUIRE(BD(2 * k, k) == double(k));
        REQUIRE(BD(2 * k + 1, k) == double(k) + 0.5);
        if (k > 0) REQUIRE(BD(2 * k, k - 1) == 0.0);
    }
}
