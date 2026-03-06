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
