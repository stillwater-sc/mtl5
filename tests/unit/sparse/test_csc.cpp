#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/sparse/util/csc.hpp>

using namespace mtl;
using namespace mtl::sparse::util;

TEST_CASE("CRS to CSC conversion", "[sparse][csc]") {
    // A = [[1 0 2]
    //      [0 3 0]
    //      [4 0 5]]
    mat::compressed2D<double> A(3, 3);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 1.0; ins[0][2] << 2.0;
        ins[1][1] << 3.0;
        ins[2][0] << 4.0; ins[2][2] << 5.0;
    }

    auto csc = crs_to_csc(A);

    REQUIRE(csc.nrows == 3);
    REQUIRE(csc.ncols == 3);
    REQUIRE(csc.nnz() == 5);

    // Column 0: rows 0, 2 with values 1, 4
    REQUIRE(csc.col_ptr[0] == 0);
    REQUIRE(csc.col_ptr[1] == 2);
    REQUIRE(csc.row_ind[0] == 0);
    REQUIRE(csc.row_ind[1] == 2);
    REQUIRE(csc.values[0] == 1.0);
    REQUIRE(csc.values[1] == 4.0);

    // Column 1: row 1 with value 3
    REQUIRE(csc.col_ptr[2] == 3);
    REQUIRE(csc.row_ind[2] == 1);
    REQUIRE(csc.values[2] == 3.0);

    // Column 2: rows 0, 2 with values 2, 5
    REQUIRE(csc.col_ptr[3] == 5);
    REQUIRE(csc.row_ind[3] == 0);
    REQUIRE(csc.row_ind[4] == 2);
    REQUIRE(csc.values[3] == 2.0);
    REQUIRE(csc.values[4] == 5.0);
}

TEST_CASE("CSC element access", "[sparse][csc]") {
    mat::compressed2D<double> A(3, 3);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 1.0; ins[0][2] << 2.0;
        ins[1][1] << 3.0;
        ins[2][0] << 4.0; ins[2][2] << 5.0;
    }

    auto csc = crs_to_csc(A);

    REQUIRE(csc(0, 0) == 1.0);
    REQUIRE(csc(0, 1) == 0.0);
    REQUIRE(csc(0, 2) == 2.0);
    REQUIRE(csc(1, 0) == 0.0);
    REQUIRE(csc(1, 1) == 3.0);
    REQUIRE(csc(2, 0) == 4.0);
    REQUIRE(csc(2, 2) == 5.0);
}

TEST_CASE("CSC to CRS round-trip", "[sparse][csc]") {
    mat::compressed2D<double> A(3, 3);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 4.0; ins[0][1] << 1.0;
        ins[1][0] << 1.0; ins[1][1] << 3.0; ins[1][2] << 1.0;
        ins[2][1] << 1.0; ins[2][2] << 2.0;
    }

    auto csc = crs_to_csc(A);
    auto B = csc_to_crs(csc);

    REQUIRE(B.num_rows() == 3);
    REQUIRE(B.num_cols() == 3);
    REQUIRE(B.nnz() == A.nnz());

    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            REQUIRE(B(i, j) == A(i, j));
}

TEST_CASE("CSC transpose", "[sparse][csc]") {
    // A = [[1 2]
    //      [3 4]
    //      [5 6]]  (3x2)
    mat::compressed2D<double> A(3, 2);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 1.0; ins[0][1] << 2.0;
        ins[1][0] << 3.0; ins[1][1] << 4.0;
        ins[2][0] << 5.0; ins[2][1] << 6.0;
    }

    auto csc = crs_to_csc(A);
    auto At = transpose_csc(csc);

    REQUIRE(At.nrows == 2);
    REQUIRE(At.ncols == 3);

    // A^T = [[1 3 5]
    //        [2 4 6]]
    REQUIRE(At(0, 0) == 1.0);
    REQUIRE(At(0, 1) == 3.0);
    REQUIRE(At(0, 2) == 5.0);
    REQUIRE(At(1, 0) == 2.0);
    REQUIRE(At(1, 1) == 4.0);
    REQUIRE(At(1, 2) == 6.0);
}
