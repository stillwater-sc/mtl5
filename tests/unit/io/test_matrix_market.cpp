#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/io/matrix_market.hpp>

#include <cstdio>
#include <fstream>
#include <string>

using namespace mtl;

static std::string temp_file(const std::string& name) {
    return "/tmp/mtl5_test_" + name + ".mtx";
}

TEST_CASE("Matrix Market: write and read dense", "[io][matrix_market]") {
    mat::dense2D<double> A(3, 3);
    A(0,0) = 1; A(0,1) = 2; A(0,2) = 3;
    A(1,0) = 4; A(1,1) = 5; A(1,2) = 6;
    A(2,0) = 7; A(2,1) = 8; A(2,2) = 9;

    auto fname = temp_file("dense");
    io::mm_write(fname, A, "test dense matrix");

    auto B = io::mm_read_dense(fname);

    REQUIRE(B.num_rows() == 3);
    REQUIRE(B.num_cols() == 3);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            REQUIRE_THAT(B(i, j), Catch::Matchers::WithinAbs(A(i, j), 1e-10));

    std::remove(fname.c_str());
}

TEST_CASE("Matrix Market: write and read sparse", "[io][matrix_market]") {
    mat::compressed2D<double> A(3, 3);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 4.0;
        ins[0][1] << 1.0;
        ins[1][0] << 1.0;
        ins[1][1] << 4.0;
        ins[1][2] << 1.0;
        ins[2][1] << 1.0;
        ins[2][2] << 4.0;
    }

    auto fname = temp_file("sparse");
    io::mm_write_sparse(fname, A, "test sparse matrix");

    auto B = io::mm_read(fname);

    REQUIRE(B.num_rows() == 3);
    REQUIRE(B.num_cols() == 3);
    REQUIRE(B.nnz() == A.nnz());

    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            REQUIRE_THAT(B(i, j), Catch::Matchers::WithinAbs(A(i, j), 1e-10));

    std::remove(fname.c_str());
}

TEST_CASE("Matrix Market: read coordinate symmetric", "[io][matrix_market]") {
    auto fname = temp_file("sym");

    // Write a symmetric .mtx file manually
    {
        std::ofstream out(fname);
        out << "%%MatrixMarket matrix coordinate real symmetric\n";
        out << "3 3 4\n";
        out << "1 1 4.0\n";
        out << "2 1 1.0\n";
        out << "2 2 5.0\n";
        out << "3 3 6.0\n";
    }

    auto A = io::mm_read_dense(fname);

    REQUIRE_THAT(A(0, 0), Catch::Matchers::WithinAbs(4.0, 1e-10));
    REQUIRE_THAT(A(1, 0), Catch::Matchers::WithinAbs(1.0, 1e-10));
    REQUIRE_THAT(A(0, 1), Catch::Matchers::WithinAbs(1.0, 1e-10)); // symmetric mirror
    REQUIRE_THAT(A(1, 1), Catch::Matchers::WithinAbs(5.0, 1e-10));
    REQUIRE_THAT(A(2, 2), Catch::Matchers::WithinAbs(6.0, 1e-10));

    std::remove(fname.c_str());
}
