#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/io/read_el.hpp>
#include <mtl/io/write_el.hpp>

#include <cstdio>
#include <filesystem>
#include <string>

using namespace mtl;
using Catch::Matchers::WithinAbs;

static std::string temp_file(const std::string& name) {
    auto dir = std::filesystem::temp_directory_path();
    return (dir / ("mtl5_test_" + name)).string();
}

TEST_CASE("element I/O: dense CSV roundtrip", "[io][element]") {
    mat::dense2D<double> A(3, 4);
    double val = 1.0;
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            A(i, j) = val++;

    auto fname = temp_file("dense.csv");
    io::write_dense(fname, A);
    auto B = io::read_dense(fname);

    REQUIRE(B.num_rows() == 3);
    REQUIRE(B.num_cols() == 4);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            REQUIRE_THAT(B(i, j), WithinAbs(A(i, j), 1e-10));

    std::remove(fname.c_str());
}

TEST_CASE("element I/O: dense whitespace roundtrip", "[io][element]") {
    mat::dense2D<double> A(2, 3);
    A(0, 0) = 1.5; A(0, 1) = 2.5; A(0, 2) = 3.5;
    A(1, 0) = 4.5; A(1, 1) = 5.5; A(1, 2) = 6.5;

    auto fname = temp_file("dense_ws.txt");
    io::write_dense(fname, A, ' ');
    auto B = io::read_dense(fname, ' ');

    REQUIRE(B.num_rows() == 2);
    REQUIRE(B.num_cols() == 3);
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            REQUIRE_THAT(B(i, j), WithinAbs(A(i, j), 1e-10));

    std::remove(fname.c_str());
}

TEST_CASE("element I/O: sparse triplet roundtrip", "[io][element]") {
    mat::compressed2D<double> A(4, 4);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 5.0;
        ins[0][2] << 1.0;
        ins[1][1] << 3.0;
        ins[2][3] << 2.0;
        ins[3][0] << 4.0;
        ins[3][3] << 6.0;
    }

    auto fname = temp_file("sparse.tri");
    io::write_sparse(fname, A);
    auto B = io::read_sparse(fname, 4, 4);

    REQUIRE(B.num_rows() == 4);
    REQUIRE(B.num_cols() == 4);
    REQUIRE(B.nnz() == A.nnz());
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            REQUIRE_THAT(B(i, j), WithinAbs(A(i, j), 1e-10));

    std::remove(fname.c_str());
}

TEST_CASE("element I/O: read_dense throws on missing file", "[io][element]") {
    REQUIRE_THROWS_AS(io::read_dense("nonexistent_file_xyz.csv"), std::runtime_error);
}

TEST_CASE("element I/O: read_sparse throws on missing file", "[io][element]") {
    REQUIRE_THROWS_AS(io::read_sparse("nonexistent_file_xyz.tri", 3, 3), std::runtime_error);
}
