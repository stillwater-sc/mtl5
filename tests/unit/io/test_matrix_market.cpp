#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/io/matrix_market.hpp>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

using namespace mtl;

static std::string temp_file(const std::string& name) {
    auto dir = std::filesystem::temp_directory_path();
    return (dir / ("mtl5_test_" + name + ".mtx")).string();
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

// Issue #125: direct-CRS assembly must match the coordinate2D path, including
// unsorted input columns, the symmetric mirror, and duplicate accumulation.
TEST_CASE("Matrix Market: sparse read assembles sorted CRS (#125)", "[io][matrix_market]") {
    auto fname = temp_file("unsorted_sym");
    {
        std::ofstream out(fname);
        out << "%%MatrixMarket matrix coordinate real symmetric\n";
        out << "% intentionally unsorted entries\n";
        out << "3 3 4\n";
        out << "3 1 7.0\n";   // (2,0) and mirror (0,2)
        out << "2 2 5.0\n";   // (1,1) diagonal -> no mirror
        out << "1 1 4.0\n";   // (0,0)
        out << "3 2 9.0\n";   // (2,1) and mirror (1,2)
    }
    auto A = io::mm_read(fname);
    REQUIRE(A.num_rows() == 3);
    REQUIRE(A.nnz() == 6);                      // 2 diagonal + 2 off-diagonal*2 mirrors
    REQUIRE_THAT(A(0,0), Catch::Matchers::WithinAbs(4.0, 1e-12));
    REQUIRE_THAT(A(1,1), Catch::Matchers::WithinAbs(5.0, 1e-12));
    REQUIRE_THAT(A(2,0), Catch::Matchers::WithinAbs(7.0, 1e-12));
    REQUIRE_THAT(A(0,2), Catch::Matchers::WithinAbs(7.0, 1e-12));   // mirror
    REQUIRE_THAT(A(2,1), Catch::Matchers::WithinAbs(9.0, 1e-12));
    REQUIRE_THAT(A(1,2), Catch::Matchers::WithinAbs(9.0, 1e-12));   // mirror
    // CRS minor indices are sorted within each row.
    const auto& starts = A.ref_major();
    const auto& idx = A.ref_minor();
    for (std::size_t i = 0; i < A.num_rows(); ++i)
        for (std::size_t k = starts[i] + 1; k < starts[i+1]; ++k)
            REQUIRE(idx[k-1] < idx[k]);
    std::remove(fname.c_str());
}

#ifdef MTL5_HAS_ZLIB
#include <zlib.h>
// Transparent gzip reading: a .mtx.gz must read identically to the plain .mtx.
TEST_CASE("Matrix Market: transparent gzip read (#125)", "[io][matrix_market][gzip]") {
    // Build the plain text once.
    std::string body =
        "%%MatrixMarket matrix coordinate real general\n"
        "% gzip round-trip\n"
        "3 3 4\n"
        "1 1 4.0\n"
        "2 2 5.0\n"
        "3 3 6.0\n"
        "1 3 2.5\n";

    auto plain = temp_file("gz_plain");
    { std::ofstream out(plain); out << body; }
    auto gz = plain + ".gz";
    { gzFile f = gzopen(gz.c_str(), "wb");
      REQUIRE(f != nullptr);
      gzwrite(f, body.data(), static_cast<unsigned>(body.size()));
      gzclose(f); }

    auto A = io::mm_read(plain);     // plain
    auto B = io::mm_read(gz);        // gzipped -> must match

    REQUIRE(B.num_rows() == A.num_rows());
    REQUIRE(B.nnz() == A.nnz());
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            REQUIRE_THAT(B(i,j), Catch::Matchers::WithinAbs(A(i,j), 1e-12));

    std::remove(plain.c_str());
    std::remove(gz.c_str());
}
#else
// Without zlib, opening a .gz path must fail with a clear, actionable error.
TEST_CASE("Matrix Market: gzip without zlib throws (#125)", "[io][matrix_market][gzip]") {
    REQUIRE_THROWS_AS(io::mm_read("nonexistent_matrix.mtx.gz"), std::runtime_error);
}
#endif

// Issue #125 (review hardening): the reader must reject malformed / unsupported
// input instead of silently producing wrong or out-of-bounds results.
TEST_CASE("Matrix Market: reader input validation (#125)", "[io][matrix_market]") {
    auto write = [](const std::string& name, const std::string& body) {
        auto fn = temp_file(name);
        std::ofstream out(fn); out << body;
        return fn;
    };

    SECTION("coordinate index out of range throws (no OOB write)") {
        auto fn = write("oob", "%%MatrixMarket matrix coordinate real general\n3 3 1\n4 1 1.0\n");
        REQUIRE_THROWS_AS(io::mm_read(fn), std::runtime_error);
        std::remove(fn.c_str());
    }
    SECTION("zero index throws (would underflow)") {
        auto fn = write("zero", "%%MatrixMarket matrix coordinate real general\n3 3 1\n0 1 1.0\n");
        REQUIRE_THROWS_AS(io::mm_read(fn), std::runtime_error);
        std::remove(fn.c_str());
    }
    SECTION("skew-symmetric banner is rejected (not mis-mirrored)") {
        auto fn = write("skew", "%%MatrixMarket matrix coordinate real skew-symmetric\n2 2 1\n2 1 1.0\n");
        REQUIRE_THROWS_AS(io::mm_read(fn), std::runtime_error);
        std::remove(fn.c_str());
    }
    SECTION("complex field is rejected") {
        auto fn = write("cplx", "%%MatrixMarket matrix coordinate complex general\n1 1 1\n1 1 1.0 0.0\n");
        REQUIRE_THROWS_AS(io::mm_read(fn), std::runtime_error);
        std::remove(fn.c_str());
    }
    SECTION("missing dimension line throws") {
        auto fn = write("nodim", "%%MatrixMarket matrix coordinate real general\n% only comments\n");
        REQUIRE_THROWS_AS(io::mm_read(fn), std::runtime_error);
        std::remove(fn.c_str());
    }
    SECTION("non-square symmetric throws") {
        auto fn = write("nonsq", "%%MatrixMarket matrix coordinate real symmetric\n2 3 1\n1 1 1.0\n");
        REQUIRE_THROWS_AS(io::mm_read(fn), std::runtime_error);
        std::remove(fn.c_str());
    }
}
