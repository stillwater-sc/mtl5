#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/io/read_el.hpp>
#include <mtl/io/write_el.hpp>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
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

// ---------------------------------------------------------------------------
// #843: the element-list readers throw on input they cannot read exactly,
// instead of returning a partial or padded matrix.

namespace {

/// A file in the temp directory, removed when the test ends.
struct temp_text {
    std::string path;
    temp_text(const std::string& name, const std::string& body) : path(temp_file(name)) {
        std::ofstream out(path, std::ios::binary);
        out << body;
    }
    ~temp_text() { std::remove(path.c_str()); }
    temp_text(const temp_text&) = delete;
    temp_text& operator=(const temp_text&) = delete;
};

} // namespace

TEST_CASE("element I/O: read_dense rejects bad tokens, empty fields and ragged rows (#843)", "[io][element]") {
    // "1,2,x,4" used to give a 1x2 matrix: the stream stopped at x.
    temp_text bad("bad_token.csv", "1,2,x,4\n");
    REQUIRE_THROWS_AS(io::read_dense(bad.path), std::runtime_error);
    temp_text trail("trailing.csv", "1,2,3x\n");
    REQUIRE_THROWS_AS(io::read_dense(trail.path), std::runtime_error);
    temp_text empty_field("empty_field.csv", "1,,2\n");
    REQUIRE_THROWS_AS(io::read_dense(empty_field.path), std::runtime_error);
    // Rows of other lengths used to be zero-padded or truncated to row 0's width.
    temp_text longer("ragged_long.csv", "1,2\n3,4,5\n");
    REQUIRE_THROWS_AS(io::read_dense(longer.path), std::runtime_error);
    temp_text shorter("ragged_short.csv", "1,2,3\n4,5\n");
    REQUIRE_THROWS_AS(io::read_dense(shorter.path), std::runtime_error);
    temp_text empty("empty.csv", "");
    REQUIRE_THROWS_AS(io::read_dense(empty.path), std::runtime_error);
    temp_text blank("blank.csv", "\n  \n\t\n");
    REQUIRE_THROWS_AS(io::read_dense(blank.path), std::runtime_error);
}

TEST_CASE("element I/O: read_dense delimiters, spacing, CRLF and blank lines", "[io][element]") {
    temp_text semi("semi.txt", "1.5; -2\r\n\r\n  3 ;4e1  \r\n");
    auto A = io::read_dense(semi.path, ';');
    REQUIRE((A.num_rows() == 2 && A.num_cols() == 2));
    REQUIRE((A(0, 0) == 1.5 && A(0, 1) == -2.0 && A(1, 0) == 3.0 && A(1, 1) == 40.0));
    temp_text tabs("tabs.txt", "1\t2\n3\t4\n");
    auto T = io::read_dense(tabs.path, '\t');
    REQUIRE((T(0, 1) == 2.0 && T(1, 0) == 3.0));
    temp_text ws("ws.txt", "  1   2\n3 \t 4  \n");
    auto W = io::read_dense(ws.path, ' ');
    REQUIRE((W.num_cols() == 2 && W(1, 1) == 4.0));
    temp_text one("one.csv", "7");
    REQUIRE(io::read_dense(one.path)(0, 0) == 7.0);
}

TEST_CASE("element I/O: read_sparse rejects bad lines and out-of-range indices (#843)", "[io][element]") {
    // A bad token used to stop the read and return the entries before it.
    temp_text tok("bad.tri", "0 0 1.0\nx\n1 1 2.0\n");
    REQUIRE_THROWS_AS(io::read_sparse(tok.path, 2, 2), std::runtime_error);
    temp_text shortl("short.tri", "0 0\n");
    REQUIRE_THROWS_AS(io::read_sparse(shortl.path, 2, 2), std::runtime_error);
    temp_text longl("long.tri", "0 0 1.0 9\n");
    REQUIRE_THROWS_AS(io::read_sparse(longl.path, 2, 2), std::runtime_error);
    temp_text neg("neg.tri", "-1 0 1.0\n");
    REQUIRE_THROWS_AS(io::read_sparse(neg.path, 2, 2), std::runtime_error);
    // An index outside the dimensions: std::out_of_range in every build. It
    // used to hit coordinate2D::insert's assert, and under NDEBUG compress()
    // indexed out of range (fixed in compress() by #831, checked here per line).
    temp_text oor("oor.tri", "0 0 1.0\n5 7 1.0\n");
    REQUIRE_THROWS_AS(io::read_sparse(oor.path, 2, 2), std::out_of_range);
    temp_text oor_col("oor_col.tri", "0 2 1.0\n");
    REQUIRE_THROWS_AS(io::read_sparse(oor_col.path, 2, 2), std::out_of_range);
}

TEST_CASE("element I/O: read_sparse blank lines, CRLF, duplicates and an empty file", "[io][element]") {
    temp_text f("dups.tri", "0 1 1.5\r\n\r\n  \n0 1 2.5\r\n1 0 -1\r\n");
    auto A = io::read_sparse(f.path, 2, 2);
    REQUIRE((A.nnz() == 2 && A(0, 1) == 4.0 && A(1, 0) == -1.0));
    temp_text empty("empty.tri", "");
    auto E = io::read_sparse(empty.path, 3, 2);
    REQUIRE((E.num_rows() == 3 && E.num_cols() == 2 && E.nnz() == 0));
}
