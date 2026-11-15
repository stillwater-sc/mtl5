#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/io/matrix_market.hpp>

#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
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

// Duplicate coordinates sum, in the dense reader as in the sparse one and SciPy.
// mm_read_dense used to keep only the last value.
TEST_CASE("Matrix Market: duplicate entries accumulate in both readers", "[io][matrix_market]") {
    struct dup_case {
        const char* name;
        const char* body;
        double expected[3][3];
    };
    const dup_case cases[] = {
        {"dup_general",
         "%%MatrixMarket matrix coordinate real general\n"
         "3 3 4\n"
         "1 1 2.0\n"
         "1 1 3.0\n"      // (0,0) = 2 + 3
         "3 2 1.5\n"
         "2 3 4.0\n",
         {{5.0, 0.0, 0.0}, {0.0, 0.0, 4.0}, {0.0, 1.5, 0.0}}},
        {"dup_symmetric",
         "%%MatrixMarket matrix coordinate real symmetric\n"
         "3 3 4\n"
         "2 1 2.0\n"
         "2 1 3.0\n"      // (1,0) and mirror (0,1) = 2 + 3
         "3 3 1.0\n"
         "3 3 6.0\n",     // diagonal duplicate, no mirror: 1 + 6
         {{0.0, 5.0, 0.0}, {5.0, 0.0, 0.0}, {0.0, 0.0, 7.0}}},
        {"dup_cancel",
         "%%MatrixMarket matrix coordinate real symmetric\n"
         "3 3 3\n"
         "3 1 4.0\n"
         "3 1 -4.0\n"     // cancels to zero, mirror included
         "2 2 1.0\n",
         {{0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 0.0}}},
    };
    for (const auto& tc : cases) {
        INFO(tc.name);
        auto fname = temp_file(tc.name);
        { std::ofstream out(fname); out << tc.body; }
        auto D = io::mm_read_dense(fname);
        auto S = io::mm_read(fname);
        REQUIRE(D.num_rows() == 3);
        REQUIRE(D.num_cols() == 3);
        for (std::size_t i = 0; i < 3; ++i)
            for (std::size_t j = 0; j < 3; ++j) {
                INFO("(" << i << ", " << j << ")");
                REQUIRE(D(i, j) == tc.expected[i][j]);
                REQUIRE(S(i, j) == tc.expected[i][j]);
            }
        std::remove(fname.c_str());
    }
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

namespace {
std::string write_gz(const std::string& name, const std::string& body) {
    auto gz = temp_file(name) + ".gz";
    gzFile f = gzopen(gz.c_str(), "wb");
    REQUIRE(f != nullptr);
    gzwrite(f, body.data(), static_cast<unsigned>(body.size()));
    gzclose(f);
    return gz;
}
} // namespace

TEST_CASE("Matrix Market: gzip long lines, no trailing newline, mm_read_dense (#843)",
          "[io][matrix_market][gzip]") {
    // A line longer than the 64 KB gzgets buffer is assembled from pieces:
    // a long comment, and an entry line padded with 70000 spaces.
    const std::string body =
        "%%MatrixMarket matrix coordinate real general\n"
        "% " + std::string(70000, 'c') + "\n"
        "2 2 2\n"
        + std::string(70000, ' ') + "1 1 4.0\n"
        "2 2 5.0";                                  // no trailing newline
    auto gz = write_gz("gz_long", body);
    auto A = io::mm_read(gz);
    REQUIRE((A.nnz() == 2 && A(0, 0) == 4.0 && A(1, 1) == 5.0));
    auto D = io::mm_read_dense(gz);
    REQUIRE((D(0, 0) == 4.0 && D(1, 1) == 5.0 && D(0, 1) == 0.0));
    std::remove(gz.c_str());
}

TEST_CASE("Matrix Market: an overstated nnz in a gzip file reaches the EOF error (#843)",
          "[io][matrix_market][gzip]") {
    // A gzip file's size bounds nothing, so the header nnz alone used to size
    // the reserve: about 240 GB here, bad_alloc or the OOM killer instead of
    // the runtime_error for the missing data.
    auto gz = write_gz("gz_overstated",
        "%%MatrixMarket matrix coordinate real general\n100000 100000 9999999999\n1 1 1.0\n");
    REQUIRE_THROWS_AS(io::mm_read(gz), std::runtime_error);
    std::remove(gz.c_str());
}

TEST_CASE("Matrix Market: a truncated gzip stream throws (#843)", "[io][matrix_market][gzip]") {
    // gzgets returning nullptr was taken as EOF without asking gzerror, so a
    // truncated file read as a short one. Cutting the last 4 bytes (part of
    // the gzip trailer) leaves every entry readable, so only the gzerror
    // check can notice; the old reader returned the full matrix.
    std::string body = "%%MatrixMarket matrix coordinate real general\n2000 2000 2000\n";
    for (int k = 1; k <= 2000; ++k)
        body += std::to_string(k) + " " + std::to_string((k * 7) % 2000 + 1) + " "
              + std::to_string(k * 0.001) + "\n";
    auto gz = write_gz("gz_trunc", body);
    std::string bytes;
    { std::ifstream in(gz, std::ios::binary); bytes.assign(std::istreambuf_iterator<char>(in), {}); }
    REQUIRE(bytes.size() > 200);
    { std::ofstream out(gz, std::ios::binary | std::ios::trunc);
      out.write(bytes.data(), static_cast<std::streamsize>(bytes.size() - 4)); }
    REQUIRE_THROWS_AS(io::mm_read(gz), std::runtime_error);
    REQUIRE_THROWS_AS(io::mm_read_dense(gz), std::runtime_error);
    std::remove(gz.c_str());
}
#else
// Without zlib, opening a .gz path must fail with the gzip-gate error (pointing
// at MTL5_WITH_ZLIB), not a generic "cannot open file" -- so use a real .gz file.
TEST_CASE("Matrix Market: gzip without zlib throws (#125)", "[io][matrix_market][gzip]") {
    auto gz = temp_file("no_zlib") + ".gz";
    { std::ofstream out(gz); out << "not actually compressed\n"; }

    bool saw_zlib_error = false;
    try {
        (void)io::mm_read(gz);
    } catch (const std::runtime_error& e) {
        saw_zlib_error = std::string(e.what()).find("MTL5_WITH_ZLIB") != std::string::npos;
    }
    std::remove(gz.c_str());
    REQUIRE(saw_zlib_error);
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
    SECTION("skew-symmetric: a diagonal entry is rejected (#843)") {
        // Skew-symmetric files are read since #843; their diagonal is zero
        // and not stored, so an entry on it is malformed.
        auto fn = write("skew", "%%MatrixMarket matrix coordinate real skew-symmetric\n2 2 1\n2 2 1.0\n");
        REQUIRE_THROWS_AS(io::mm_read(fn), std::runtime_error);
        REQUIRE_THROWS_AS(io::mm_read_dense(fn), std::runtime_error);
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

// ---------------------------------------------------------------------------
// #843: header, banner and entry validation, the symmetric array formats,
// complex values, and the write-then-read round trip.

namespace {

/// A .mtx file in the temp directory, removed when the test ends.
struct temp_mtx {
    std::string path;
    temp_mtx(const std::string& name, const std::string& body) : path(temp_file(name)) {
        std::ofstream out(path, std::ios::binary);   // binary: "\r\n" stays as written
        out << body;
    }
    ~temp_mtx() { std::remove(path.c_str()); }
    temp_mtx(const temp_mtx&) = delete;
    temp_mtx& operator=(const temp_mtx&) = delete;
};

/// Both readers must throw std::runtime_error for this file.
void require_both_throw(const std::string& name, const std::string& body) {
    INFO(name << ":\n" << body);
    temp_mtx f(name, body);
    REQUIRE_THROWS_AS(io::mm_read(f.path), std::runtime_error);
    REQUIRE_THROWS_AS(io::mm_read_dense(f.path), std::runtime_error);
}

const std::string coord = "%%MatrixMarket matrix coordinate real general\n";
const std::string array = "%%MatrixMarket matrix array real general\n";

} // namespace

TEST_CASE("Matrix Market: malformed sizes throw in both readers (#843)", "[io][matrix_market]") {
    // A negative dimension parsed into size_t as 2^64-1; nrows+1 wrapped to 0
    // and mm_read wrote out of bounds (SIGSEGV), mm_read_dense threw bad_alloc.
    require_both_throw("neg_rows", coord + "-1 2 0\n");
    require_both_throw("neg_cols", coord + "2 -1 0\n");
    require_both_throw("neg_nnz", coord + "2 2 -1\n");
    require_both_throw("neg_array", array + "-1 2\n");
    // A header nnz the matrix cannot hold used to reserve it: bad_alloc.
    require_both_throw("huge_nnz", coord + "2 2 99999999999999\n1 1 1.0\n");
    require_both_throw("nnz_over_sym",
        "%%MatrixMarket matrix coordinate real symmetric\n2 2 4\n1 1 1\n2 1 1\n2 2 1\n2 1 1\n");
    // nrows * ncols overflows size_t.
    require_both_throw("overflow", array + "4294967296 4294967297\n");
    require_both_throw("overflow_coord", coord + "4294967296 4294967297 1\n1 1 1.0\n");
    // A size of 2^64-1 with no columns holds no entries, so only the size
    // limit stops nrows + 1 from wrapping to 0.
    require_both_throw("max_rows", coord + "18446744073709551615 0 0\n");
    // A size above 2^64.
    require_both_throw("too_big", coord + "99999999999999999999999 1 0\n");
    // Malformed dimension lines.
    require_both_throw("dim_text", coord + "two 2 1\n1 1 1.0\n");
    require_both_throw("dim_short", coord + "2 2\n");
    require_both_throw("dim_long", coord + "2 2 1 7\n1 1 1.0\n");
    require_both_throw("dim_frac", coord + "2.5 2 1\n1 1 1.0\n");
    require_both_throw("array_dim_long", array + "2 2 4\n1\n2\n3\n4\n");
    require_both_throw("empty", "");
    require_both_throw("banner_only", coord);
}

TEST_CASE("Matrix Market: the banner is checked in full (#843)", "[io][matrix_market]") {
    require_both_throw("tag", "foo matrix coordinate real general\n1 1 1\n1 1 1.0\n");
    require_both_throw("object", "%%MatrixMarket vector coordinate real general\n1 1 1\n1 1 1.0\n");
    require_both_throw("no_symmetry", "%%MatrixMarket matrix coordinate real\n1 1 1\n1 1 1.0\n");
    require_both_throw("no_field", "%%MatrixMarket matrix coordinate\n1 1 1\n1 1 1.0\n");
    require_both_throw("extra", "%%MatrixMarket matrix coordinate real general x\n1 1 1\n1 1 1.0\n");
    require_both_throw("format", "%%MatrixMarket matrix sparse real general\n1 1 1\n1 1 1.0\n");
    require_both_throw("field", "%%MatrixMarket matrix coordinate double general\n1 1 1\n1 1 1.0\n");
    require_both_throw("symmetry", "%%MatrixMarket matrix coordinate real lower\n1 1 1\n1 1 1.0\n");
    require_both_throw("pattern", "%%MatrixMarket matrix coordinate pattern general\n1 1 1\n1 1\n");
    require_both_throw("hermitian_real",
        "%%MatrixMarket matrix coordinate real hermitian\n1 1 1\n1 1 1.0\n");
    // Case does not matter.
    temp_mtx f("banner_case", "%%matrixmarket MATRIX Coordinate REAL General\n1 1 1\n1 1 2.5\n");
    REQUIRE(io::mm_read(f.path)(0, 0) == 2.5);
}

TEST_CASE("Matrix Market: malformed and miscounted entries throw (#843)", "[io][matrix_market]") {
    require_both_throw("entry_text", coord + "2 2 1\n1 x 1.0\n");
    require_both_throw("entry_value", coord + "2 2 1\n1 1 abc\n");
    require_both_throw("entry_short", coord + "2 2 1\n1 1\n");
    require_both_throw("entry_long", coord + "2 2 1\n1 1 1.0 2.0\n");
    require_both_throw("entry_neg", coord + "2 2 1\n-1 1 1.0\n");
    require_both_throw("entry_trailing", coord + "2 2 1\n1 1 1.0x\n");
    require_both_throw("truncated", coord + "2 2 2\n1 1 1.0\n");
    require_both_throw("array_truncated", array + "2 2\n1\n2\n3\n");
    require_both_throw("array_value", array + "1 2\n1\nx\n");
    require_both_throw("array_two", array + "1 1\n1 2\n");
    // Data beyond the header's count used to be ignored.
    require_both_throw("extra_entry", coord + "2 2 1\n1 1 1.0\n2 2 2.0\n");
    require_both_throw("extra_value", array + "1 1\n1\n2\n");
    // A real field read into an integer value type must be exact.
    temp_mtx f("int_from_real", coord + "1 1 1\n1 1 1.5\n");
    REQUIRE_THROWS_AS(io::mm_read<int>(f.path), std::runtime_error);
}

TEST_CASE("Matrix Market: blank lines, CRLF and edge sizes (#843)", "[io][matrix_market]") {
    SECTION("whitespace-only and blank lines are skipped, trailing comments allowed") {
        // A whitespace-only line used to throw "malformed".
        temp_mtx f("ws_lines", coord + "% c\n\n2 2 2\n  \n1 1 1.0\n\t\n2 2 2.0\n\n% end\n");
        auto A = io::mm_read(f.path);
        REQUIRE((A(0, 0) == 1.0 && A(1, 1) == 2.0 && A.nnz() == 2));
        auto D = io::mm_read_dense(f.path);
        REQUIRE((D(0, 0) == 1.0 && D(1, 1) == 2.0));
    }
    SECTION("CRLF line endings") {
        temp_mtx f("crlf", "%%MatrixMarket matrix coordinate real general\r\n% c\r\n"
                           "2 2 2\r\n1 1 1.5\r\n2 1 -2\r\n");
        auto A = io::mm_read(f.path);
        REQUIRE((A(0, 0) == 1.5 && A(1, 0) == -2.0 && A.nnz() == 2));
        temp_mtx g("crlf_array", "%%MatrixMarket matrix array real general\r\n1 2\r\n3\r\n4\r\n");
        auto D = io::mm_read_dense(g.path);
        REQUIRE((D(0, 0) == 3.0 && D(0, 1) == 4.0));
    }
    SECTION("no trailing newline") {
        temp_mtx f("no_nl", coord + "1 1 1\n1 1 7.25");
        REQUIRE(io::mm_read(f.path)(0, 0) == 7.25);
        REQUIRE(io::mm_read_dense(f.path)(0, 0) == 7.25);
    }
    SECTION("0x0, 0x3, 1x1, non-square, an empty row") {
        temp_mtx z("zero", coord + "0 0 0\n");
        REQUIRE(io::mm_read(z.path).num_rows() == 0);
        REQUIRE(io::mm_read_dense(z.path).num_cols() == 0);
        temp_mtx z3("zero3", array + "0 3\n");
        auto Z3 = io::mm_read(z3.path);
        REQUIRE((Z3.num_rows() == 0 && Z3.num_cols() == 3));
        temp_mtx one("one", array + "1 1\n-4\n");
        REQUIRE(io::mm_read(one.path)(0, 0) == -4.0);
        temp_mtx rect("rect", coord + "2 3 2\n1 3 5\n2 1 6\n");
        auto R = io::mm_read(rect.path);
        REQUIRE((R.num_rows() == 2 && R.num_cols() == 3 && R(0, 2) == 5.0 && R(1, 0) == 6.0));
        temp_mtx empty_row("empty_row", coord + "3 2 2\n1 1 1\n3 2 2\n");
        auto E = io::mm_read(empty_row.path);
        REQUIRE(E.ref_major()[1] == E.ref_major()[2]);   // row 1 has no entries
        REQUIRE(E(2, 1) == 2.0);
    }
}

TEST_CASE("Matrix Market: the array branch of mm_read", "[io][matrix_market]") {
    // 2x3 column-major: [[1 0 5] [0 4 6]]; zeros are not stored.
    temp_mtx f("array_sparse", array + "2 3\n1\n0\n0\n4\n5\n6\n");
    auto A = io::mm_read(f.path);
    REQUIRE((A.num_rows() == 2 && A.num_cols() == 3 && A.nnz() == 4));
    REQUIRE((A(0, 0) == 1.0 && A(1, 1) == 4.0 && A(0, 2) == 5.0 && A(1, 2) == 6.0));
    REQUIRE((A(1, 0) == 0.0 && A(0, 1) == 0.0));
}

TEST_CASE("Matrix Market: array symmetric, skew-symmetric and hermitian (#843)", "[io][matrix_market]") {
    SECTION("symmetric: the lower triangle by columns, mirrored") {
        // [[1 2 3] [2 4 5] [3 5 6]] stored as 1 2 3 | 4 5 | 6. A valid file
        // like this used to throw "unexpected EOF in array data".
        temp_mtx f("arr_sym", "%%MatrixMarket matrix array real symmetric\n3 3\n1\n2\n3\n4\n5\n6\n");
        const double want[3][3] = {{1, 2, 3}, {2, 4, 5}, {3, 5, 6}};
        auto D = io::mm_read_dense(f.path);
        auto S = io::mm_read(f.path);
        for (std::size_t i = 0; i < 3; ++i)
            for (std::size_t j = 0; j < 3; ++j) {
                REQUIRE(D(i, j) == want[i][j]);
                REQUIRE(S(i, j) == want[i][j]);
            }
        // An n*n-value file under a symmetric banner has too many values; it
        // used to be read as general, unmirrored.
        require_both_throw("arr_sym_full",
            "%%MatrixMarket matrix array real symmetric\n2 2\n1\n2\n2\n3\n");
        require_both_throw("arr_sym_rect", "%%MatrixMarket matrix array real symmetric\n2 3\n1\n2\n3\n");
    }
    SECTION("skew-symmetric: the strict lower triangle, mirrored negated") {
        // [[0 -2 -3] [2 0 -5] [3 5 0]] stored as 2 3 | 5.
        temp_mtx f("arr_skew", "%%MatrixMarket matrix array real skew-symmetric\n3 3\n2\n3\n5\n");
        const double want[3][3] = {{0, -2, -3}, {2, 0, -5}, {3, 5, 0}};
        auto D = io::mm_read_dense(f.path);
        auto S = io::mm_read(f.path);
        REQUIRE(S.nnz() == 6);
        for (std::size_t i = 0; i < 3; ++i)
            for (std::size_t j = 0; j < 3; ++j) {
                REQUIRE(D(i, j) == want[i][j]);
                REQUIRE(S(i, j) == want[i][j]);
            }
        temp_mtx c("coord_skew", "%%MatrixMarket matrix coordinate real skew-symmetric\n2 2 1\n2 1 4.0\n");
        auto C = io::mm_read(c.path);
        REQUIRE((C(1, 0) == 4.0 && C(0, 1) == -4.0 && C(0, 0) == 0.0));
    }
    SECTION("hermitian: mirrored conjugated") {
        using cd = std::complex<double>;
        temp_mtx f("arr_herm", "%%MatrixMarket matrix array complex hermitian\n2 2\n1 0\n2 3\n4 0\n");
        auto D = io::mm_read_dense<cd>(f.path);
        REQUIRE((D(0, 0) == cd(1, 0) && D(1, 0) == cd(2, 3) && D(0, 1) == cd(2, -3) && D(1, 1) == cd(4, 0)));
        temp_mtx c("coord_herm", "%%MatrixMarket matrix coordinate complex hermitian\n2 2 2\n1 1 1 0\n2 1 2 3\n");
        auto S = io::mm_read<cd>(c.path);
        REQUIRE((S(1, 0) == cd(2, 3) && S(0, 1) == cd(2, -3)));
    }
}

TEST_CASE("Matrix Market: nan and inf round trip (#843)", "[io][matrix_market]") {
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    // The writers emit these, and the reader used to reject them as malformed.
    mat::dense2D<double> A(2, 2);
    A(0, 0) = nan; A(0, 1) = inf; A(1, 0) = -inf; A(1, 1) = -0.5;
    auto fname = temp_file("nan_dense");
    io::mm_write(fname, A);
    auto B = io::mm_read_dense(fname);
    REQUIRE(std::isnan(B(0, 0)));
    REQUIRE(B(0, 1) == inf);
    REQUIRE(B(1, 0) == -inf);
    REQUIRE(B(1, 1) == -0.5);
    std::remove(fname.c_str());

    mat::compressed2D<double> S(2, 2);
    { mat::inserter<mat::compressed2D<double>> ins(S); ins[0][1] << inf; ins[1][0] << nan; }
    auto sname = temp_file("nan_sparse");
    io::mm_write_sparse(sname, S);
    auto T = io::mm_read(sname);
    REQUIRE(T.nnz() == 2);
    REQUIRE(T(0, 1) == inf);
    REQUIRE(std::isnan(T(1, 0)));
    std::remove(sname.c_str());
}

TEST_CASE("Matrix Market: the banner field follows the value type (#843)", "[io][matrix_market]") {
    auto first_line = [](const std::string& fn) {
        std::ifstream in(fn); std::string l; std::getline(in, l); return l;
    };
    SECTION("complex round trip, dense and sparse") {
        using cd = std::complex<double>;
        mat::dense2D<cd> A(2, 1);
        A(0, 0) = cd(1.5, -2.0); A(1, 0) = cd(0.0, 0.25);
        auto fname = temp_file("cplx_dense");
        io::mm_write(fname, A, "complex");
        REQUIRE(first_line(fname) == "%%MatrixMarket matrix array complex general");
        auto B = io::mm_read_dense<cd>(fname);
        REQUIRE((B(0, 0) == A(0, 0) && B(1, 0) == A(1, 0)));
        REQUIRE_THROWS_AS(io::mm_read_dense<double>(fname), std::runtime_error);   // needs complex
        std::remove(fname.c_str());

        mat::compressed2D<cd> S(2, 2);
        { mat::inserter<mat::compressed2D<cd>> ins(S); ins[1][0] << cd(3.0, 4.0); }
        auto sname = temp_file("cplx_sparse");
        io::mm_write_sparse(sname, S);
        REQUIRE(first_line(sname) == "%%MatrixMarket matrix coordinate complex general");
        auto T = io::mm_read<cd>(sname);
        REQUIRE((T.nnz() == 1 && T(1, 0) == cd(3.0, 4.0)));
        std::remove(sname.c_str());
    }
    SECTION("integer and real") {
        mat::dense2D<int> I(1, 2);
        I(0, 0) = -7; I(0, 1) = 12;
        auto fname = temp_file("int_dense");
        io::mm_write(fname, I);
        REQUIRE(first_line(fname) == "%%MatrixMarket matrix array integer general");
        auto J = io::mm_read_dense<int>(fname);
        REQUIRE((J(0, 0) == -7 && J(0, 1) == 12));
        std::remove(fname.c_str());

        mat::dense2D<float> F(1, 1);
        F(0, 0) = 0.1f;                          // needs 9 digits to read back exactly
        auto ffn = temp_file("float_dense");
        io::mm_write(ffn, F);
        REQUIRE(first_line(ffn) == "%%MatrixMarket matrix array real general");
        REQUIRE(io::mm_read_dense<float>(ffn)(0, 0) == 0.1f);
        std::remove(ffn.c_str());
    }
    SECTION("a real file reads into complex values") {
        temp_mtx f("real_to_cplx", coord + "1 1 1\n1 1 2.5\n");
        REQUIRE(io::mm_read<std::complex<double>>(f.path)(0, 0) == std::complex<double>(2.5, 0.0));
    }
}

TEST_CASE("Matrix Market: mm_write_sparse with a comment, mm_write of a sparse matrix", "[io][matrix_market]") {
    mat::compressed2D<double> A(2, 3);
    { mat::inserter<mat::compressed2D<double>> ins(A); ins[0][2] << 1.25; ins[1][0] << -3.0; }

    auto sname = temp_file("sparse_comment");
    io::mm_write_sparse(sname, A, "a comment");
    {
        std::ifstream in(sname);
        std::string banner, comment;
        std::getline(in, banner);
        std::getline(in, comment);
        REQUIRE(comment == "% a comment");
    }
    auto B = io::mm_read(sname);
    REQUIRE((B.nnz() == 2 && B(0, 2) == 1.25 && B(1, 0) == -3.0));
    std::remove(sname.c_str());

    // mm_write takes any matrix with (i, j) access and writes array format.
    auto dname = temp_file("sparse_as_array");
    io::mm_write(dname, A);
    auto D = io::mm_read_dense(dname);
    REQUIRE((D.num_rows() == 2 && D.num_cols() == 3));
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 3; ++j) REQUIRE(D(i, j) == A(i, j));
    std::remove(dname.c_str());
}

TEST_CASE("Matrix Market: a narrow size_type, a short array file, char-sized integers (#843)",
          "[io][matrix_market]") {
    SECTION("a 32-bit size_type reads a coordinate file whose m*n exceeds it") {
        using P32 = mat::parameters<tag::row_major, mtl::detail::c_index, mat::non_fixed::dimensions,
                                    tag::on_heap, unsigned>;
        temp_mtx f("narrow", coord + "100000 100000 2\n1 1 1.5\n100000 99999 -2\n");
        auto A = io::mm_read<double, P32>(f.path);
        REQUIRE((A.num_rows() == 100000u && A.nnz() == 2u));
        REQUIRE(A(99999u, 99998u) == -2.0);
        // An array file stores m*n values, which must fit the index type.
        temp_mtx g("narrow_array", array + "100000 100000\n1\n");
        REQUIRE_THROWS_AS((io::mm_read<double, P32>(g.path)), std::runtime_error);
    }
    SECTION("an array header the file cannot back throws before allocating") {
        // mm_read_dense used to allocate and zero 80 GB before noticing.
        require_both_throw("short_array", array + "100000 100000\n");
    }
    SECTION("int8_t and uint8_t are written as numbers") {
        mat::dense2D<std::int8_t> A(1, 3);
        A(0, 0) = 5; A(0, 1) = -7; A(0, 2) = 100;
        auto fname = temp_file("int8");
        io::mm_write(fname, A);
        {
            std::ifstream in(fname);
            std::string l;
            std::getline(in, l); std::getline(in, l); std::getline(in, l);
            REQUIRE(l == "5");                 // operator<< used to write the character 0x05
        }
        auto B = io::mm_read_dense<std::int8_t>(fname);
        REQUIRE((B(0, 0) == 5 && B(0, 1) == -7 && B(0, 2) == 100));
        std::remove(fname.c_str());

        mat::compressed2D<std::uint8_t> S(2, 2);
        { mat::inserter<mat::compressed2D<std::uint8_t>> ins(S); ins[1][0] << std::uint8_t(200); }
        auto sname = temp_file("uint8");
        io::mm_write_sparse(sname, S);
        auto T = io::mm_read<std::uint8_t>(sname);
        REQUIRE((T.nnz() == 1u && T(1, 0) == 200));
        std::remove(sname.c_str());
    }
}

#ifdef __linux__
TEST_CASE("Matrix Market: a failed write throws, even for a small matrix (#843)", "[io][matrix_market]") {
    // /dev/full accepts the open and fails every write. A small matrix sits in
    // the stream buffer until the flush, so checking the stream before closing
    // it saw no error.
    if (!std::filesystem::exists("/dev/full")) SKIP("no /dev/full");
    mat::dense2D<double> A(1, 1);
    A(0, 0) = 1.0;
    REQUIRE_THROWS_AS(io::mm_write("/dev/full", A), std::runtime_error);
    mat::compressed2D<double> S(1, 1);
    { mat::inserter<mat::compressed2D<double>> ins(S); ins[0][0] << 1.0; }
    REQUIRE_THROWS_AS(io::mm_write_sparse("/dev/full", S), std::runtime_error);
}
#endif
