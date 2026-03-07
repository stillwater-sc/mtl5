#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/operation/diagonal.hpp>
#include <mtl/operation/print.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/vec/dense_vector.hpp>

#include <sstream>
#include <string>

using namespace mtl;
using Catch::Matchers::WithinAbs;

// ── diag() tests ──────────────────────────────────────────────────────────

TEST_CASE("diag: creates diagonal matrix from vector", "[operation][diag]") {
    vec::dense_vector<double> v({3.0, 5.0, 7.0});
    auto D = diag(v);

    REQUIRE(D.num_rows() == 3);
    REQUIRE(D.num_cols() == 3);
    REQUIRE(D.nnz() == 3);

    REQUIRE_THAT(D(0, 0), WithinAbs(3.0, 1e-15));
    REQUIRE_THAT(D(1, 1), WithinAbs(5.0, 1e-15));
    REQUIRE_THAT(D(2, 2), WithinAbs(7.0, 1e-15));
    REQUIRE_THAT(D(0, 1), WithinAbs(0.0, 1e-15));
    REQUIRE_THAT(D(1, 0), WithinAbs(0.0, 1e-15));
}

TEST_CASE("diag/diagonal roundtrip", "[operation][diag]") {
    // Build a matrix with known diagonal
    mat::dense2D<double> A({{4.0, 1.0, 0.0},
                             {1.0, 5.0, 2.0},
                             {0.0, 2.0, 6.0}});
    auto d = diagonal(A);     // extract diagonal
    auto D = diag(d);         // rebuild diagonal matrix

    REQUIRE(D.num_rows() == 3);
    REQUIRE(D.nnz() == 3);
    REQUIRE_THAT(D(0, 0), WithinAbs(4.0, 1e-15));
    REQUIRE_THAT(D(1, 1), WithinAbs(5.0, 1e-15));
    REQUIRE_THAT(D(2, 2), WithinAbs(6.0, 1e-15));
}

// ── print() tests ─────────────────────────────────────────────────────────

TEST_CASE("print vector with precision", "[operation][print]") {
    vec::dense_vector<double> v({1.23456789, 2.34567890});
    std::ostringstream os;
    print(os, v, 3);
    std::string out = os.str();
    // Should contain truncated values (3 significant digits after decimal)
    REQUIRE(out.find('[') != std::string::npos);
    REQUIRE(out.find(']') != std::string::npos);
    // Precision 3 should NOT show all 8 digits
    REQUIRE(out.find("1.23456789") == std::string::npos);
}

TEST_CASE("print matrix with precision", "[operation][print]") {
    mat::dense2D<double> A({{1.111111, 2.222222},
                             {3.333333, 4.444444}});
    std::ostringstream os;
    print(os, A, 2);
    std::string out = os.str();
    REQUIRE(out.find('[') != std::string::npos);
    // Precision 2 should not show full digits
    REQUIRE(out.find("1.111111") == std::string::npos);
}

// ── print_sparse() tests ─────────────────────────────────────────────────

TEST_CASE("print_sparse outputs triplet format", "[operation][print]") {
    mat::compressed2D<double> A(3, 3);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 1.0;
        ins[1][2] << 2.0;
        ins[2][1] << 3.0;
    }

    std::ostringstream os;
    print_sparse(os, A);
    std::string out = os.str();

    REQUIRE(out.find("(0, 0) = 1") != std::string::npos);
    REQUIRE(out.find("(1, 2) = 2") != std::string::npos);
    REQUIRE(out.find("(2, 1) = 3") != std::string::npos);
}

// ── print_matlab() tests ─────────────────────────────────────────────────

TEST_CASE("print_matlab produces MATLAB format", "[operation][print]") {
    mat::dense2D<double> A({{1.0, 2.0}, {3.0, 4.0}});
    std::ostringstream os;
    print_matlab(os, A, "M");
    std::string out = os.str();

    REQUIRE(out.find("M = [") != std::string::npos);
    REQUIRE(out.find("];") != std::string::npos);
    REQUIRE(out.find("; ") != std::string::npos);
}

TEST_CASE("print_matlab custom name and precision", "[operation][print]") {
    mat::dense2D<double> A({{3.14159265358979}});
    std::ostringstream os;
    print_matlab(os, A, "pi_mat", 4);
    std::string out = os.str();

    REQUIRE(out.find("pi_mat = [") != std::string::npos);
    // Precision 4 should not show all digits
    REQUIRE(out.find("3.14159265358979") == std::string::npos);
}
