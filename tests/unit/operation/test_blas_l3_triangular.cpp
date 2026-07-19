// Tests for the BLAS Level-3 triangular operators: trmm, trsm (#229, batch 2).
// Exercises the generic (row-major) path and the column-major (BLAS-dispatch)
// path with the same reference checks; trsm inverts trmm as a round-trip.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/parameter.hpp>
#include <mtl/tag/orientation.hpp>
#include <mtl/operation/trmm.hpp>
#include <mtl/operation/trsm.hpp>

#include <cmath>

using namespace mtl;
template <typename Params = mat::parameters<>>
using dmat = mat::dense2D<double, Params>;
using colmat = mat::dense2D<double, mat::parameters<tag::col_major>>;

namespace {

template <typename MA, typename MB>
double max_abs_diff(const MA& A, const MB& B, std::size_t m, std::size_t n) {
    double d = 0.0;
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j)
            d = std::max(d, std::abs(A(i, j) - B(i, j)));
    return d;
}

} // namespace

TEST_CASE("trmm: B = alpha * A * B, upper triangular", "[operation][blas][trmm]") {
    // U = [[2,1,3],[0,4,1],[0,0,5]], B = 3x2.
    dmat<> U(3, 3);
    double u[3][3] = {{2,1,3},{0,4,1},{0,0,5}};
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j) U(i, j) = u[i][j];
    dmat<> B(3, 2);
    double b[3][2] = {{1,0},{2,1},{3,2}};
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 2; ++j) B(i, j) = b[i][j];

    trmm(1.0, U, B, /*upper=*/true);
    // col0: U*[1,2,3] = [13,11,15]; col1: U*[0,1,2] = [7,6,10]
    double exp[3][2] = {{13,7},{11,6},{15,10}};
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 2; ++j)
            REQUIRE_THAT(B(i, j), Catch::Matchers::WithinAbs(exp[i][j], 1e-12));
}

TEST_CASE("trmm: lower triangular with alpha (column-major/BLAS path)", "[operation][blas][trmm]") {
    colmat L(3, 3);
    double l[3][3] = {{2,0,0},{1,3,0},{4,1,5}};
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j) L(i, j) = l[i][j];
    colmat B(3, 1);
    B(0,0) = 1; B(1,0) = 2; B(2,0) = 3;

    trmm(2.0, L, B, /*upper=*/false);   // 2 * L * [1,2,3]
    // L*[1,2,3] = [2,7,21] -> *2 = [4,14,42]
    REQUIRE_THAT(B(0,0), Catch::Matchers::WithinAbs(4.0, 1e-12));
    REQUIRE_THAT(B(1,0), Catch::Matchers::WithinAbs(14.0, 1e-12));
    REQUIRE_THAT(B(2,0), Catch::Matchers::WithinAbs(42.0, 1e-12));
}

TEST_CASE("trsm: solves A X = alpha B (upper, generic + BLAS)", "[operation][blas][trsm]") {
    // Build a well-conditioned upper-triangular A, a known X, form B = A*X,
    // then trsm must recover X (up to the alpha scaling).
    const std::size_t m = 5, n = 3;
    colmat A(m, m), X(m, n), B(m, n);
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < m; ++j)
            A(i, j) = (j >= i) ? static_cast<double>(i + j + 1) : 0.0;
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j)
            X(i, j) = static_cast<double>((i + 1) * (j + 2)) - 3.0;

    // B = A * X
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            double s = 0.0;
            for (std::size_t k = 0; k < m; ++k) s += A(i, k) * X(k, j);
            B(i, j) = s;
        }

    trsm(1.0, A, B, /*upper=*/true);    // solve A * Xr = B, B := Xr
    REQUIRE(max_abs_diff(B, X, m, n) < 1e-9);
}

TEST_CASE("trsm inverts trmm (lower, round-trip)", "[operation][blas][trsm]") {
    const std::size_t m = 4, n = 2;
    dmat<> A(m, m), B0(m, n), B(m, n);
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < m; ++j)
            A(i, j) = (j <= i) ? static_cast<double>(i + j + 2) : 0.0;   // lower, nonzero diag
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            B0(i, j) = static_cast<double>(i) - 2.0 * static_cast<double>(j) + 1.0;
            B(i, j)  = B0(i, j);
        }

    trmm(1.0, A, B, /*upper=*/false);   // B = A * B0
    trsm(1.0, A, B, /*upper=*/false);   // solve A * X = B  ->  X == B0
    REQUIRE(max_abs_diff(B, B0, m, n) < 1e-10);
}

TEST_CASE("trmm/trsm: empty matrices (BLAS leading-dimension edge)", "[operation][blas][trmm][trsm]") {
    // Column-major 0x0 is BLAS-eligible; the BLAS path must pass lda,ldb >= 1
    // (max(1,m)) rather than 0. Both ops are a no-op and must not crash.
    colmat A0(0, 0), B0(0, 0);
    REQUIRE_NOTHROW(trmm(1.0, A0, B0, /*upper=*/true));
    REQUIRE_NOTHROW(trsm(1.0, A0, B0, /*upper=*/true));
    REQUIRE(B0.num_rows() == 0);
    REQUIRE(B0.num_cols() == 0);
}

TEST_CASE("trmm/trsm: unit diagonal", "[operation][blas][trmm][trsm]") {
    // Unit-diagonal lower triangular (diagonal ignored, treated as 1).
    dmat<> A(3, 3);
    double a[3][3] = {{9,0,0},{2,9,0},{3,4,9}};   // off-diagonals used; diag ignored
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j) A(i, j) = a[i][j];
    dmat<> B(3, 1), B0(3, 1);
    B(0,0) = B0(0,0) = 1; B(1,0) = B0(1,0) = 2; B(2,0) = B0(2,0) = 3;

    trmm(1.0, A, B, /*upper=*/false, /*unit_diag=*/true);
    // unit-diag L*[1,2,3]: [1, 2+2*1, 3+3*1+4*2] = [1, 4, 14]
    REQUIRE_THAT(B(0,0), Catch::Matchers::WithinAbs(1.0, 1e-12));
    REQUIRE_THAT(B(1,0), Catch::Matchers::WithinAbs(4.0, 1e-12));
    REQUIRE_THAT(B(2,0), Catch::Matchers::WithinAbs(14.0, 1e-12));

    trsm(1.0, A, B, /*upper=*/false, /*unit_diag=*/true);   // recover B0
    REQUIRE(max_abs_diff(B, B0, 3, 1) < 1e-12);
}
