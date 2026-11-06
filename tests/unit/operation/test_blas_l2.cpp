// Tests for the BLAS Level-2 operators: ger, symv, trmv, trsv (#229).
// Exercise the generic (non-column-major / non-BLAS) path with row-major
// dense2D; a LAPACK/BLAS build additionally routes column-major inputs through
// the external BLAS, validated by the same reference checks.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/parameter.hpp>
#include <mtl/tag/orientation.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/ger.hpp>
#include <mtl/operation/mult.hpp>
#include <mtl/operation/symv.hpp>
#include <mtl/operation/trmv.hpp>
#include <mtl/operation/trsv.hpp>

#include <cmath>
#include <cstddef>
#include <utility>

using namespace mtl;
using colmat = mat::dense2D<double, mat::parameters<tag::col_major>>;

TEST_CASE("ger: rank-1 update A += alpha x y^T", "[operation][blas][ger]") {
    mat::dense2D<double> A(2, 3);
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 3; ++j) A(i, j) = 1.0;
    vec::dense_vector<double> x = {2.0, 3.0};
    vec::dense_vector<double> y = {1.0, 0.5, -1.0};

    ger(2.0, x, y, A);   // A(i,j) += 2*x(i)*y(j)
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            REQUIRE_THAT(A(i, j), Catch::Matchers::WithinAbs(1.0 + 2.0 * x(i) * y(j), 1e-12));
}

TEST_CASE("ger: column-major (BLAS-eligible) path matches reference", "[operation][blas][ger]") {
    colmat A(3, 3);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j) A(i, j) = static_cast<double>(i + j);
    vec::dense_vector<double> x = {1.0, -2.0, 0.5};
    vec::dense_vector<double> y = {2.0, 1.0, -1.0};

    ger(1.5, x, y, A);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            REQUIRE_THAT(A(i, j),
                Catch::Matchers::WithinAbs(static_cast<double>(i + j) + 1.5 * x(i) * y(j), 1e-12));
}

TEST_CASE("symv: symmetric matrix-vector y = alpha A x + beta y", "[operation][blas][symv]") {
    // A = [[2,1,0],[1,3,1],[0,1,2]] symmetric.
    mat::dense2D<double> A(3, 3);
    double v[3][3] = {{2,1,0},{1,3,1},{0,1,2}};
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j) A(i, j) = v[i][j];
    vec::dense_vector<double> x = {1.0, 2.0, 3.0};
    vec::dense_vector<double> y = {1.0, 1.0, 1.0};

    symv(2.0, A, x, 0.5, y);   // y = 2*A*x + 0.5*y
    // A*x = [4, 10, 8]; 2*A*x + 0.5*[1,1,1] = [8.5, 20.5, 16.5]
    REQUIRE_THAT(y(0), Catch::Matchers::WithinAbs(8.5, 1e-12));
    REQUIRE_THAT(y(1), Catch::Matchers::WithinAbs(20.5, 1e-12));
    REQUIRE_THAT(y(2), Catch::Matchers::WithinAbs(16.5, 1e-12));
}

TEST_CASE("symv: column-major path matches reference", "[operation][blas][symv]") {
    colmat A(3, 3);
    double v[3][3] = {{4,1,2},{1,5,3},{2,3,6}};   // symmetric
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j) A(i, j) = v[i][j];
    vec::dense_vector<double> x = {1.0, -1.0, 2.0};
    vec::dense_vector<double> y = {0.0, 0.0, 0.0};

    symv(1.0, A, x, 0.0, y);   // y = A*x
    // A*x = [4-1+4, 1-5+6, 2-3+12] = [7, 2, 11]
    REQUIRE_THAT(y(0), Catch::Matchers::WithinAbs(7.0, 1e-12));
    REQUIRE_THAT(y(1), Catch::Matchers::WithinAbs(2.0, 1e-12));
    REQUIRE_THAT(y(2), Catch::Matchers::WithinAbs(11.0, 1e-12));
}

TEST_CASE("trmv: triangular matrix-vector x = A x", "[operation][blas][trmv]") {
    // Upper triangular U = [[2,1,3],[0,4,1],[0,0,5]].
    mat::dense2D<double> U(3, 3);
    double u[3][3] = {{2,1,3},{0,4,1},{0,0,5}};
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j) U(i, j) = u[i][j];
    vec::dense_vector<double> x = {1.0, 2.0, 3.0};
    trmv(U, x, /*upper=*/true);
    // U*x = [2+2+9, 8+3, 15] = [13, 11, 15]
    REQUIRE_THAT(x(0), Catch::Matchers::WithinAbs(13.0, 1e-12));
    REQUIRE_THAT(x(1), Catch::Matchers::WithinAbs(11.0, 1e-12));
    REQUIRE_THAT(x(2), Catch::Matchers::WithinAbs(15.0, 1e-12));

    // Lower triangular L = [[2,0,0],[1,3,0],[4,1,5]].
    mat::dense2D<double> L(3, 3);
    double l[3][3] = {{2,0,0},{1,3,0},{4,1,5}};
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j) L(i, j) = l[i][j];
    vec::dense_vector<double> z = {1.0, 2.0, 3.0};
    trmv(L, z, /*upper=*/false);
    // L*z = [2, 1+6, 4+2+15] = [2, 7, 21]
    REQUIRE_THAT(z(0), Catch::Matchers::WithinAbs(2.0, 1e-12));
    REQUIRE_THAT(z(1), Catch::Matchers::WithinAbs(7.0, 1e-12));
    REQUIRE_THAT(z(2), Catch::Matchers::WithinAbs(21.0, 1e-12));
}

TEST_CASE("trsv: triangular solve A x = b inverts trmv", "[operation][blas][trsv]") {
    // Upper triangular; solve U y = (U x) recovers x.
    colmat U(4, 4);
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            U(i, j) = (j >= i) ? static_cast<double>(i + j + 1) : 0.0;
    vec::dense_vector<double> x = {1.0, -2.0, 3.0, 0.5};

    vec::dense_vector<double> b(4);
    for (std::size_t i = 0; i < 4; ++i) b(i) = x(i);
    trmv(U, b, /*upper=*/true);          // b = U x

    vec::dense_vector<double> sol(4);
    trsv(U, sol, b, /*upper=*/true);     // solve U sol = b
    for (std::size_t i = 0; i < 4; ++i)
        REQUIRE_THAT(sol(i), Catch::Matchers::WithinAbs(x(i), 1e-10));

    // Lower triangular, in-place form.
    mat::dense2D<double> Lm(3, 3);
    double l[3][3] = {{2,0,0},{1,3,0},{4,1,5}};
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j) Lm(i, j) = l[i][j];
    vec::dense_vector<double> rhs = {2.0, 7.0, 21.0};   // = L * [1,2,3]
    trsv(Lm, rhs, /*upper=*/false);
    REQUIRE_THAT(rhs(0), Catch::Matchers::WithinAbs(1.0, 1e-12));
    REQUIRE_THAT(rhs(1), Catch::Matchers::WithinAbs(2.0, 1e-12));
    REQUIRE_THAT(rhs(2), Catch::Matchers::WithinAbs(3.0, 1e-12));
}

// Empty shapes (#693). BLAS requires lda >= max(1, rows) even when a dimension
// is 0, and returns without touching y when either gemv dimension is 0. The
// generic path makes y = A*x over an empty inner dimension 0, so every path
// must too. Row-major runs the generic path, or BLAS gemv('T') for mult;
// column-major runs BLAS in a BLAS build.
namespace {

template <typename M>
void check_empty_l2(std::size_t m, std::size_t n) {
    CAPTURE(m, n);
    // mult (gemv): y = A*x, y prefilled with 7.
    {
        M A(m, n);
        vec::dense_vector<double> x(n), y(m);
        for (std::size_t j = 0; j < n; ++j) x(j) = 1.0;
        for (std::size_t i = 0; i < m; ++i) y(i) = 7.0;
        mult(A, x, y);
        for (std::size_t i = 0; i < m; ++i) REQUIRE(y(i) == 0.0);
    }
    // ger: A += alpha x y^T has no elements to update; it must not fail.
    {
        M A(m, n);
        vec::dense_vector<double> x(m), y(n);
        for (std::size_t i = 0; i < m; ++i) x(i) = 1.0;
        for (std::size_t j = 0; j < n; ++j) y(j) = 1.0;
        ger(2.0, x, y, A);
        REQUIRE(A.num_rows() == m);
        REQUIRE(A.num_cols() == n);
    }
}

template <typename M>
void check_empty_square() {
    M A(0, 0);
    vec::dense_vector<double> x(0), y(0);
    symv(1.0, A, x, 0.0, y);
    REQUIRE(y.size() == 0);
    for (bool upper : {true, false})
        for (bool unit : {true, false}) {
            trmv(A, x, upper, unit);
            trsv(A, x, upper, unit);
            REQUIRE(x.size() == 0);
        }
}

} // namespace

TEST_CASE("Level 2 on empty shapes: mult (gemv) and ger", "[operation][blas][693]") {
    for (auto [m, n] : {std::pair<std::size_t, std::size_t>{3, 0}, {0, 3}, {0, 0}}) {
        check_empty_l2<mat::dense2D<double>>(m, n);
        check_empty_l2<colmat>(m, n);
    }
}

TEST_CASE("Level 2 on empty shapes: symv, trmv, trsv", "[operation][blas][693]") {
    check_empty_square<mat::dense2D<double>>();
    check_empty_square<colmat>();
}
