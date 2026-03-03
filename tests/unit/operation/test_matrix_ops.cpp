#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/operators.hpp>
#include <mtl/vec/operators.hpp>
#include <mtl/operation/mult.hpp>
#include <mtl/operation/trans.hpp>

using namespace mtl;
using Catch::Matchers::WithinAbs;

// ── Matrix arithmetic ───────────────────────────────────────────────────

TEST_CASE("matrix addition", "[mat][operators]") {
    dense2D<double> a = {{1.0, 2.0}, {3.0, 4.0}};
    dense2D<double> b = {{5.0, 6.0}, {7.0, 8.0}};
    auto c = a + b;
    REQUIRE(c(0, 0) == 6.0);
    REQUIRE(c(0, 1) == 8.0);
    REQUIRE(c(1, 0) == 10.0);
    REQUIRE(c(1, 1) == 12.0);
}

TEST_CASE("matrix subtraction", "[mat][operators]") {
    dense2D<double> a = {{5.0, 6.0}, {7.0, 8.0}};
    dense2D<double> b = {{1.0, 2.0}, {3.0, 4.0}};
    auto c = a - b;
    REQUIRE(c(0, 0) == 4.0);
    REQUIRE(c(0, 1) == 4.0);
    REQUIRE(c(1, 0) == 4.0);
    REQUIRE(c(1, 1) == 4.0);
}

TEST_CASE("matrix unary negation", "[mat][operators]") {
    dense2D<double> a = {{1.0, -2.0}, {3.0, -4.0}};
    auto b = -a;
    REQUIRE(b(0, 0) == -1.0);
    REQUIRE(b(0, 1) == 2.0);
    REQUIRE(b(1, 0) == -3.0);
    REQUIRE(b(1, 1) == 4.0);
}

TEST_CASE("scalar * matrix", "[mat][operators]") {
    dense2D<double> m = {{1.0, 2.0}, {3.0, 4.0}};
    auto r = 2.0 * m;
    REQUIRE(r(0, 0) == 2.0);
    REQUIRE(r(0, 1) == 4.0);
    REQUIRE(r(1, 0) == 6.0);
    REQUIRE(r(1, 1) == 8.0);
}

TEST_CASE("matrix / scalar", "[mat][operators]") {
    dense2D<double> m = {{4.0, 6.0}, {8.0, 10.0}};
    auto r = m / 2.0;
    REQUIRE(r(0, 0) == 2.0);
    REQUIRE(r(0, 1) == 3.0);
    REQUIRE(r(1, 0) == 4.0);
    REQUIRE(r(1, 1) == 5.0);
}

// ── Compound assignment ─────────────────────────────────────────────────

TEST_CASE("matrix +=", "[mat][compound]") {
    dense2D<double> a = {{1.0, 2.0}, {3.0, 4.0}};
    dense2D<double> b = {{10.0, 20.0}, {30.0, 40.0}};
    a += b;
    REQUIRE(a(0, 0) == 11.0);
    REQUIRE(a(1, 1) == 44.0);
}

TEST_CASE("matrix -=", "[mat][compound]") {
    dense2D<double> a = {{10.0, 20.0}, {30.0, 40.0}};
    dense2D<double> b = {{1.0, 2.0}, {3.0, 4.0}};
    a -= b;
    REQUIRE(a(0, 0) == 9.0);
    REQUIRE(a(1, 1) == 36.0);
}

TEST_CASE("matrix *=", "[mat][compound]") {
    dense2D<double> m = {{1.0, 2.0}, {3.0, 4.0}};
    m *= 3.0;
    REQUIRE(m(0, 0) == 3.0);
    REQUIRE(m(1, 1) == 12.0);
}

TEST_CASE("matrix /=", "[mat][compound]") {
    dense2D<double> m = {{10.0, 20.0}, {30.0, 40.0}};
    m /= 10.0;
    REQUIRE(m(0, 0) == 1.0);
    REQUIRE(m(1, 1) == 4.0);
}

// ── Matrix-vector multiply ──────────────────────────────────────────────

TEST_CASE("matrix * vector operator", "[mat][operators][matvec]") {
    dense2D<double> A = {{1.0, 2.0}, {3.0, 4.0}};
    dense_vector<double> x = {1.0, 1.0};
    auto y = A * x;
    REQUIRE(y(0) == 3.0);   // 1+2
    REQUIRE(y(1) == 7.0);   // 3+4
}

TEST_CASE("identity matrix * vector", "[mat][operators][matvec]") {
    dense2D<double> I = {{1.0, 0.0}, {0.0, 1.0}};
    dense_vector<double> x = {5.0, 7.0};
    auto y = I * x;
    REQUIRE(y(0) == 5.0);
    REQUIRE(y(1) == 7.0);
}

// ── Matrix-matrix multiply ──────────────────────────────────────────────

TEST_CASE("matrix * matrix operator", "[mat][operators][matmat]") {
    dense2D<double> A = {{1.0, 2.0}, {3.0, 4.0}};
    dense2D<double> B = {{5.0, 6.0}, {7.0, 8.0}};
    auto C = A * B;
    // C[0,0] = 1*5+2*7 = 19, C[0,1] = 1*6+2*8 = 22
    // C[1,0] = 3*5+4*7 = 43, C[1,1] = 3*6+4*8 = 50
    REQUIRE(C(0, 0) == 19.0);
    REQUIRE(C(0, 1) == 22.0);
    REQUIRE(C(1, 0) == 43.0);
    REQUIRE(C(1, 1) == 50.0);
}

// ── mult() into pre-allocated ───────────────────────────────────────────

TEST_CASE("mult(A, x, y) mat*vec", "[operation][mult]") {
    dense2D<double> A = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    dense_vector<double> x = {1.0, 2.0, 3.0};
    dense_vector<double> y(2);
    mult(A, x, y);
    REQUIRE(y(0) == 14.0);  // 1+4+9
    REQUIRE(y(1) == 32.0);  // 4+10+18
}

TEST_CASE("mult(A, B, C) mat*mat", "[operation][mult]") {
    dense2D<double> A = {{1.0, 2.0}, {3.0, 4.0}};
    dense2D<double> B = {{5.0, 6.0}, {7.0, 8.0}};
    dense2D<double> C(2, 2);
    mult(A, B, C);
    REQUIRE(C(0, 0) == 19.0);
    REQUIRE(C(0, 1) == 22.0);
    REQUIRE(C(1, 0) == 43.0);
    REQUIRE(C(1, 1) == 50.0);
}

// ── Transposed view ─────────────────────────────────────────────────────

TEST_CASE("trans() swaps dimensions", "[operation][trans]") {
    dense2D<double> m = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    auto t = trans(m);
    REQUIRE(t.num_rows() == 3);
    REQUIRE(t.num_cols() == 2);
}

TEST_CASE("trans() swaps element access", "[operation][trans]") {
    dense2D<double> m = {{1.0, 2.0}, {3.0, 4.0}};
    auto t = trans(m);
    REQUIRE(t(0, 0) == 1.0);
    REQUIRE(t(0, 1) == 3.0);
    REQUIRE(t(1, 0) == 2.0);
    REQUIRE(t(1, 1) == 4.0);
}

TEST_CASE("trans() reflects changes in original", "[operation][trans]") {
    dense2D<double> m = {{1.0, 2.0}, {3.0, 4.0}};
    auto t = trans(m);
    m(0, 1) = 99.0;
    REQUIRE(t(1, 0) == 99.0);
}

// ── Combined: y = A*x + b ──────────────────────────────────────────────

TEST_CASE("y = A*x + b pattern", "[mat][operators]") {
    dense2D<double> A = {{1.0, 0.0}, {0.0, 1.0}};
    dense_vector<double> x = {3.0, 4.0};
    dense_vector<double> b = {1.0, 2.0};
    auto y = A * x + b;
    REQUIRE(y(0) == 4.0);
    REQUIRE(y(1) == 6.0);
}
