#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/vec/operators.hpp>
#include <mtl/operation/sum.hpp>
#include <mtl/operation/product.hpp>
#include <mtl/operation/max.hpp>
#include <mtl/operation/min.hpp>
#include <mtl/operation/abs.hpp>
#include <mtl/operation/conj.hpp>
#include <mtl/operation/scale.hpp>
#include <mtl/operation/negate.hpp>
#include <mtl/operation/sqrt.hpp>
#include <complex>
#include <cmath>

using namespace mtl;
using Catch::Matchers::WithinAbs;

// ── Vector + - operators ────────────────────────────────────────────────

TEST_CASE("vector addition", "[vec][operators]") {
    dense_vector<double> a = {1.0, 2.0, 3.0};
    dense_vector<double> b = {4.0, 5.0, 6.0};
    auto c = a + b;
    REQUIRE(c(0) == 5.0);
    REQUIRE(c(1) == 7.0);
    REQUIRE(c(2) == 9.0);
}

TEST_CASE("vector subtraction", "[vec][operators]") {
    dense_vector<double> a = {10.0, 20.0, 30.0};
    dense_vector<double> b = {1.0, 2.0, 3.0};
    auto c = a - b;
    REQUIRE(c(0) == 9.0);
    REQUIRE(c(1) == 18.0);
    REQUIRE(c(2) == 27.0);
}

TEST_CASE("vector unary negation", "[vec][operators]") {
    dense_vector<double> a = {1.0, -2.0, 3.0};
    auto b = -a;
    REQUIRE(b(0) == -1.0);
    REQUIRE(b(1) == 2.0);
    REQUIRE(b(2) == -3.0);
}

// ── Scalar-vector multiply / divide ─────────────────────────────────────

TEST_CASE("scalar * vector", "[vec][operators]") {
    dense_vector<double> v = {1.0, 2.0, 3.0};
    auto r = 2.0 * v;
    REQUIRE(r(0) == 2.0);
    REQUIRE(r(1) == 4.0);
    REQUIRE(r(2) == 6.0);
}

TEST_CASE("vector * scalar", "[vec][operators]") {
    dense_vector<double> v = {1.0, 2.0, 3.0};
    auto r = v * 3.0;
    REQUIRE(r(0) == 3.0);
    REQUIRE(r(1) == 6.0);
    REQUIRE(r(2) == 9.0);
}

TEST_CASE("vector / scalar", "[vec][operators]") {
    dense_vector<double> v = {4.0, 6.0, 8.0};
    auto r = v / 2.0;
    REQUIRE(r(0) == 2.0);
    REQUIRE(r(1) == 3.0);
    REQUIRE(r(2) == 4.0);
}

// ── Compound assignment ─────────────────────────────────────────────────

TEST_CASE("vector +=", "[vec][compound]") {
    dense_vector<double> a = {1.0, 2.0, 3.0};
    dense_vector<double> b = {10.0, 20.0, 30.0};
    a += b;
    REQUIRE(a(0) == 11.0);
    REQUIRE(a(1) == 22.0);
    REQUIRE(a(2) == 33.0);
}

TEST_CASE("vector -=", "[vec][compound]") {
    dense_vector<double> a = {10.0, 20.0, 30.0};
    dense_vector<double> b = {1.0, 2.0, 3.0};
    a -= b;
    REQUIRE(a(0) == 9.0);
    REQUIRE(a(1) == 18.0);
    REQUIRE(a(2) == 27.0);
}

TEST_CASE("vector *=", "[vec][compound]") {
    dense_vector<double> a = {1.0, 2.0, 3.0};
    a *= 5.0;
    REQUIRE(a(0) == 5.0);
    REQUIRE(a(1) == 10.0);
    REQUIRE(a(2) == 15.0);
}

TEST_CASE("vector /=", "[vec][compound]") {
    dense_vector<double> a = {10.0, 20.0, 30.0};
    a /= 10.0;
    REQUIRE(a(0) == 1.0);
    REQUIRE(a(1) == 2.0);
    REQUIRE(a(2) == 3.0);
}

// ── Reductions ──────────────────────────────────────────────────────────

TEST_CASE("sum of vector", "[operation][sum]") {
    dense_vector<double> v = {1.0, 2.0, 3.0, 4.0};
    REQUIRE(sum(v) == 10.0);
}

TEST_CASE("sum of empty vector", "[operation][sum]") {
    dense_vector<double> v;
    REQUIRE(sum(v) == 0.0);
}

TEST_CASE("product of vector", "[operation][product]") {
    dense_vector<int> v = {1, 2, 3, 4};
    REQUIRE(product(v) == 24);
}

TEST_CASE("product of single element", "[operation][product]") {
    dense_vector<int> v = {7};
    REQUIRE(product(v) == 7);
}

TEST_CASE("product of empty vector", "[operation][product]") {
	dense_vector<int> v;
	REQUIRE(product(v) == 1);
}

TEST_CASE("max of vector", "[operation][max]") {
    dense_vector<double> v = {1.0, 5.0, 3.0, 2.0};
    REQUIRE(max(v) == 5.0);
}

TEST_CASE("max with negative values", "[operation][max]") {
    dense_vector<int> v = {-10, -5, -1, -20};
    REQUIRE(max(v) == -1);
}

TEST_CASE("min of vector", "[operation][min]") {
    dense_vector<double> v = {5.0, 1.0, 3.0, 2.0};
    REQUIRE(min(v) == 1.0);
}

TEST_CASE("min with negative values", "[operation][min]") {
    dense_vector<int> v = {-10, -5, -1, -20};
    REQUIRE(min(v) == -20);
}

// ── Element-wise operations ─────────────────────────────────────────────

TEST_CASE("abs of vector", "[operation][abs]") {
    dense_vector<double> v = {-1.0, 2.0, -3.0};
    auto r = abs(v);
    REQUIRE(r(0) == 1.0);
    REQUIRE(r(1) == 2.0);
    REQUIRE(r(2) == 3.0);
}

TEST_CASE("conj of real vector is identity", "[operation][conj]") {
    dense_vector<double> v = {1.0, 2.0, 3.0};
    auto r = conj(v);
    REQUIRE(r(0) == 1.0);
    REQUIRE(r(1) == 2.0);
    REQUIRE(r(2) == 3.0);
}

TEST_CASE("conj of complex vector", "[operation][conj]") {
    using cd = std::complex<double>;
    dense_vector<cd> v = {cd(1.0, 2.0), cd(3.0, -4.0)};
    auto r = conj(v);
    REQUIRE(r(0) == cd(1.0, -2.0));
    REQUIRE(r(1) == cd(3.0, 4.0));
}

TEST_CASE("negate of vector", "[operation][negate]") {
    dense_vector<double> v = {1.0, -2.0, 3.0};
    auto r = negate(v);
    REQUIRE(r(0) == -1.0);
    REQUIRE(r(1) == 2.0);
    REQUIRE(r(2) == -3.0);
}

TEST_CASE("scale in-place", "[operation][scale]") {
    dense_vector<double> v = {1.0, 2.0, 3.0};
    scale(2.0, v);
    REQUIRE(v(0) == 2.0);
    REQUIRE(v(1) == 4.0);
    REQUIRE(v(2) == 6.0);
}

TEST_CASE("scaled returns copy", "[operation][scale]") {
    dense_vector<double> v = {1.0, 2.0, 3.0};
    auto r = scaled(3.0, v);
    REQUIRE(r(0) == 3.0);
    REQUIRE(r(1) == 6.0);
    REQUIRE(r(2) == 9.0);
    // original unchanged
    REQUIRE(v(0) == 1.0);
}

// ── sqrt on vectors ─────────────────────────────────────────────────────

TEST_CASE("sqrt of vector", "[operation][sqrt]") {
    dense_vector<double> v = {4.0, 9.0, 16.0};
    auto r = mtl::sqrt(v);
    REQUIRE_THAT(r(0), WithinAbs(2.0, 1e-10));
    REQUIRE_THAT(r(1), WithinAbs(3.0, 1e-10));
    REQUIRE_THAT(r(2), WithinAbs(4.0, 1e-10));
}

TEST_CASE("sqrt of vector with non-perfect squares", "[operation][sqrt]") {
    dense_vector<double> v = {2.0, 3.0};
    auto r = mtl::sqrt(v);
    REQUIRE_THAT(r(0), WithinAbs(std::sqrt(2.0), 1e-10));
    REQUIRE_THAT(r(1), WithinAbs(std::sqrt(3.0), 1e-10));
}

// ── Mixed-type vector operations ────────────────────────────────────────

TEST_CASE("int + double vector addition", "[vec][operators][mixed]") {
    dense_vector<int> a = {1, 2, 3};
    dense_vector<double> b = {0.5, 1.5, 2.5};
    auto c = a + b;
    STATIC_REQUIRE(std::is_same_v<typename decltype(c)::value_type, double>);
    REQUIRE(c(0) == 1.5);
    REQUIRE(c(1) == 3.5);
    REQUIRE(c(2) == 5.5);
}

TEST_CASE("int - double vector subtraction", "[vec][operators][mixed]") {
    dense_vector<int> a = {10, 20, 30};
    dense_vector<double> b = {0.5, 1.5, 2.5};
    auto c = a - b;
    STATIC_REQUIRE(std::is_same_v<typename decltype(c)::value_type, double>);
    REQUIRE(c(0) == 9.5);
    REQUIRE(c(1) == 18.5);
    REQUIRE(c(2) == 27.5);
}

TEST_CASE("int scalar * double vector", "[vec][operators][mixed]") {
    dense_vector<double> v = {1.5, 2.5, 3.5};
    auto r = 2 * v;
    STATIC_REQUIRE(std::is_same_v<typename decltype(r)::value_type, double>);
    REQUIRE(r(0) == 3.0);
    REQUIRE(r(1) == 5.0);
    REQUIRE(r(2) == 7.0);
}

// ── Chained expression: y = A*x + b pattern ────────────────────────────

TEST_CASE("y = alpha*x + b pattern", "[vec][operators]") {
    dense_vector<double> x = {1.0, 2.0, 3.0};
    dense_vector<double> b = {10.0, 20.0, 30.0};
    auto y = 2.0 * x + b;
    REQUIRE(y(0) == 12.0);
    REQUIRE(y(1) == 24.0);
    REQUIRE(y(2) == 36.0);
}

// -- construction from std::vector and initializer list

TEST_CASE("construct from std::vector", "[vec][construction]") {
    std::vector<double> std_vec = {1.0, 2.0, 3.0};
    dense_vector<double> v(std_vec);
    REQUIRE(v(0) == 1.0);
    REQUIRE(v(1) == 2.0);
    REQUIRE(v(2) == 3.0);
}

// -- move semantics

TEST_CASE("move constructor", "[vec][move]") {
    dense_vector<double> a = {1.0, 2.0, 3.0};
    dense_vector<double> b = std::move(a);
    REQUIRE(b(0) == 1.0);
    REQUIRE(b(1) == 2.0);
    REQUIRE(b(2) == 3.0);
}

// -- construct/destruct

TEST_CASE("construct and destruct", "[vec][lifetime]") {
    {
        dense_vector<double> v(1024*1024, 3.14);
        REQUIRE(v(0) == 3.14);
        REQUIRE(v(999) == 3.14);
		REQUIRE(v(1024 * 1024 - 1) == 3.14);
    } // v goes out of scope here, should not leak memory
    {
		dense_vector<std::complex<double>> v(1024 * 1024, std::complex<double>(1.0, 2.0));
		REQUIRE(v(0) == std::complex<double>(1.0, 2.0));
		REQUIRE(v(1024) == std::complex<double>(1.0, 2.0));
		REQUIRE(v(1024 * 1024 - 1) == std::complex<double>(1.0, 2.0));
    }
}