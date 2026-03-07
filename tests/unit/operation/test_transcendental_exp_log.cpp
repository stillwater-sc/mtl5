#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/operation/exp.hpp>
#include <mtl/operation/log.hpp>
#include <mtl/operation/exp2.hpp>
#include <mtl/operation/log2.hpp>
#include <mtl/operation/log10.hpp>
#include <mtl/operation/cbrt.hpp>
#include <mtl/operation/pow.hpp>
#include <cmath>
#include <numbers>

using namespace mtl;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

// -- exp -----------------------------------------------------------------

TEST_CASE("exp of vector", "[operation][transcendental][exp]") {
    dense_vector<double> v = {0.0, 1.0, 2.0};
    auto r = mtl::exp(v);
    REQUIRE_THAT(r(0), WithinAbs(1.0, 1e-10));
    REQUIRE_THAT(r(1), WithinAbs(std::numbers::e, 1e-10));
    REQUIRE_THAT(r(2), WithinAbs(std::exp(2.0), 1e-10));
}

TEST_CASE("exp of matrix", "[operation][transcendental][exp]") {
    mat::dense2D<double> m(2, 2);
    m(0,0) = 0.0; m(0,1) = 1.0;
    m(1,0) = -1.0; m(1,1) = 2.0;
    auto r = mtl::exp(m);
    REQUIRE_THAT(r(0,0), WithinAbs(1.0, 1e-10));
    REQUIRE_THAT(r(0,1), WithinAbs(std::numbers::e, 1e-10));
    REQUIRE_THAT(r(1,0), WithinAbs(std::exp(-1.0), 1e-10));
    REQUIRE_THAT(r(1,1), WithinAbs(std::exp(2.0), 1e-10));
}

// -- log -----------------------------------------------------------------

TEST_CASE("log of vector", "[operation][transcendental][log]") {
    dense_vector<double> v = {1.0, std::numbers::e, std::exp(2.0)};
    auto r = mtl::log(v);
    REQUIRE_THAT(r(0), WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(r(1), WithinAbs(1.0, 1e-10));
    REQUIRE_THAT(r(2), WithinAbs(2.0, 1e-10));
}

TEST_CASE("log of matrix", "[operation][transcendental][log]") {
    mat::dense2D<double> m(1, 2);
    m(0,0) = 1.0; m(0,1) = 10.0;
    auto r = mtl::log(m);
    REQUIRE_THAT(r(0,0), WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(r(0,1), WithinAbs(std::log(10.0), 1e-10));
}

// -- exp2 / log2 ---------------------------------------------------------

TEST_CASE("exp2 of vector", "[operation][transcendental][exp2]") {
    dense_vector<double> v = {0.0, 1.0, 3.0, 10.0};
    auto r = mtl::exp2(v);
    REQUIRE_THAT(r(0), WithinAbs(1.0, 1e-10));
    REQUIRE_THAT(r(1), WithinAbs(2.0, 1e-10));
    REQUIRE_THAT(r(2), WithinAbs(8.0, 1e-10));
    REQUIRE_THAT(r(3), WithinAbs(1024.0, 1e-10));
}

TEST_CASE("log2 of vector", "[operation][transcendental][log2]") {
    dense_vector<double> v = {1.0, 2.0, 8.0, 1024.0};
    auto r = mtl::log2(v);
    REQUIRE_THAT(r(0), WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(r(1), WithinAbs(1.0, 1e-10));
    REQUIRE_THAT(r(2), WithinAbs(3.0, 1e-10));
    REQUIRE_THAT(r(3), WithinAbs(10.0, 1e-10));
}

// -- log10 ---------------------------------------------------------------

TEST_CASE("log10 of vector", "[operation][transcendental][log10]") {
    dense_vector<double> v = {1.0, 10.0, 100.0, 1000.0};
    auto r = mtl::log10(v);
    REQUIRE_THAT(r(0), WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(r(1), WithinAbs(1.0, 1e-10));
    REQUIRE_THAT(r(2), WithinAbs(2.0, 1e-10));
    REQUIRE_THAT(r(3), WithinAbs(3.0, 1e-10));
}

TEST_CASE("log10 of matrix", "[operation][transcendental][log10]") {
    mat::dense2D<double> m(2, 2);
    m(0,0) = 1.0;   m(0,1) = 10.0;
    m(1,0) = 100.0;  m(1,1) = 1000.0;
    auto r = mtl::log10(m);
    REQUIRE_THAT(r(0,0), WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(r(0,1), WithinAbs(1.0, 1e-10));
    REQUIRE_THAT(r(1,0), WithinAbs(2.0, 1e-10));
    REQUIRE_THAT(r(1,1), WithinAbs(3.0, 1e-10));
}

// -- cbrt ----------------------------------------------------------------

TEST_CASE("cbrt of vector", "[operation][transcendental][cbrt]") {
    dense_vector<double> v = {0.0, 1.0, 8.0, 27.0, -8.0};
    auto r = mtl::cbrt(v);
    REQUIRE_THAT(r(0), WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(r(1), WithinAbs(1.0, 1e-10));
    REQUIRE_THAT(r(2), WithinAbs(2.0, 1e-10));
    REQUIRE_THAT(r(3), WithinAbs(3.0, 1e-10));
    REQUIRE_THAT(r(4), WithinAbs(-2.0, 1e-10));
}

// -- pow -----------------------------------------------------------------

TEST_CASE("pow of vector with scalar exponent", "[operation][transcendental][pow]") {
    dense_vector<double> v = {1.0, 2.0, 3.0, 4.0};
    auto r = mtl::pow(v, 2.0);
    REQUIRE_THAT(r(0), WithinAbs(1.0, 1e-10));
    REQUIRE_THAT(r(1), WithinAbs(4.0, 1e-10));
    REQUIRE_THAT(r(2), WithinAbs(9.0, 1e-10));
    REQUIRE_THAT(r(3), WithinAbs(16.0, 1e-10));
}

TEST_CASE("pow of matrix with scalar exponent", "[operation][transcendental][pow]") {
    mat::dense2D<double> m(2, 2);
    m(0,0) = 2.0; m(0,1) = 3.0;
    m(1,0) = 4.0; m(1,1) = 5.0;
    auto r = mtl::pow(m, 3.0);
    REQUIRE_THAT(r(0,0), WithinAbs(8.0, 1e-10));
    REQUIRE_THAT(r(0,1), WithinAbs(27.0, 1e-10));
    REQUIRE_THAT(r(1,0), WithinAbs(64.0, 1e-10));
    REQUIRE_THAT(r(1,1), WithinAbs(125.0, 1e-10));
}

TEST_CASE("pow with fractional exponent", "[operation][transcendental][pow]") {
    dense_vector<double> v = {4.0, 9.0, 16.0};
    auto r = mtl::pow(v, 0.5);
    REQUIRE_THAT(r(0), WithinAbs(2.0, 1e-10));
    REQUIRE_THAT(r(1), WithinAbs(3.0, 1e-10));
    REQUIRE_THAT(r(2), WithinAbs(4.0, 1e-10));
}

// -- Round-trip: log(exp(x)) == x ----------------------------------------

TEST_CASE("exp-log round trip", "[operation][transcendental][identity]") {
    dense_vector<double> v = {-2.0, -1.0, 0.0, 0.5, 1.0, 3.0};
    auto r = mtl::log(mtl::exp(v));
    for (std::size_t i = 0; i < v.size(); ++i) {
        REQUIRE_THAT(r(i), WithinAbs(v(i), 1e-10));
    }
}
