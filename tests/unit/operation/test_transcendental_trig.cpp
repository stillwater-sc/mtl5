#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/operation/sin.hpp>
#include <mtl/operation/cos.hpp>
#include <mtl/operation/tan.hpp>
#include <mtl/operation/asin.hpp>
#include <mtl/operation/acos.hpp>
#include <mtl/operation/atan.hpp>
#include <cmath>
#include <numbers>

using namespace mtl;
using Catch::Matchers::WithinAbs;

// -- sin -----------------------------------------------------------------

TEST_CASE("sin of vector", "[operation][transcendental][sin]") {
    const double pi = std::numbers::pi;
    dense_vector<double> v = {0.0, pi / 6.0, pi / 2.0, pi};
    auto r = mtl::sin(v);
    REQUIRE_THAT(r(0), WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(r(1), WithinAbs(0.5, 1e-10));
    REQUIRE_THAT(r(2), WithinAbs(1.0, 1e-10));
    REQUIRE_THAT(r(3), WithinAbs(0.0, 1e-10));
}

TEST_CASE("sin of matrix", "[operation][transcendental][sin]") {
    const double pi = std::numbers::pi;
    mat::dense2D<double> m(1, 2);
    m(0,0) = 0.0; m(0,1) = pi / 2.0;
    auto r = mtl::sin(m);
    REQUIRE_THAT(r(0,0), WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(r(0,1), WithinAbs(1.0, 1e-10));
}

// -- cos -----------------------------------------------------------------

TEST_CASE("cos of vector", "[operation][transcendental][cos]") {
    const double pi = std::numbers::pi;
    dense_vector<double> v = {0.0, pi / 3.0, pi / 2.0, pi};
    auto r = mtl::cos(v);
    REQUIRE_THAT(r(0), WithinAbs(1.0, 1e-10));
    REQUIRE_THAT(r(1), WithinAbs(0.5, 1e-10));
    REQUIRE_THAT(r(2), WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(r(3), WithinAbs(-1.0, 1e-10));
}

// -- tan -----------------------------------------------------------------

TEST_CASE("tan of vector", "[operation][transcendental][tan]") {
    const double pi = std::numbers::pi;
    dense_vector<double> v = {0.0, pi / 4.0};
    auto r = mtl::tan(v);
    REQUIRE_THAT(r(0), WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(r(1), WithinAbs(1.0, 1e-10));
}

// -- asin ----------------------------------------------------------------

TEST_CASE("asin of vector", "[operation][transcendental][asin]") {
    const double pi = std::numbers::pi;
    dense_vector<double> v = {0.0, 0.5, 1.0};
    auto r = mtl::asin(v);
    REQUIRE_THAT(r(0), WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(r(1), WithinAbs(pi / 6.0, 1e-10));
    REQUIRE_THAT(r(2), WithinAbs(pi / 2.0, 1e-10));
}

// -- acos ----------------------------------------------------------------

TEST_CASE("acos of vector", "[operation][transcendental][acos]") {
    const double pi = std::numbers::pi;
    dense_vector<double> v = {1.0, 0.5, 0.0};
    auto r = mtl::acos(v);
    REQUIRE_THAT(r(0), WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(r(1), WithinAbs(pi / 3.0, 1e-10));
    REQUIRE_THAT(r(2), WithinAbs(pi / 2.0, 1e-10));
}

// -- atan ----------------------------------------------------------------

TEST_CASE("atan of vector", "[operation][transcendental][atan]") {
    const double pi = std::numbers::pi;
    dense_vector<double> v = {0.0, 1.0};
    auto r = mtl::atan(v);
    REQUIRE_THAT(r(0), WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(r(1), WithinAbs(pi / 4.0, 1e-10));
}

TEST_CASE("atan of matrix", "[operation][transcendental][atan]") {
    mat::dense2D<double> m(1, 2);
    m(0,0) = 0.0; m(0,1) = 1.0;
    auto r = mtl::atan(m);
    REQUIRE_THAT(r(0,0), WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(r(0,1), WithinAbs(std::numbers::pi / 4.0, 1e-10));
}

// -- Pythagorean identity: sin^2(x) + cos^2(x) = 1 ------------------------

TEST_CASE("sin^2 + cos^2 = 1 identity", "[operation][transcendental][identity]") {
    const double pi = std::numbers::pi;
    dense_vector<double> v = {0.0, pi / 6.0, pi / 4.0, pi / 3.0, pi / 2.0, pi};
    auto s = mtl::sin(v);
    auto c = mtl::cos(v);
    for (std::size_t i = 0; i < v.size(); ++i) {
        REQUIRE_THAT(s(i) * s(i) + c(i) * c(i), WithinAbs(1.0, 1e-10));
    }
}

// -- asin-sin round trip -------------------------------------------------

TEST_CASE("sin(asin(x)) round trip", "[operation][transcendental][identity]") {
    dense_vector<double> v = {-0.5, 0.0, 0.3, 0.7, 1.0};
    auto r = mtl::sin(mtl::asin(v));
    for (std::size_t i = 0; i < v.size(); ++i) {
        REQUIRE_THAT(r(i), WithinAbs(v(i), 1e-10));
    }
}
