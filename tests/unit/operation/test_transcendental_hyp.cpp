#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/operation/sinh.hpp>
#include <mtl/operation/cosh.hpp>
#include <mtl/operation/tanh.hpp>
#include <mtl/operation/asinh.hpp>
#include <mtl/operation/acosh.hpp>
#include <mtl/operation/atanh.hpp>
#include <cmath>

using namespace mtl;
using Catch::Matchers::WithinAbs;

// -- sinh ----------------------------------------------------------------

TEST_CASE("sinh of vector", "[operation][transcendental][sinh]") {
    dense_vector<double> v = {0.0, 1.0, -1.0};
    auto r = mtl::sinh(v);
    REQUIRE_THAT(r(0), WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(r(1), WithinAbs(std::sinh(1.0), 1e-10));
    REQUIRE_THAT(r(2), WithinAbs(std::sinh(-1.0), 1e-10));
}

TEST_CASE("sinh of matrix", "[operation][transcendental][sinh]") {
    mat::dense2D<double> m(1, 2);
    m(0,0) = 0.0; m(0,1) = 1.0;
    auto r = mtl::sinh(m);
    REQUIRE_THAT(r(0,0), WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(r(0,1), WithinAbs(std::sinh(1.0), 1e-10));
}

// -- cosh ----------------------------------------------------------------

TEST_CASE("cosh of vector", "[operation][transcendental][cosh]") {
    dense_vector<double> v = {0.0, 1.0, -1.0};
    auto r = mtl::cosh(v);
    REQUIRE_THAT(r(0), WithinAbs(1.0, 1e-10));
    REQUIRE_THAT(r(1), WithinAbs(std::cosh(1.0), 1e-10));
    REQUIRE_THAT(r(2), WithinAbs(std::cosh(-1.0), 1e-10));
}

// -- tanh ----------------------------------------------------------------

TEST_CASE("tanh of vector", "[operation][transcendental][tanh]") {
    dense_vector<double> v = {0.0, 1.0, -1.0};
    auto r = mtl::tanh(v);
    REQUIRE_THAT(r(0), WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(r(1), WithinAbs(std::tanh(1.0), 1e-10));
    REQUIRE_THAT(r(2), WithinAbs(std::tanh(-1.0), 1e-10));
}

TEST_CASE("tanh of matrix", "[operation][transcendental][tanh]") {
    mat::dense2D<double> m(2, 2);
    m(0,0) = 0.0;  m(0,1) = 0.5;
    m(1,0) = -0.5; m(1,1) = 1.0;
    auto r = mtl::tanh(m);
    REQUIRE_THAT(r(0,0), WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(r(0,1), WithinAbs(std::tanh(0.5), 1e-10));
    REQUIRE_THAT(r(1,0), WithinAbs(std::tanh(-0.5), 1e-10));
    REQUIRE_THAT(r(1,1), WithinAbs(std::tanh(1.0), 1e-10));
}

// -- asinh ---------------------------------------------------------------

TEST_CASE("asinh of vector", "[operation][transcendental][asinh]") {
    dense_vector<double> v = {0.0, std::sinh(1.0), std::sinh(2.0)};
    auto r = mtl::asinh(v);
    REQUIRE_THAT(r(0), WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(r(1), WithinAbs(1.0, 1e-10));
    REQUIRE_THAT(r(2), WithinAbs(2.0, 1e-10));
}

// -- acosh ---------------------------------------------------------------

TEST_CASE("acosh of vector", "[operation][transcendental][acosh]") {
    dense_vector<double> v = {1.0, std::cosh(1.0), std::cosh(2.0)};
    auto r = mtl::acosh(v);
    REQUIRE_THAT(r(0), WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(r(1), WithinAbs(1.0, 1e-10));
    REQUIRE_THAT(r(2), WithinAbs(2.0, 1e-10));
}

// -- atanh ---------------------------------------------------------------

TEST_CASE("atanh of vector", "[operation][transcendental][atanh]") {
    dense_vector<double> v = {0.0, std::tanh(0.5), std::tanh(1.0)};
    auto r = mtl::atanh(v);
    REQUIRE_THAT(r(0), WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(r(1), WithinAbs(0.5, 1e-10));
    REQUIRE_THAT(r(2), WithinAbs(1.0, 1e-10));
}

// -- Hyperbolic identity: cosh^2(x) - sinh^2(x) = 1 -----------------------

TEST_CASE("cosh^2 - sinh^2 = 1 identity", "[operation][transcendental][identity]") {
    dense_vector<double> v = {-2.0, -1.0, 0.0, 0.5, 1.0, 2.0};
    auto sh = mtl::sinh(v);
    auto ch = mtl::cosh(v);
    for (std::size_t i = 0; i < v.size(); ++i) {
        REQUIRE_THAT(ch(i) * ch(i) - sh(i) * sh(i), WithinAbs(1.0, 1e-10));
    }
}

// -- asinh-sinh round trip -----------------------------------------------

TEST_CASE("asinh(sinh(x)) round trip", "[operation][transcendental][identity]") {
    dense_vector<double> v = {-2.0, -1.0, 0.0, 1.0, 2.0};
    auto r = mtl::asinh(mtl::sinh(v));
    for (std::size_t i = 0; i < v.size(); ++i) {
        REQUIRE_THAT(r(i), WithinAbs(v(i), 1e-10));
    }
}
