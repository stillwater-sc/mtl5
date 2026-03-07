#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/operation/ceil.hpp>
#include <mtl/operation/floor.hpp>
#include <mtl/operation/round.hpp>
#include <mtl/operation/signum.hpp>
#include <mtl/operation/erf.hpp>
#include <mtl/operation/erfc.hpp>
#include <mtl/operation/real.hpp>
#include <mtl/operation/imag.hpp>
#include <cmath>
#include <complex>

using namespace mtl;
using Catch::Matchers::WithinAbs;

// ── ceil ────────────────────────────────────────────────────────────────

TEST_CASE("ceil of vector", "[operation][transcendental][ceil]") {
    dense_vector<double> v = {1.2, 2.7, -1.2, -2.7, 3.0};
    auto r = mtl::ceil(v);
    REQUIRE(r(0) == 2.0);
    REQUIRE(r(1) == 3.0);
    REQUIRE(r(2) == -1.0);
    REQUIRE(r(3) == -2.0);
    REQUIRE(r(4) == 3.0);
}

TEST_CASE("ceil of matrix", "[operation][transcendental][ceil]") {
    mat::dense2D<double> m(1, 2);
    m(0,0) = 1.3; m(0,1) = -1.3;
    auto r = mtl::ceil(m);
    REQUIRE(r(0,0) == 2.0);
    REQUIRE(r(0,1) == -1.0);
}

// ── floor ───────────────────────────────────────────────────────────────

TEST_CASE("floor of vector", "[operation][transcendental][floor]") {
    dense_vector<double> v = {1.2, 2.7, -1.2, -2.7, 3.0};
    auto r = mtl::floor(v);
    REQUIRE(r(0) == 1.0);
    REQUIRE(r(1) == 2.0);
    REQUIRE(r(2) == -2.0);
    REQUIRE(r(3) == -3.0);
    REQUIRE(r(4) == 3.0);
}

// ── round ───────────────────────────────────────────────────────────────

TEST_CASE("round of vector", "[operation][transcendental][round]") {
    dense_vector<double> v = {1.4, 1.5, 2.5, -1.4, -1.5};
    auto r = mtl::round(v);
    REQUIRE(r(0) == 1.0);
    REQUIRE(r(1) == 2.0);
    REQUIRE(r(2) == 3.0);  // round half away from zero (std::round behavior)
    REQUIRE(r(3) == -1.0);
    REQUIRE(r(4) == -2.0);
}

TEST_CASE("round of matrix", "[operation][transcendental][round]") {
    mat::dense2D<double> m(1, 2);
    m(0,0) = 3.14; m(0,1) = -2.71;
    auto r = mtl::round(m);
    REQUIRE(r(0,0) == 3.0);
    REQUIRE(r(0,1) == -3.0);
}

// ── signum ──────────────────────────────────────────────────────────────

TEST_CASE("signum of vector", "[operation][transcendental][signum]") {
    dense_vector<double> v = {-5.0, -0.1, 0.0, 0.1, 5.0};
    auto r = signum(v);
    REQUIRE(r(0) == -1.0);
    REQUIRE(r(1) == -1.0);
    REQUIRE(r(2) == 0.0);
    REQUIRE(r(3) == 1.0);
    REQUIRE(r(4) == 1.0);
}

TEST_CASE("signum of matrix", "[operation][transcendental][signum]") {
    mat::dense2D<double> m(2, 2);
    m(0,0) = -3.0; m(0,1) = 0.0;
    m(1,0) = 0.0;  m(1,1) = 7.0;
    auto r = signum(m);
    REQUIRE(r(0,0) == -1.0);
    REQUIRE(r(0,1) == 0.0);
    REQUIRE(r(1,0) == 0.0);
    REQUIRE(r(1,1) == 1.0);
}

TEST_CASE("signum of integer vector", "[operation][transcendental][signum]") {
    dense_vector<int> v = {-3, 0, 5};
    auto r = signum(v);
    REQUIRE(r(0) == -1);
    REQUIRE(r(1) == 0);
    REQUIRE(r(2) == 1);
}

// ── erf / erfc ──────────────────────────────────────────────────────────

TEST_CASE("erf of vector", "[operation][transcendental][erf]") {
    dense_vector<double> v = {0.0, 1.0, -1.0};
    auto r = mtl::erf(v);
    REQUIRE_THAT(r(0), WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(r(1), WithinAbs(std::erf(1.0), 1e-10));
    REQUIRE_THAT(r(2), WithinAbs(std::erf(-1.0), 1e-10));
}

TEST_CASE("erfc of vector", "[operation][transcendental][erfc]") {
    dense_vector<double> v = {0.0, 1.0, 2.0};
    auto r = mtl::erfc(v);
    REQUIRE_THAT(r(0), WithinAbs(1.0, 1e-10));
    REQUIRE_THAT(r(1), WithinAbs(std::erfc(1.0), 1e-10));
    REQUIRE_THAT(r(2), WithinAbs(std::erfc(2.0), 1e-10));
}

TEST_CASE("erf + erfc = 1 identity", "[operation][transcendental][identity]") {
    dense_vector<double> v = {-2.0, -1.0, 0.0, 0.5, 1.0, 2.0};
    auto e = mtl::erf(v);
    auto ec = mtl::erfc(v);
    for (std::size_t i = 0; i < v.size(); ++i) {
        REQUIRE_THAT(e(i) + ec(i), WithinAbs(1.0, 1e-10));
    }
}

TEST_CASE("erf of matrix", "[operation][transcendental][erf]") {
    mat::dense2D<double> m(1, 2);
    m(0,0) = 0.0; m(0,1) = 1.0;
    auto r = mtl::erf(m);
    REQUIRE_THAT(r(0,0), WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(r(0,1), WithinAbs(std::erf(1.0), 1e-10));
}

// ── real / imag for real vectors ────────────────────────────────────────

TEST_CASE("real of real vector", "[operation][transcendental][real]") {
    dense_vector<double> v = {1.0, 2.0, 3.0};
    auto r = real(v);
    REQUIRE(r(0) == 1.0);
    REQUIRE(r(1) == 2.0);
    REQUIRE(r(2) == 3.0);
}

TEST_CASE("imag of real vector is zero", "[operation][transcendental][imag]") {
    dense_vector<double> v = {1.0, 2.0, 3.0};
    auto r = imag(v);
    REQUIRE(r(0) == 0.0);
    REQUIRE(r(1) == 0.0);
    REQUIRE(r(2) == 0.0);
}

// ── real / imag for complex vectors ─────────────────────────────────────

TEST_CASE("real of complex vector", "[operation][transcendental][real]") {
    using cd = std::complex<double>;
    dense_vector<cd> v = {cd(1.0, 2.0), cd(3.0, -4.0), cd(0.0, 5.0)};
    auto r = real(v);
    REQUIRE(r(0) == 1.0);
    REQUIRE(r(1) == 3.0);
    REQUIRE(r(2) == 0.0);
}

TEST_CASE("imag of complex vector", "[operation][transcendental][imag]") {
    using cd = std::complex<double>;
    dense_vector<cd> v = {cd(1.0, 2.0), cd(3.0, -4.0), cd(0.0, 5.0)};
    auto r = imag(v);
    REQUIRE(r(0) == 2.0);
    REQUIRE(r(1) == -4.0);
    REQUIRE(r(2) == 5.0);
}

// ── real / imag for complex matrices ────────────────────────────────────

TEST_CASE("real of complex matrix", "[operation][transcendental][real]") {
    using cd = std::complex<double>;
    mat::dense2D<cd> m(1, 2);
    m(0,0) = cd(1.0, 2.0); m(0,1) = cd(3.0, -4.0);
    auto r = real(m);
    REQUIRE(r(0,0) == 1.0);
    REQUIRE(r(0,1) == 3.0);
}

TEST_CASE("imag of complex matrix", "[operation][transcendental][imag]") {
    using cd = std::complex<double>;
    mat::dense2D<cd> m(1, 2);
    m(0,0) = cd(1.0, 2.0); m(0,1) = cd(3.0, -4.0);
    auto r = imag(m);
    REQUIRE(r(0,0) == 2.0);
    REQUIRE(r(0,1) == -4.0);
}
