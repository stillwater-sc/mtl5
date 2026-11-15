#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <mtl/generators/ranges.hpp>

using namespace mtl;
using Catch::Matchers::WithinAbs;

// -- arange (half-open, NumPy semantics) ---------------------------------

TEST_CASE("arange basic half-open", "[generators][ranges][arange]") {
    auto v = generators::arange<double>(0, 5);
    REQUIRE(v.size() == 5);
    for (std::size_t i = 0; i < 5; ++i) REQUIRE_THAT(v[i], WithinAbs(double(i), 1e-12));
    // stop is excluded
}

TEST_CASE("arange with step", "[generators][ranges][arange]") {
    auto v = generators::arange<double>(1, 10, 2);   // 1,3,5,7,9
    REQUIRE(v.size() == 5);
    REQUIRE_THAT(v[0], WithinAbs(1.0, 1e-12));
    REQUIRE_THAT(v[4], WithinAbs(9.0, 1e-12));
}

TEST_CASE("arange negative step", "[generators][ranges][arange]") {
    auto v = generators::arange<double>(5, 0, -1);   // 5,4,3,2,1
    REQUIRE(v.size() == 5);
    REQUIRE_THAT(v[0], WithinAbs(5.0, 1e-12));
    REQUIRE_THAT(v[4], WithinAbs(1.0, 1e-12));
}

TEST_CASE("arange empty and degenerate", "[generators][ranges][arange]") {
    REQUIRE(generators::arange<double>(5, 0).size() == 0);   // wrong direction
    REQUIRE(generators::arange<double>(0, 0).size() == 0);   // empty
    REQUIRE(generators::arange<double>(0, 5, 0).size() == 0);// zero step
}

TEST_CASE("arange int64 extremes do not overflow", "[generators][ranges][arange]") {
    constexpr std::int64_t IMAX = std::numeric_limits<std::int64_t>::max();
    constexpr std::int64_t IMIN = std::numeric_limits<std::int64_t>::min();
    // One element just below the max: the unused final increment must not overflow.
    auto a = generators::arange<double>(IMAX - 1, IMAX, 2);
    REQUIRE(a.size() == 1);
    REQUIRE_THAT(a[0], WithinAbs(double(IMAX - 1), 0.0));
    // Descending with step == INT64_MIN: |step| must be computed without overflow.
    auto b = generators::arange<double>(0, IMIN, IMIN);   // one element: 0
    REQUIRE(b.size() == 1);
    REQUIRE_THAT(b[0], WithinAbs(0.0, 0.0));
    // Full-range span: the ceiling-division count must not wrap in uint64.
    auto c = generators::arange<double>(IMIN, IMAX, IMAX);   // IMIN, -1, IMAX-1
    REQUIRE(c.size() == 3);
    REQUIRE_THAT(c[0], WithinAbs(double(IMIN), 0.0));
    REQUIRE_THAT(c[1], WithinAbs(-1.0, 0.0));
}

// -- linspace ------------------------------------------------------------

TEST_CASE("linspace endpoint inclusive", "[generators][ranges][linspace]") {
    auto v = generators::linspace<double>(0.0, 1.0, 5);   // 0,.25,.5,.75,1
    REQUIRE(v.size() == 5);
    REQUIRE_THAT(v[0], WithinAbs(0.0, 1e-12));
    REQUIRE_THAT(v[1], WithinAbs(0.25, 1e-12));
    REQUIRE_THAT(v[2], WithinAbs(0.50, 1e-12));
    REQUIRE_THAT(v[3], WithinAbs(0.75, 1e-12));
    REQUIRE_THAT(v[4], WithinAbs(1.0, 1e-12));   // exact endpoint
}

TEST_CASE("linspace endpoint exclusive", "[generators][ranges][linspace]") {
    auto v = generators::linspace<double>(0.0, 1.0, 5, false);   // 0,.2,.4,.6,.8
    REQUIRE(v.size() == 5);
    REQUIRE_THAT(v[0], WithinAbs(0.0, 1e-12));
    REQUIRE_THAT(v[1], WithinAbs(0.2, 1e-12));
    REQUIRE_THAT(v[4], WithinAbs(0.8, 1e-12));
}

TEST_CASE("linspace edge cases", "[generators][ranges][linspace]") {
    REQUIRE(generators::linspace<double>(0.0, 1.0, 0).size() == 0);
    auto one = generators::linspace<double>(3.0, 9.0, 1);
    REQUIRE(one.size() == 1);
    REQUIRE_THAT(one[0], WithinAbs(3.0, 1e-12));
}

// -- logspace (start/stop are exponents) ---------------------------------

TEST_CASE("logspace base 10 exponents", "[generators][ranges][logspace]") {
    auto v = generators::logspace<double>(0.0, 3.0, 4);   // 10^0..10^3
    REQUIRE(v.size() == 4);
    REQUIRE_THAT(v[0], WithinAbs(1.0, 1e-9));
    REQUIRE_THAT(v[1], WithinAbs(10.0, 1e-9));
    REQUIRE_THAT(v[2], WithinAbs(100.0, 1e-9));
    REQUIRE_THAT(v[3], WithinAbs(1000.0, 1e-9));
}

// -- geomspace (start/stop are endpoints -- fixes the old Universal bug) --

TEST_CASE("geomspace decade endpoints", "[generators][ranges][geomspace]") {
    auto v = generators::geomspace<double>(1.0, 1000.0, 4);   // 1,10,100,1000
    REQUIRE(v.size() == 4);
    REQUIRE_THAT(v[0], WithinAbs(1.0, 1e-9));
    REQUIRE_THAT(v[1], WithinAbs(10.0, 1e-9));
    REQUIRE_THAT(v[2], WithinAbs(100.0, 1e-9));
    REQUIRE_THAT(v[3], WithinAbs(1000.0, 1e-9));
}

TEST_CASE("geomspace powers of two", "[generators][ranges][geomspace]") {
    auto v = generators::geomspace<double>(1.0, 256.0, 9);   // 1,2,4,...,256
    REQUIRE(v.size() == 9);
    for (std::size_t i = 0; i < 9; ++i)
        REQUIRE_THAT(v[i], WithinAbs(std::pow(2.0, double(i)), 1e-6));
}

TEST_CASE("geomspace negative same-sign endpoints", "[generators][ranges][geomspace]") {
    auto v = generators::geomspace<double>(-1000.0, -1.0, 4);   // -1000,-100,-10,-1
    REQUIRE(v.size() == 4);
    REQUIRE_THAT(v[0], WithinAbs(-1000.0, 1e-6));
    REQUIRE_THAT(v[1], WithinAbs(-100.0, 1e-6));   // interior values, geometric
    REQUIRE_THAT(v[2], WithinAbs(-10.0, 1e-6));
    REQUIRE_THAT(v[3], WithinAbs(-1.0, 1e-9));
}

TEST_CASE("geomspace rejects invalid domains", "[generators][ranges][geomspace]") {
    using generators::geomspace;
    // zero endpoint: ratio would divide by zero
    REQUIRE_THROWS_AS(geomspace<double>(0.0, 10.0, 4), std::invalid_argument);
    REQUIRE_THROWS_AS(geomspace<double>(1.0, 0.0, 4), std::invalid_argument);
    // opposite signs: pow(negative ratio, fractional) is NaN
    REQUIRE_THROWS_AS(geomspace<double>(-1.0, 1000.0, 4), std::invalid_argument);
    REQUIRE_THROWS_AS(geomspace<double>(1.0, -1000.0, 4), std::invalid_argument);
}

// -- #844 gaps -------------------------------------------------------------

TEST_CASE("logspace with endpoint=false and another base", "[generators][ranges][logspace]") {
    auto v = generators::logspace<double>(0.0, 4.0, 4, false, 2.0);   // exponents 0, 1, 2, 3
    REQUIRE(v.size() == 4);
    REQUIRE_THAT(v[0], WithinAbs(1.0, 1e-15));
    REQUIRE_THAT(v[1], WithinAbs(2.0, 1e-15));
    REQUIRE_THAT(v[2], WithinAbs(4.0, 1e-14));
    REQUIRE_THAT(v[3], WithinAbs(8.0, 1e-14));
    auto w = generators::logspace<double>(1.0, 3.0, 3, true, 3.0);    // 3, 9, 27
    REQUIRE_THAT(w[2], WithinAbs(27.0, 1e-12));
}

TEST_CASE("geomspace with endpoint=false and with one step", "[generators][ranges][geomspace]") {
    auto v = generators::geomspace<double>(1.0, 16.0, 4, false);      // 1, 2, 4, 8
    REQUIRE(v.size() == 4);
    REQUIRE_THAT(v[0], WithinAbs(1.0, 1e-15));
    REQUIRE_THAT(v[1], WithinAbs(2.0, 1e-14));
    REQUIRE_THAT(v[3], WithinAbs(8.0, 1e-13));
    for (bool endpoint : {true, false}) {
        auto one = generators::geomspace<double>(3.0, 300.0, 1, endpoint);
        REQUIRE(one.size() == 1);
        REQUIRE(one[0] == 3.0);
    }
}

TEST_CASE("geomspace rejects NaN endpoints", "[generators][ranges][geomspace]") {
    // NaN fails both the == 0 and the sign comparisons, so it used to pass
    const double nan = std::numeric_limits<double>::quiet_NaN();
    REQUIRE_THROWS_AS(generators::geomspace<double>(nan, 10.0, 4), std::invalid_argument);
    REQUIRE_THROWS_AS(generators::geomspace<double>(1.0, nan, 4), std::invalid_argument);
}

TEST_CASE("arange with a negative step near the int64 limits", "[generators][ranges][arange]") {
    constexpr std::int64_t hi = std::numeric_limits<std::int64_t>::max();
    constexpr std::int64_t lo = std::numeric_limits<std::int64_t>::min();
    auto a = generators::arange<std::int64_t>(hi, hi - 10, -3);       // hi, hi-3, hi-6, hi-9
    REQUIRE(a.size() == 4);
    REQUIRE(a[0] == hi);
    REQUIRE(a[3] == hi - 9);
    auto b = generators::arange<std::int64_t>(lo + 5, lo, -2);        // lo+5, lo+3, lo+1
    REQUIRE(b.size() == 3);
    REQUIRE(b[2] == lo + 1);
    auto c = generators::arange<std::int64_t>(hi, lo, lo);            // one step of INT64_MIN
    REQUIRE(c.size() == 2);
    REQUIRE(c[0] == hi);
    REQUIRE(c[1] == -1);
}
