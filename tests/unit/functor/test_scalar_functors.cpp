#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <mtl/functor/scalar/plus.hpp>
#include <mtl/functor/scalar/minus.hpp>
#include <mtl/functor/scalar/times.hpp>
#include <mtl/functor/scalar/divide.hpp>
#include <mtl/functor/scalar/assign.hpp>
#include <mtl/functor/scalar/negate.hpp>
#include <mtl/functor/scalar/abs.hpp>
#include <mtl/functor/scalar/conj.hpp>
#include <mtl/functor/scalar/sqrt.hpp>
#include <mtl/functor/typed/scale.hpp>
#include <mtl/functor/typed/rscale.hpp>
#include <mtl/functor/typed/divide_by.hpp>
#include <complex>

using Catch::Matchers::WithinAbs;

// -- Scalar functors -----------------------------------------------------

TEST_CASE("scalar::plus", "[functor][scalar]") {
    mtl::functor::scalar::plus<double> f;
    REQUIRE(f(3.0, 4.0) == 7.0);
    REQUIRE(mtl::functor::scalar::plus<int>::apply(2, 3) == 5);
}

TEST_CASE("scalar::minus", "[functor][scalar]") {
    mtl::functor::scalar::minus<double> f;
    REQUIRE(f(10.0, 3.0) == 7.0);
}

TEST_CASE("scalar::times", "[functor][scalar]") {
    mtl::functor::scalar::times<double> f;
    REQUIRE(f(3.0, 4.0) == 12.0);
}

TEST_CASE("scalar::divide", "[functor][scalar]") {
    mtl::functor::scalar::divide<double> f;
    REQUIRE(f(12.0, 4.0) == 3.0);
}

TEST_CASE("scalar::assign", "[functor][scalar]") {
    double a = 0.0;
    mtl::functor::scalar::assign<double> f;
    f(a, 42.0);
    REQUIRE(a == 42.0);
}

TEST_CASE("scalar::negate", "[functor][scalar]") {
    mtl::functor::scalar::negate<double> f;
    REQUIRE(f(5.0) == -5.0);
    REQUIRE(f(-3.0) == 3.0);
}

TEST_CASE("scalar::abs", "[functor][scalar]") {
    mtl::functor::scalar::abs<double> f;
    REQUIRE(f(-5.0) == 5.0);
    REQUIRE(f(3.0) == 3.0);
}

TEST_CASE("scalar::abs for complex", "[functor][scalar]") {
    using cd = std::complex<double>;
    mtl::functor::scalar::abs<cd> f;
    REQUIRE_THAT(f(cd(3.0, 4.0)), WithinAbs(5.0, 1e-10));
}

TEST_CASE("scalar::conj for real is identity", "[functor][scalar]") {
    mtl::functor::scalar::conj<double> f;
    REQUIRE(f(5.0) == 5.0);
}

TEST_CASE("scalar::conj for complex", "[functor][scalar]") {
    using cd = std::complex<double>;
    mtl::functor::scalar::conj<cd> f;
    auto r = f(cd(1.0, 2.0));
    REQUIRE(r == cd(1.0, -2.0));
}

TEST_CASE("scalar::sqrt", "[functor][scalar]") {
    mtl::functor::scalar::sqrt<double> f;
    REQUIRE_THAT(f(9.0), WithinAbs(3.0, 1e-10));
    REQUIRE_THAT(f(2.0), WithinAbs(std::sqrt(2.0), 1e-10));
}

// -- Mixed-type scalar functors ------------------------------------------

TEST_CASE("scalar::plus with mixed types", "[functor][scalar]") {
    mtl::functor::scalar::plus<int, double> f;
    auto r = f(2, 3.5);
    STATIC_REQUIRE(std::is_same_v<decltype(r), double>);
    REQUIRE(r == 5.5);
}

// -- Typed functors ------------------------------------------------------

TEST_CASE("typed::scale (alpha * x)", "[functor][typed]") {
    mtl::functor::typed::scale<double> f(3.0);
    REQUIRE(f(4.0) == 12.0);
    REQUIRE(f(0.0) == 0.0);
}

TEST_CASE("typed::rscale (x * alpha)", "[functor][typed]") {
    mtl::functor::typed::rscale<double> f(5.0);
    REQUIRE(f(2.0) == 10.0);
}

TEST_CASE("typed::divide_by (x / alpha)", "[functor][typed]") {
    mtl::functor::typed::divide_by<double> f(4.0);
    REQUIRE(f(12.0) == 3.0);
}
