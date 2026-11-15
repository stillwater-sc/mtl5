// The scalar functors that had no test (#838), each against its std:: function,
// and abs and signum for unsigned types and NaN.
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>

#include <mtl/functor/scalar/abs.hpp>
#include <mtl/functor/scalar/acos.hpp>
#include <mtl/functor/scalar/acosh.hpp>
#include <mtl/functor/scalar/asin.hpp>
#include <mtl/functor/scalar/asinh.hpp>
#include <mtl/functor/scalar/atan.hpp>
#include <mtl/functor/scalar/atanh.hpp>
#include <mtl/functor/scalar/cbrt.hpp>
#include <mtl/functor/scalar/ceil.hpp>
#include <mtl/functor/scalar/cos.hpp>
#include <mtl/functor/scalar/cosh.hpp>
#include <mtl/functor/scalar/erf.hpp>
#include <mtl/functor/scalar/erfc.hpp>
#include <mtl/functor/scalar/exp.hpp>
#include <mtl/functor/scalar/exp2.hpp>
#include <mtl/functor/scalar/floor.hpp>
#include <mtl/functor/scalar/imag.hpp>
#include <mtl/functor/scalar/log.hpp>
#include <mtl/functor/scalar/log10.hpp>
#include <mtl/functor/scalar/log2.hpp>
#include <mtl/functor/scalar/pow.hpp>
#include <mtl/functor/scalar/round.hpp>
#include <mtl/functor/scalar/signum.hpp>
#include <mtl/functor/scalar/sin.hpp>
#include <mtl/functor/scalar/sinh.hpp>
#include <mtl/functor/scalar/tan.hpp>
#include <mtl/functor/scalar/tanh.hpp>

namespace fs = mtl::functor::scalar;

TEST_CASE("the transcendental and rounding functors match std", "[functor][scalar][838]") {
    // Read through volatile, so that both sides are run-time calls: GCC folds
    // std::cbrt(constant) with MPFR, correctly rounded, while the run-time
    // libm call may differ in the last bit.
    volatile double vx = 0.37, vy = 2.75, vz = -1.6;
    const double x = vx, y = vy, z = vz;
    CHECK(fs::acos<double>::apply(x)  == std::acos(x));
    CHECK(fs::acosh<double>::apply(y) == std::acosh(y));
    CHECK(fs::asin<double>::apply(x)  == std::asin(x));
    CHECK(fs::asinh<double>::apply(z) == std::asinh(z));
    CHECK(fs::atan<double>::apply(z)  == std::atan(z));
    CHECK(fs::atanh<double>::apply(x) == std::atanh(x));
    CHECK(fs::cbrt<double>::apply(z)  == std::cbrt(z));
    CHECK(fs::ceil<double>::apply(z)  == std::ceil(z));
    CHECK(fs::cos<double>::apply(y)   == std::cos(y));
    CHECK(fs::cosh<double>::apply(z)  == std::cosh(z));
    CHECK(fs::erf<double>::apply(x)   == std::erf(x));
    CHECK(fs::erfc<double>::apply(x)  == std::erfc(x));
    CHECK(fs::exp<double>::apply(z)   == std::exp(z));
    CHECK(fs::exp2<double>::apply(y)  == std::exp2(y));
    CHECK(fs::floor<double>::apply(z) == std::floor(z));
    CHECK(fs::log<double>::apply(y)   == std::log(y));
    CHECK(fs::log10<double>::apply(y) == std::log10(y));
    CHECK(fs::log2<double>::apply(y)  == std::log2(y));
    CHECK(fs::pow<double>::apply(y, 3.5) == std::pow(y, 3.5));
    CHECK(fs::round<double>::apply(y) == std::round(y));
    CHECK(fs::sin<double>::apply(y)   == std::sin(y));
    CHECK(fs::sinh<double>::apply(z)  == std::sinh(z));
    CHECK(fs::tan<double>::apply(x)   == std::tan(x));
    CHECK(fs::tanh<double>::apply(z)  == std::tanh(z));
    // A float argument stays float.
    STATIC_REQUIRE(std::is_same_v<decltype(fs::exp<float>::apply(1.0f)), float>);
    CHECK(fs::exp<float>::apply(0.5f) == std::exp(0.5f));
    // Complex arguments, where std has them.
    const std::complex<double> c(0.3, -0.4);
    CHECK(fs::exp<std::complex<double>>::apply(c) == std::exp(c));
    CHECK(fs::sin<std::complex<double>>::apply(c) == std::sin(c));
    CHECK(fs::log<std::complex<double>>::apply(c) == std::log(c));
}

TEST_CASE("imag: zero for a real, the imaginary part for a complex", "[functor][scalar][838]") {
    CHECK(fs::imag<double>::apply(3.5) == 0.0);
    CHECK(fs::imag<std::complex<double>>::apply({1.0, -2.0}) == -2.0);
    STATIC_REQUIRE(std::is_same_v<fs::imag<std::complex<float>>::result_type, float>);
}

TEST_CASE("abs and signum for unsigned types", "[functor][scalar][838]") {
    // Neither compiled before #838: std::abs(unsigned) is ambiguous and
    // signum's T{-1} narrows.
    CHECK(fs::abs<unsigned>::apply(7u) == 7u);
    CHECK(fs::abs<std::uint8_t>::apply(std::uint8_t{200}) == 200);
    CHECK(fs::abs<std::uint64_t>::apply(0u) == 0u);
    CHECK(fs::signum<unsigned>::apply(7u) == 1u);
    CHECK(fs::signum<unsigned>::apply(0u) == 0u);
    CHECK(fs::signum<std::uint16_t>::apply(std::uint16_t{9}) == 1);
    CHECK(fs::abs<int>::apply(-7) == 7);
    CHECK(fs::signum<int>::apply(-7) == -1);
    // bool is unsigned too (#855): signum is the value itself.
    CHECK(fs::signum<bool>::apply(true) == true);
    CHECK(fs::signum<bool>::apply(false) == false);
    STATIC_REQUIRE(fs::signum<bool>::apply(true));
}

TEST_CASE("signum of NaN is NaN", "[functor][scalar][838]") {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    CHECK(std::isnan(fs::signum<double>::apply(nan)));   // was 0 before #838
    CHECK(std::isnan(fs::signum<float>::apply(std::numeric_limits<float>::quiet_NaN())));
    CHECK(fs::signum<double>::apply(-0.5) == -1.0);
    CHECK(fs::signum<double>::apply(0.0) == 0.0);
    CHECK(fs::signum<double>::apply(std::numeric_limits<double>::infinity()) == 1.0);
}
