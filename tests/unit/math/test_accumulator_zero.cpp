// MTL5 -- accumulator_traits::clear must produce zero, not a value-initialized Acc.
//
// The two agree for every built-in type, which is why this went unnoticed: a
// logarithmic number system encodes 2^0 in its all-zero bit pattern, so
// Universal's `lns{}` is 1.0, not 0. With `clear(a) { a = Acc{}; }` every sum of
// products over an lns accumulator started at 1. 5.13.0 then routed
// householder()'s default-path sum of squares through clear() (#537), which made
// QR/LQ over lns wrong outright (mtl5-python: Q orthogonality error 1.5).
//
// MTL5 stays Universal-free, so `shifted_zero` stands in for lns: it is a
// well-behaved arithmetic type in every respect EXCEPT that value-initializing
// it yields 1. Every test below passed by accident with Acc{} for double, and
// fails with Acc{} for shifted_zero.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>

#include <mtl/math/accumulator_traits.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/dot.hpp>
#include <mtl/operation/householder.hpp>

using namespace mtl;
using Catch::Matchers::WithinAbs;

namespace {

struct shifted_zero {
    double v = 1.0;                                   // like lns{}: value-init is NOT zero
    shifted_zero() = default;
    constexpr shifted_zero(double d) : v(d) {}        // shifted_zero{0} is a real zero
    explicit constexpr operator double() const { return v; }
    shifted_zero& operator+=(const shifted_zero& o) { v += o.v; return *this; }
    friend shifted_zero operator*(const shifted_zero& a, const shifted_zero& b) { return {a.v * b.v}; }
    friend shifted_zero fma(const shifted_zero& a, const shifted_zero& b, const shifted_zero& c) {
        return {std::fma(a.v, b.v, c.v)};
    }
};

} // namespace

TEST_CASE("clear() yields zero even when Acc{} does not", "[math][accumulator][zero]") {
    REQUIRE(static_cast<double>(shifted_zero{}) == 1.0);   // the premise

    using AT = math::accumulator_traits<shifted_zero, double>;
    shifted_zero a{};
    AT::clear(a);
    REQUIRE(AT::value<double>(a) == 0.0);

    AT::add_product(a, 3.0, 4.0);
    REQUIRE(AT::value<double>(a) == 12.0);                 // not 13
}

TEST_CASE("fma_accumulator starts at zero, both constructed and cleared", "[math][accumulator][zero]") {
    using FA = math::fma_accumulator<shifted_zero>;
    using AT = math::accumulator_traits<FA, double>;

    FA fresh{};
    REQUIRE(AT::value<double>(fresh) == 0.0);              // member default

    FA a{};
    AT::add_product(a, 5.0, 5.0);
    AT::clear(a);
    REQUIRE(AT::value<double>(a) == 0.0);
}

TEST_CASE("dot through a non-zero-default accumulator is exact", "[math][accumulator][zero][dot]") {
    vec::dense_vector<double> x = {1.0, 2.0, 3.0};
    vec::dense_vector<double> y = {4.0, 5.0, 6.0};
    REQUIRE(mtl::dot<shifted_zero, double>(x, y) == 32.0); // not 33
}

TEST_CASE("householder's sum of squares starts at zero", "[math][accumulator][zero][householder]") {
    // The #537 regression path: the reflector norm is built from a sum of squares
    // that goes through clear(). Reference: the default (void) path in double.
    vec::dense_vector<double> x = {3.0, 4.0, 12.0};       // ||x|| = 13
    auto [v_ref, tau_ref] = householder(x);
    auto [v, tau] = householder<shifted_zero>(x);

    REQUIRE_THAT(tau, WithinAbs(tau_ref, 1e-14));
    for (std::size_t i = 0; i < x.size(); ++i)
        REQUIRE_THAT(v(i), WithinAbs(v_ref(i), 1e-14));
}
