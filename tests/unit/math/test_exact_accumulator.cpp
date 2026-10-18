// MTL5 -- math::exact_accumulator, the library's exact (Kulisch) accumulator (#611).
//
// The accumulator was the test-only reference of #586; these are its own unit
// tests in the library: the protocol it satisfies, exact sums with one
// correctly rounded round-out (ties, subnormals, overflow, a narrower result
// type without double rounding), IEEE non-finite propagation, an exact merge,
// and one MTL5 operation run through it.
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include <utility>
#include <initializer_list>
#include <limits>
#include <random>
#include <type_traits>

#include <mtl/math/accumulator_traits.hpp>
#include <mtl/math/exact_accumulator.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/dot.hpp>

using mtl::math::exact_accumulator;

namespace {

template <typename T>
exact_accumulator<T> cleared() {
    exact_accumulator<T> a;
    accumulator_clear(a);
    return a;
}

template <typename R, typename T>
R value(const exact_accumulator<T>& a) { return accumulator_value(a, std::type_identity<R>{}); }

template <typename T>
bool same_bits(T a, T b) { return std::memcmp(&a, &b, sizeof(T)) == 0; }

} // namespace

using E32 = exact_accumulator<float>;
using E64 = exact_accumulator<double>;

static_assert(mtl::math::adl_accumulator<E32, float>);
static_assert(mtl::math::adl_accumulator<E64, double>);
static_assert(mtl::math::adl_accumulator<E64, float>);           // float terms, double range
static_assert(!mtl::math::adl_accumulator<E32, double>);          // a double is not a float
static_assert(mtl::math::mergeable_accumulator<E64, double>);
static_assert(mtl::math::splits_product_terms_v<E64, double>);    // #602
static_assert(std::is_trivially_default_constructible_v<E64>);
static_assert(E32::limbs == 10 && E64::limbs == 67);
static_assert(!mtl::math::exact_accumulable<long double> ||
              std::numeric_limits<long double>::digits == 53);   // MSVC: long double is double

TEST_CASE("exact_accumulator: zero, value-initialized or cleared", "[math][accumulator][611]") {
    E64 v{};
    CHECK(value<double>(v) == 0.0);
    CHECK(!std::signbit(value<double>(v)));
    CHECK(value<double>(cleared<double>()) == 0.0);
}

TEST_CASE("exact_accumulator: a single product rounds to the IEEE product", "[math][accumulator][611]") {
    std::mt19937_64 gen(1);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::uniform_int_distribution<int> ex(-200, 200);
    for (int i = 0; i < 20000; ++i) {
        const double a = std::ldexp(u(gen), ex(gen)), b = std::ldexp(u(gen), ex(gen));
        auto acc = cleared<double>();
        accumulator_add_product(acc, a, b);
        REQUIRE(same_bits(value<double>(acc), a * b));
        // Exponents within float range: an underflow to -0 would meet the documented
        // +0 of an exactly zero sum.
        const float fa = float(std::ldexp(u(gen), ex(gen) / 4)), fb = float(std::ldexp(u(gen), ex(gen) / 4));
        auto accf = cleared<float>();
        accumulator_add_product(accf, fa, fb);
        REQUIRE(same_bits(value<float>(accf), fa * fb));
    }
}

TEST_CASE("exact_accumulator: sums are exact and rounded once", "[math][accumulator][611]") {
    SECTION("cancellation keeps what plain arithmetic absorbs") {
        auto a = cleared<float>();
        accumulator_add_product(a, 16777216.0f, 1.0f);
        accumulator_add_product(a, 1.0f, 1.0f);
        accumulator_add_product(a, -16777216.0f, 1.0f);
        CHECK(value<float>(a) == 1.0f);
    }
    SECTION("an intermediate beyond the double range is held exactly") {
        auto a = cleared<double>();
        accumulator_add_product(a, 1e300, 1e300);          // 1e600: not a double
        accumulator_add_product(a, 0.5, 0.25);
        accumulator_add_product(a, -1e300, 1e300);
        CHECK(value<double>(a) == 0.125);
    }
    SECTION("ties round to even") {
        const double h = std::ldexp(1.0, -53);              // half an ulp of 1
        auto a = cleared<double>();
        accumulator_assign(a, 1.0);
        accumulator_add_product(a, h, 1.0);
        CHECK(value<double>(a) == 1.0);                     // tie, 1 is even
        accumulator_add_product(a, std::ldexp(1.0, -80), 1.0);
        CHECK(value<double>(a) == std::nextafter(1.0, 2.0)); // just above the tie
    }
    SECTION("a float result from a double-range sum is not double-rounded") {
        // 1 + 2^-24 + 2^-60: rounding to double first gives 1 + 2^-24, a float
        // tie that goes to 1; rounding once gives the float above 1.
        auto a = cleared<double>();
        accumulator_assign(a, 1.0);
        accumulator_add_product(a, std::ldexp(1.0, -24), 1.0);
        accumulator_add_product(a, std::ldexp(1.0, -60), 1.0);
        CHECK(value<float>(a) == std::nextafter(1.0f, 2.0f));
        CHECK(float(value<double>(a)) == 1.0f);
    }
    SECTION("subnormals and underflow") {
        const double tiny = std::numeric_limits<double>::denorm_min();
        auto a = cleared<double>();
        accumulator_add_product(a, tiny, 3.0);
        CHECK(value<double>(a) == 3.0 * tiny);
        auto b = cleared<double>();
        accumulator_add_product(b, tiny, tiny);             // far below the least subnormal
        CHECK(value<double>(b) == 0.0);
        accumulator_add_product(b, tiny, 1.0);
        CHECK(value<double>(b) == tiny);                    // the tiny part does not reach a tie
    }
    SECTION("the round-out overflows to infinity, signed") {
        const double big = std::numeric_limits<double>::max();
        auto a = cleared<double>();
        accumulator_add_product(a, big, 2.0);
        CHECK(value<double>(a) == std::numeric_limits<double>::infinity());
        auto b = cleared<double>();
        accumulator_add_product(b, -big, 2.0);
        CHECK(value<double>(b) == -std::numeric_limits<double>::infinity());
        accumulator_add_product(b, big, 1.5);
        CHECK(value<double>(b) == -big / 2);
    }
}

TEST_CASE("exact_accumulator: non-finite inputs follow IEEE", "[math][accumulator][611]") {
    const double inf = std::numeric_limits<double>::infinity();
    const double qnan = std::numeric_limits<double>::quiet_NaN();
    auto with = [](std::initializer_list<std::pair<double, double>> terms) {
        auto a = cleared<double>();
        accumulator_add_product(a, 1.0, 1.0);
        for (auto [m, v] : terms) accumulator_add_product(a, m, v);
        return value<double>(a);
    };
    CHECK(std::isnan(with({{qnan, 1.0}})));
    CHECK(std::isnan(with({{inf, 0.0}})));
    CHECK(with({{inf, 2.0}}) == inf);
    CHECK(with({{inf, -2.0}}) == -inf);
    CHECK(with({{inf, 2.0}, {-inf, 2.0}}) != with({{inf, 2.0}, {-inf, 2.0}}));   // NaN
    CHECK(with({{inf, 2.0}, {inf, 3.0}}) == inf);

    auto a = cleared<double>();
    accumulator_add_product(a, inf, 1.0);
    accumulator_clear(a);
    CHECK(value<double>(a) == 0.0);                         // clear resets the special state
    accumulator_assign(a, -inf);
    CHECK(value<double>(a) == -inf);
}

TEST_CASE("exact_accumulator: merge is exact, so the partition does not matter",
          "[math][accumulator][611]") {
    std::mt19937_64 gen(5);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::uniform_int_distribution<int> ex(-60, 60);
    std::vector<double> x(500), y(500);
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] = std::ldexp(u(gen), ex(gen));
        y[i] = std::ldexp(u(gen), ex(gen));
    }
    auto all = cleared<double>();
    for (std::size_t i = 0; i < x.size(); ++i) accumulator_add_product(all, x[i], y[i]);
    for (std::size_t cut : {0u, 1u, 137u, 499u}) {
        auto lo = cleared<double>(), hi = cleared<double>();
        for (std::size_t i = 0; i < x.size(); ++i)
            accumulator_add_product(i < cut ? lo : hi, x[i], y[i]);
        accumulator_merge(hi, lo);
        CHECK(hi.w == all.w);
    }
    auto n = cleared<double>(), f = cleared<double>();
    accumulator_add_product(n, std::numeric_limits<double>::infinity(), 1.0);
    accumulator_add_product(f, 3.0, 1.0);
    accumulator_merge(f, n);
    CHECK(value<double>(f) == std::numeric_limits<double>::infinity());
}

TEST_CASE("exact_accumulator: through mtl::dot, the correctly rounded dot product",
          "[math][accumulator][611]") {
    mtl::vec::dense_vector<double> x(4), y(4);
    x(0) = 1e16; x(1) = 1.0;  x(2) = -1e16; x(3) = 0.5;
    y(0) = 1.0;  y(1) = 1.0;  y(2) = 1.0;   y(3) = 1.0;
    CHECK((mtl::dot<E64, double>(x, y)) == 1.5);
    mtl::vec::dense_vector<float> xf(3), yf(3, 1.0f);
    xf(0) = 16777216.0f; xf(1) = 1.0f; xf(2) = -16777216.0f;
    CHECK((mtl::dot<E64, float>(xf, yf)) == 1.0f);           // float terms, double-range accumulator
}
