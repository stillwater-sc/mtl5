// MTL5 -- math::compensated_accumulator, configuration 2c: Ogita, Rump and
// Oishi's Sum2 / Dot2 as an accumulation configuration (#611).
//
// The protocol it satisfies; that it IS Dot2 -- through mtl::dot the same bits
// as the paper's loop written out with Dekker's TwoProduct and Knuth's TwoSum;
// the cancellation configurations 1 and 2 lose; a merge that keeps the errors;
// a wider initial value and a wider result; non-finite propagation. Its
// accuracy against the condition number, and the Dot2 bound, are asserted in
// the #586 matrix (operation/test_accumulation_configs_mt.cpp, [586:dot]).
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <random>
#include <type_traits>

#include <mtl/math/accumulator_traits.hpp>
#include <mtl/math/error_free.hpp>
#include <mtl/math/exact_accumulator.hpp>
#include <mtl/operation/dot.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/vec/dense_vector.hpp>

using namespace mtl;
using math::compensated_accumulator;

namespace {

template <typename T>
using AT = math::accumulator_traits<compensated_accumulator<T>, T>;

static_assert(math::mergeable_accumulator<compensated_accumulator<float>, float>);
static_assert(math::mergeable_accumulator<compensated_accumulator<double>, double>);
static_assert(!math::adl_accumulator<compensated_accumulator<double>, double>);
static_assert(std::is_same_v<math::accumulator_round_type_t<compensated_accumulator<float>, double>,
                             float>);
static_assert(sizeof(compensated_accumulator<double>) == 2 * sizeof(double));

template <typename T>
bool same_bits(T a, T b) { return std::memcmp(&a, &b, sizeof(T)) == 0; }

/// Dot2 as the paper writes it: TwoProduct (Dekker's, no FMA) and TwoSum on
/// the running sum p, the errors summed in s, the result p + s.
template <typename T>
T dot2_reference(const vec::dense_vector<T>& x, const vec::dense_vector<T>& y) {
    T p = 0, s = 0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const auto h = math::two_product_dekker(x(i), y(i));
        const auto q = math::two_sum(p, h.hi);
        p = q.hi;
        s += q.lo + h.lo;
    }
    return p + s;
}

/// x . y exactly, rounded once to T.
template <typename T>
T exact_dot(const vec::dense_vector<T>& x, const vec::dense_vector<T>& y) {
    math::exact_accumulator<T> a{};
    for (std::size_t i = 0; i < x.size(); ++i) accumulator_add_product(a, x(i), y(i));
    return accumulator_value(a, std::type_identity<T>{});
}

} // namespace

TEMPLATE_TEST_CASE("compensated accumulator: mtl::dot is the paper's Dot2, bit for bit",
                   "[math][accumulator][compensated][611]", float, double) {
    using T = TestType;
    std::mt19937 gen(11);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    for (std::size_t n : {1u, 2u, 7u, 100u, 1000u}) {
        vec::dense_vector<T> x(n), y(n);
        for (std::size_t i = 0; i < n; ++i) {
            x(i) = T(std::ldexp(u(gen), int(u(gen) * 30)));
            y(i) = T(u(gen));
        }
        INFO("n = " << n);
        // Below the threading grain, so one serial pass in every build.
        REQUIRE(same_bits(mtl::dot<compensated_accumulator<T>, T>(x, y), dot2_reference(x, y)));
    }
}

TEMPLATE_TEST_CASE("compensated accumulator: keeps what plain and FMA cancel away",
                   "[math][accumulator][compensated][611]", float, double) {
    using T = TestType;
    // (2^p, 1, -2^p) . (1, 1, 1) = 1. Plain and FMA absorb the 1 into 2^p.
    const T big = std::ldexp(T(1), std::numeric_limits<T>::digits);
    vec::dense_vector<T> x(3), y(3, T(1));
    x(0) = big; x(1) = T(1); x(2) = -big;
    CHECK(mtl::dot<T, T>(x, y) == T(0));
    CHECK(mtl::dot<math::fma_accumulator<T>, T>(x, y) == T(0));
    CHECK(mtl::dot<compensated_accumulator<T>, T>(x, y) == T(1));

    // A product's rounding error, recovered: dot([-p, a], [1, b]), p = fl(a b).
    const T a = T(1) / T(3), b = T(2) / T(3), p = a * b;
    vec::dense_vector<T> r(2), s(2);
    r(0) = -p; s(0) = T(1);
    r(1) = a;  s(1) = b;
    CHECK(mtl::dot<compensated_accumulator<T>, T>(r, s) == std::fma(a, b, -p));
}

TEMPLATE_TEST_CASE("compensated accumulator: merge keeps both partials' errors",
                   "[math][accumulator][compensated][611]", float, double) {
    using T = TestType;
    const T big = std::ldexp(T(1), std::numeric_limits<T>::digits);
    // Three partials whose rounded sums cancel: only the errors are left.
    compensated_accumulator<T> a, b, c;
    AT<T>::clear(a); AT<T>::clear(b); AT<T>::clear(c);
    AT<T>::add_product(a, big, T(1));
    AT<T>::add_product(a, T(1), T(1));      // a: sum 2^p, err 1
    AT<T>::add_product(b, T(0.5), T(1));
    AT<T>::add_product(c, -big, T(1));
    AT<T>::merge(a, b);
    AT<T>::merge(a, c);
    CHECK(AT<T>::value(a) == T(1.5));

    // A random reduction split in two and merged: within a few u of the
    // correctly rounded value, like the serial one.
    std::mt19937 gen(12);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    constexpr std::size_t n = 2000;
    vec::dense_vector<T> x(n), y(n);
    for (std::size_t i = 0; i < n; ++i) {
        x(i) = T(std::ldexp(u(gen), int(u(gen) * 20)));
        y(i) = T(u(gen));
    }
    compensated_accumulator<T> lo, hi;
    AT<T>::clear(lo); AT<T>::clear(hi);
    for (std::size_t i = 0; i < n / 3; ++i) AT<T>::add_product(lo, x(i), y(i));
    for (std::size_t i = n / 3; i < n; ++i) AT<T>::add_product(hi, x(i), y(i));
    AT<T>::merge(lo, hi);
    const T want = exact_dot(x, y);
    CHECK(std::fabs(AT<T>::value(lo) - want) <= 2 * std::numeric_limits<T>::epsilon() * std::fabs(want));
}

TEST_CASE("compensated accumulator: a wider initial value and a wider result",
          "[math][accumulator][compensated][611]") {
    // A double assigned to a float pair keeps about 48 bits; rounded out to
    // double it comes back exactly.
    using A = compensated_accumulator<float>;
    const double v = 1.0 + std::ldexp(1.0, -30);
    A a;
    math::accumulator_traits<A, double>::assign(a, v);
    CHECK(a.sum == 1.0f);
    CHECK(a.err == float(std::ldexp(1.0, -30)));
    CHECK(math::accumulator_traits<A, double>::value<double>(a) == v);
    CHECK(math::accumulator_traits<A, double>::value<float>(a) == 1.0f);

    // The pair as a double result: sum + err added in double, no rounding to float.
    A b;
    AT<float>::clear(b);
    AT<float>::add_product(b, 1.0f, 1.0f);
    AT<float>::add_product(b, std::ldexp(1.0f, -40), 1.0f);
    CHECK(AT<float>::value<double>(b) == 1.0 + std::ldexp(1.0, -40));
    CHECK(AT<float>::value<float>(b) == 1.0f);

    // The same through an operation: dot rounded out to double.
    vec::dense_vector<float> x(2), y(2, 1.0f);
    x(0) = 1.0f; x(1) = std::ldexp(1.0f, -40);
    CHECK(mtl::dot<A, double>(x, y) == 1.0 + std::ldexp(1.0, -40));
}

TEMPLATE_TEST_CASE("compensated accumulator: non-finite terms propagate",
                   "[math][accumulator][compensated][611]", float, double) {
    using T = TestType;
    using A = compensated_accumulator<T>;
    const T max = std::numeric_limits<T>::max();
    const T inf = std::numeric_limits<T>::infinity();
    A a;
    AT<T>::clear(a);
    AT<T>::add_product(a, T(1), T(1));
    AT<T>::add_product(a, max, T(2));               // overflows: lo = 0, not NaN
    CHECK(AT<T>::value(a) == inf);
    AT<T>::add_product(a, T(1), T(1));
    CHECK(AT<T>::value(a) == inf);
    AT<T>::add_product(a, -inf, T(1));
    CHECK(std::isnan(AT<T>::value(a)));

    A b;
    AT<T>::clear(b);
    AT<T>::add_product(b, std::numeric_limits<T>::quiet_NaN(), T(1));
    CHECK(std::isnan(AT<T>::value(b)));

    // two_norm rounds out to T before the sqrt, and an overflowing square is inf.
    vec::dense_vector<T> v(2, max);
    CHECK(std::isinf(mtl::two_norm<A, T>(v)));
}
