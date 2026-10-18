// MTL5 -- error-free transformations (#611).
//
// Each transformation is checked to be exact with the library's exact
// accumulator: a + b - hi - lo (or a * b - hi - lo, or a - hi - lo) summed in
// math::exact_accumulator must be exactly zero, every limb. Beyond exactness:
// hi is the rounded result, lo is below half an ulp of hi (fl(hi + lo) == hi),
// split's halves have the bit widths Dekker's product needs, Dekker's
// TwoProduct gives the FMA pair's bits, and a non-finite hi gets lo = 0.
//
// This target is built with -ffp-contract=fast (tests/unit/CMakeLists.txt) on
// GCC and Clang: wherever the target has an FMA (ARM64, or x86-64 with
// -march=native) that would contract split into the identity, unless the
// header's own contraction guard holds.
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <random>

#include <mtl/math/error_free.hpp>
#include <mtl/math/exact_accumulator.hpp>

using namespace mtl::math;

namespace {

template <typename T>
struct checker {
    exact_accumulator<T> acc{};

    void add(T v) { accumulator_add_product(acc, v, T(1)); }
    void sub(T v) { accumulator_add_product(acc, v, T(-1)); }
    void add_product(T a, T b) { accumulator_add_product(acc, a, b); }

    bool exactly_zero() const {
        if (acc.special) return false;
        for (auto limb : acc.w)
            if (limb) return false;
        return true;
    }
};

/// Significant bits of v: the width of its odd integer significand.
template <typename T>
int significant_bits(T v) {
    if (v == T(0)) return 0;
    int e = 0;
    const T f = std::frexp(std::fabs(v), &e);
    auto m = static_cast<std::uint64_t>(std::ldexp(f, std::numeric_limits<T>::digits));
    m >>= std::countr_zero(m);
    return 64 - std::countl_zero(m);
}

template <typename T>
bool same_bits(T a, T b) { return std::memcmp(&a, &b, sizeof(T)) == 0; }

/// Random values with mixed signs over 2^-range .. 2^range.
template <typename T>
struct values {
    std::mt19937 gen;
    int range;
    std::uniform_real_distribution<double> u{-1.0, 1.0};
    values(std::uint32_t seed, int r) : gen(seed), range(r) {}
    T operator()() { return T(std::ldexp(u(gen), int(u(gen) * range))); }
};

} // namespace

TEMPLATE_TEST_CASE("two_sum and fast_two_sum are exact", "[math][error_free][611]", float, double) {
    using T = TestType;
    values<T> v(1, std::numeric_limits<T>::digits + 10);   // overlapping and disjoint operands
    for (int t = 0; t < 20000; ++t) {
        const T a = v(), b = v();
        const auto s = two_sum(a, b);
        REQUIRE(same_bits(s.hi, T(a + b)));
        REQUIRE(T(s.hi + s.lo) == s.hi);
        checker<T> c;
        c.add(a); c.add(b); c.sub(s.hi); c.sub(s.lo);
        REQUIRE(c.exactly_zero());

        // FastTwoSum on ordered operands: the same pair.
        const auto f = std::fabs(a) >= std::fabs(b) ? fast_two_sum(a, b) : fast_two_sum(b, a);
        REQUIRE(same_bits(f.hi, s.hi));
        REQUIRE(same_bits(f.lo, s.lo));
    }

    // The error two_sum recovers: 2^p + 1 rounds to 2^p (a tie, to even).
    const T big = std::ldexp(T(1), std::numeric_limits<T>::digits);
    const auto s = two_sum(big, T(1));
    CHECK(s.hi == big);
    CHECK(s.lo == T(1));
    // FastTwoSum with |a| < |b| breaks its precondition: 1 + (2^p + 2) is a
    // tie that rounds to 2^p + 4, and so does 1 - (2^p + 4), so the error comes
    // out -2 where it is -1.
    CHECK(two_sum(T(1), big + T(2)).lo == T(-1));
    CHECK(fast_two_sum(big + T(2), T(1)).lo == T(-1));
    CHECK(fast_two_sum(T(1), big + T(2)).lo == T(-2));
}

TEMPLATE_TEST_CASE("split halves a significand exactly", "[math][error_free][611]", float, double) {
    using T = TestType;
    constexpr int p = std::numeric_limits<T>::digits;
    constexpr int s = (p + 1) / 2;
    values<T> v(2, std::numeric_limits<T>::max_exponent - s - 1);   // the whole domain
    for (int t = 0; t < 20000; ++t) {
        const T a = v();
        const auto [hi, lo] = split(a);
        REQUIRE(significant_bits(hi) <= p - s);
        REQUIRE(significant_bits(lo) <= s - 1);
        checker<T> c;
        c.add(a); c.sub(hi); c.sub(lo);
        REQUIRE(c.exactly_zero());
    }
    // The edge of the domain, 2^(max_exponent - s - 1): still exact.
    const T edge = std::ldexp(T(1), std::numeric_limits<T>::max_exponent - s - 1);
    REQUIRE(edge == error_free_detail::split_limit<T>);
    for (T a : {edge, -edge, std::nextafter(edge, T(0)), std::nextafter(edge, T(0)) * T(-0.75)}) {
        const auto [hi, lo] = split(a);
        CHECK(significant_bits(hi) <= p - s);
        CHECK(significant_bits(lo) <= s - 1);
        checker<T> c;
        c.add(a); c.sub(hi); c.sub(lo);
        CHECK(c.exactly_zero());
    }
}

TEMPLATE_TEST_CASE("two_product and two_product_dekker are exact and agree",
                   "[math][error_free][611]", float, double) {
    using T = TestType;
    // Exponents small enough that no product underflows past the subnormals
    // or overflows: there both are exact, so they must give the same pair.
    values<T> v(3, std::numeric_limits<T>::max_exponent / 3);
    for (int t = 0; t < 20000; ++t) {
        const T a = v(), b = v();
        const auto f = two_product(a, b);
        const auto d = two_product_dekker(a, b);
        REQUIRE(same_bits(f.hi, T(a * b)));
        REQUIRE(same_bits(d.hi, f.hi));
        REQUIRE(same_bits(d.lo, f.lo));
        REQUIRE(T(f.hi + f.lo) == f.hi);
        checker<T> c;
        c.add_product(a, b); c.sub(f.hi); c.sub(f.lo);
        REQUIRE(c.exactly_zero());
    }
    // An operand beyond split's domain, up to max itself: two_product_dekker
    // rescales the pair, and the product is still exact and the FMA pair.
    const T max = std::numeric_limits<T>::max();
    for (T a : {max, -max, std::nextafter(max, T(0)), max / T(3)}) {
        for (T b : {T(0.75), T(-0.3), T(1) / T(3), std::ldexp(T(1), -40)}) {
            const auto d = two_product_dekker(a, b);
            const auto f = two_product(a, b);
            REQUIRE(std::isfinite(d.hi));
            CHECK(same_bits(d.hi, f.hi));
            CHECK(d.lo == f.lo);   // an exact product's lo: either sign of zero
            checker<T> c;
            c.add_product(a, b); c.sub(d.hi); c.sub(d.lo);
            CHECK(c.exactly_zero());
            CHECK(two_product_dekker(b, a).lo == d.lo);   // either operand order
        }
    }
}

TEMPLATE_TEST_CASE("a non-finite result gets lo = 0", "[math][error_free][611]", float, double) {
    using T = TestType;
    const T inf = std::numeric_limits<T>::infinity();
    const T max = std::numeric_limits<T>::max();
    const T nan = std::numeric_limits<T>::quiet_NaN();

    const auto so = two_sum(max, max);
    CHECK(so.hi == inf);
    CHECK(so.lo == T(0));
    CHECK(fast_two_sum(-max, -max).hi == -inf);
    CHECK(fast_two_sum(-max, -max).lo == T(0));
    CHECK(two_sum(inf, T(1)).lo == T(0));
    CHECK(std::isnan(two_sum(inf, -inf).hi));
    CHECK(two_sum(inf, -inf).lo == T(0));

    for (auto r : {two_product(max, T(2)), two_product_dekker(max, T(2))}) {
        CHECK(r.hi == inf);
        CHECK(r.lo == T(0));
    }
    CHECK(std::isnan(two_product_dekker(nan, T(1)).hi));
    CHECK(two_product_dekker(nan, T(1)).lo == T(0));
    CHECK(split(inf).hi == inf);
    CHECK(split(inf).lo == T(0));
    CHECK(std::isnan(split(nan).hi));
}
