// mtl::detail wrapping arithmetic -- the modular helpers behind the integer
// contract in #451.
//
// Two things are being pinned here, and the first is subtler than it looks.
//
// 1. THE OPERATION MUST BE PERFORMED WIDE ENOUGH.
//
//    Casting the operands to `std::make_unsigned_t<T>` is not sufficient to
//    make the arithmetic defined. Integral promotion runs before any operator
//    and promotes anything narrower than `int` to *signed* `int`, so for
//    T = int16_t the expression `uint16_t(-1) * uint16_t(-1)` is `65535 * 65535`
//    evaluated in `int` -- which overflows, and is undefined. The unsigned cast
//    intended to prevent UB reintroduced it for every type narrower than `int`.
//    Reachable from the public API: `dot_real` over a `dense_vector<int16_t>`
//    of -1s tripped it. The helpers now multiply in
//    `common_type_t<make_unsigned_t<T>, unsigned>`, so the operation stays
//    unsigned, where overflow is defined as reduction.
//
//    The cases below therefore sweep every standard width, INCLUDING the
//    narrow ones that no mtl::simd lane type uses -- `generic_fma` is
//    instantiated on whatever element type a caller's vector holds, not on the
//    lane types, so int8_t and int16_t are live even while batch<> rejects them.
//    Run this file under -fsanitize=undefined to make the claim real.
//
// 2. `generic_fma` MAY ONLY TAKE THE MODULAR PATH WHEN EVERYTHING IS INTEGRAL.
//
//    It converts the operands to the result type before multiplying, which is
//    lossless between integers -- reduction mod 2^N is a ring homomorphism -- but
//    destructive with a floating-point operand and an integral result, where it
//    would turn 2.5 * 2.5 into 2 * 2 rather than rounding 6.25 once.
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include <mtl/detail/wrapping_arithmetic.hpp>

#include <cstdint>
#include <limits>
#include <type_traits>

namespace {

/// Reference: do it in 64-bit unsigned, then reduce to T's width. Correct for
/// 64-bit T too, since uint64 arithmetic already wraps mod 2^64.
template <typename T, typename Op>
T ref(T a, T b, Op op) {
    using U = std::make_unsigned_t<T>;
    const std::uint64_t wide = op(static_cast<std::uint64_t>(static_cast<U>(a)),
                                  static_cast<std::uint64_t>(static_cast<U>(b)));
    const std::uint64_t mask = sizeof(T) == 8
        ? ~std::uint64_t{0}
        : (std::uint64_t{1} << (8 * sizeof(T))) - 1;
    return static_cast<T>(static_cast<U>(wide & mask));
}

} // namespace

TEMPLATE_TEST_CASE("wrapping arithmetic is modular at every width", "[detail][wrapping]",
                   std::int8_t, std::uint8_t, std::int16_t, std::uint16_t,
                   std::int32_t, std::uint32_t, std::int64_t, std::uint64_t) {
    using T = TestType;
    constexpr T lo = std::numeric_limits<T>::lowest();
    constexpr T hi = std::numeric_limits<T>::max();
    const T values[] = {T(0), T(1), T(2), T(3), lo, hi, static_cast<T>(hi - 1),
                        static_cast<T>(lo + 1), static_cast<T>(-1), static_cast<T>(hi / 2)};

    for (T a : values)
        for (T b : values) {
            INFO("a=" << std::int64_t(a) << " b=" << std::int64_t(b)
                      << " width=" << (8 * sizeof(T)));
            CHECK(mtl::detail::wrap_add(a, b) == ref(a, b, [](auto x, auto y) { return x + y; }));
            CHECK(mtl::detail::wrap_sub(a, b) == ref(a, b, [](auto x, auto y) { return x - y; }));
            CHECK(mtl::detail::wrap_mul(a, b) == ref(a, b, [](auto x, auto y) { return x * y; }));
        }
}

TEST_CASE("the narrow-width promotion case, stated directly", "[detail][wrapping]") {
    // Each of these was signed-overflow UB when the multiply was performed in
    // the promoted `int` rather than in an unsigned type wide enough to hold it.
    CHECK(mtl::detail::wrap_mul<std::int16_t>(-1, -1) == 1);
    CHECK(mtl::detail::wrap_mul<std::uint16_t>(65535, 65535) == 1);
    CHECK(mtl::detail::wrap_mul<std::int16_t>(256, 256) == 0);       // 65536 mod 2^16
    CHECK(mtl::detail::wrap_mul<std::int8_t>(-1, -1) == 1);
    CHECK(mtl::detail::wrap_mul<std::int16_t>(181, 181) == 32761);   // no wrap, still exact
}

TEST_CASE("bool is not a wrapping integer", "[detail][wrapping]") {
    // std::make_unsigned_t<bool> is ill-formed, so bool must not reach the
    // modular branch -- a dense_vector<bool> would otherwise fail to compile.
    STATIC_REQUIRE_FALSE(mtl::detail::is_wrapping_integer_v<bool>);
    STATIC_REQUIRE(mtl::detail::is_wrapping_integer_v<std::int32_t>);
    STATIC_REQUIRE_FALSE(mtl::detail::is_wrapping_integer_v<double>);
    CHECK(mtl::detail::generic_fma<bool>(false, true, true) == true);
}

TEST_CASE("a bool OPERAND still takes the modular path", "[detail][wrapping]") {
    // The asymmetry is deliberate and easy to get backwards. `bool` is excluded
    // as a RESULT type, because make_unsigned_t<bool> is ill-formed. It must NOT
    // be excluded as an OPERAND: converting it to an integral result is exact
    // (0 or 1), and pushing the expression onto the plain branch instead would
    // reinstate the signed-overflow UB the helper exists to remove. Reachable as
    // dot(dense_vector<bool>, dense_vector<int32_t>), whose result type is int.
    constexpr auto big = std::numeric_limits<std::int32_t>::max();
    CHECK(mtl::detail::generic_fma<std::int32_t>(1, true, big) ==
          static_cast<std::int32_t>(static_cast<std::uint32_t>(1u) +
                                    static_cast<std::uint32_t>(big)));
    CHECK(mtl::detail::generic_fma<std::int32_t>(5, false, big) == 5);
    CHECK(mtl::detail::generic_fma<std::int32_t>(0, true, true) == 1);
}

TEST_CASE("generic_fma wraps only when every operand is integral", "[detail][wrapping]") {
    SECTION("all integral -> modular, and exact mod 2^N") {
        const auto a = static_cast<std::int32_t>(0x9E3779B9u);
        const auto b = static_cast<std::int32_t>(0x85EBCA77u);
        const auto expect = static_cast<std::int32_t>(static_cast<std::uint32_t>(
            static_cast<std::uint64_t>(static_cast<std::uint32_t>(a)) *
            static_cast<std::uint32_t>(b) + 7u));
        CHECK(mtl::detail::generic_fma<std::int32_t>(7, a, b) == expect);
    }

    SECTION("narrowing the operands to the result width is lossless") {
        // (a mod 2^32)(b mod 2^32) == (a b) mod 2^32, so an int64 operand pair
        // with an int32 result gets the same low bits either way.
        const std::int64_t a = 0x1'0000'9E37LL, b = 0x2'0000'85EBLL;
        const auto expect = static_cast<std::int32_t>(
            static_cast<std::uint32_t>(static_cast<std::uint64_t>(a) * static_cast<std::uint64_t>(b)));
        CHECK(mtl::detail::generic_fma<std::int32_t>(0, a, b) == expect);
    }

    SECTION("floating operands with an integral result keep the plain expression") {
        // Truncating first would give int(2.5) * int(2.5) == 4; the original
        // expression accumulates 6.25 and rounds once on conversion.
        CHECK(mtl::detail::generic_fma<int>(0, 2.5, 2.5) == 6);
        CHECK(mtl::detail::generic_fma<double>(1.0, 2.5, 2.5) == 7.25);
    }
}
