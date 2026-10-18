#pragma once
// MTL5 -- error-free transformations (#611)
//
// An error-free transformation turns one rounded operation into a pair
// (hi, lo) of floating-point values with hi the rounded result and
// hi + lo the EXACT result:
//
//   two_sum(a, b)            a + b == hi + lo     Knuth: 6 flops, any a, b
//   fast_two_sum(a, b)       a + b == hi + lo     Dekker: 3 flops, needs |a| >= |b|
//   split(a)                 a     == hi + lo     Veltkamp: hi and lo each fit in
//                                                 half the significand
//   two_product_dekker(a, b) a * b == hi + lo     Dekker: 17 flops, no FMA
//   two_product(a, b)        a * b == hi + lo     2 flops with an FMA
//
// These are the building blocks of Ogita, Rump and Oishi, "Accurate sum and
// dot product", SIAM J. Sci. Comput. 26(6), 2005: Sum2 and Dot2 are a loop of
// two_sum and two_product that carries the lo parts in a second accumulator,
// and SumK / DotK repeat the transformation K - 1 times. They are exact for
// binary floating point in round-to-nearest, absent overflow, and for the
// products absent underflow: when a * b is so small that its error falls below
// the smallest subnormal, lo is that error rounded.
//
// NON-FINITE. A non-finite hi gets lo = 0, so an overflowing sum or product
// is the infinity (or NaN) it would be without the transformation, not
// inf - inf = NaN.
//
// STRICT IEEE SEMANTICS. Each of these depends on every operation being
// rounded to T exactly as written:
//
//   * No -ffast-math (GCC, Clang) or /fp:fast (MSVC). Reassociation folds
//     two_sum's lo = (a - (hi - z)) + (b - z), z = hi - a, to 0. MSVC's
//     default /fp:precise is correct.
//   * No FMA contraction in split and two_product_dekker. Contracting
//     c - (c - a), with c = factor * a, into an FMA forms c - a from the
//     unrounded product, and split returns (a, 0) or close to it: hi is no
//     longer half a significand, and Dekker's product is no longer exact. GCC contracts by default (-ffp-contract=fast)
//     wherever the target has an FMA: ARM64, or x86-64 with -march=native. So
//     these two functions switch contraction off locally: a GCC optimize region
//     and a Clang `fp contract(off)` pragma below. GCC honours its region
//     under any -ffp-contract; Clang honours the pragma under its default
//     (-ffp-contract=on) and -ffp-contract=fast-honor-pragmas, but plain
//     -ffp-contract=fast overrides pragmas in Clang, and is not supported. MSVC 2022 and later contract only under /fp:contract (or
//     /fp:fast); older versions could contract under /fp:precise with
//     /arch:AVX2, and need /fp:strict there.
//     two_sum and fast_two_sum have no products to contract; two_product uses
//     an explicit std::fma.
//   * No x87 extended-precision intermediates (32-bit x86 without SSE2).
//
// These functions are why math::splits_product_terms_v can feed a three-factor
// term to an exact accumulator as two products (#602): see accumulator_traits.hpp.

#include <cmath>
#include <concepts>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace mtl::math {

/// Binary IEEE floating point: the types the error-free transformations hold for.
template <typename T>
concept binary_floating_point =
    std::floating_point<T> && std::numeric_limits<T>::is_iec559 &&
    std::numeric_limits<T>::radix == 2;

/// The exact result of one operation as `hi + lo`: `hi` the rounded result,
/// `lo` its rounding error.
template <typename T>
struct exact_pair {
    T hi;
    T lo;
};

/// The name `two_product` returned before the other transformations (#602).
template <typename T>
using split_product = exact_pair<T>;

/// Knuth's TwoSum: `a + b == hi + lo` for any `a`, `b`, with `hi = fl(a + b)`.
template <typename T>
    requires binary_floating_point<T>
exact_pair<T> two_sum(T a, T b) {
    const T x = a + b;
    if (!std::isfinite(x)) return {x, T(0)};
    const T z = x - a;
    return {x, (a - (x - z)) + (b - z)};
}

/// Dekker's FastTwoSum: `a + b == hi + lo`, provided `|a| >= |b|` (more
/// precisely, exponent(a) >= exponent(b), or a == 0). Not checked: the caller
/// orders the operands, which is what makes it half the cost of `two_sum`.
template <typename T>
    requires binary_floating_point<T>
exact_pair<T> fast_two_sum(T a, T b) {
    const T x = a + b;
    if (!std::isfinite(x)) return {x, T(0)};
    return {x, (a - x) + b};
}

/// The error-free product with an FMA: `a * b == hi + lo`, `hi = fl(a * b)`.
template <typename T>
    requires std::is_floating_point_v<T>
exact_pair<T> two_product(T a, T b) {
    const T p = a * b;
    if (!std::isfinite(p)) return {p, T(0)};
    return {p, std::fma(a, b, -p)};
}

namespace error_free_detail {

/// Veltkamp's splitting point: s = ceil(digits / 2), factor = 2^s + 1.
template <typename T>
inline constexpr int split_shift = (std::numeric_limits<T>::digits + 1) / 2;

template <typename T>
inline constexpr T split_factor = T((std::uint64_t(1) << split_shift<T>) + 1);

template <typename T>
constexpr T pow2(int e) {
    T r = 1;
    for (; e > 0; --e) r *= 2;
    return r;
}

/// The end of split's domain, 2^(max_exponent - s - 1): factor * a is at most
/// 2^(max_exponent - 1) + 2^(max_exponent - s - 1), so it cannot overflow.
/// (max / factor itself rounds up in double, and factor times it overflows.)
template <typename T>
inline constexpr T split_limit = pow2<T>(std::numeric_limits<T>::max_exponent - split_shift<T> - 1);

} // namespace error_free_detail

// split and two_product_dekker must not be contracted into FMAs (header).
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC push_options
#pragma GCC optimize("fp-contract=off")
#endif

/// Veltkamp's Split: `a == hi + lo` exactly, with `hi` at most `digits - s`
/// significant bits and `lo` at most `s - 1` (s = ceil(digits / 2)): 26 and 26
/// for double, 12 and 11 for float. So a product of two halves is exact in T.
/// Domain: |a| <= 2^(max_exponent - s - 1) (`split_limit`: 2^996 for double,
/// 2^115 for float), where `factor * a` cannot overflow; far enough above it
/// hi is NaN, as in the paper. (Splitting a scaled-down
/// copy does not rescue the top of the range: there hi can round up to
/// 2^max_exponent.) `two_product_dekker` rescales its operands into the domain.
template <typename T>
    requires binary_floating_point<T>
exact_pair<T> split(T a) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    if (!std::isfinite(a)) return {a, T(0)};
    const T c = error_free_detail::split_factor<T> * a;
    const T hi = c - (c - a);
    return {hi, a - hi};
}

/// Dekker's TwoProduct, without an FMA: `a * b == hi + lo`, `hi = fl(a * b)`.
/// The same pair as `two_product` wherever both are exact; this is the form
/// of the Ogita-Rump-Oishi paper, and the one for a target without an FMA.
/// An operand beyond `split`'s domain is scaled down by 2^(s+1) and the other
/// up by 2^(s+1): both exact, the product unchanged. (Both beyond it, the product
/// overflows.) A product above 2^(max_exponent - 1) is formed at half scale, so
/// no intermediate overflows either: exact for every finite product that does
/// not underflow, `max * 1` included.
template <typename T>
    requires binary_floating_point<T>
exact_pair<T> two_product_dekker(T a, T b) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    using namespace error_free_detail;
    const T x = a * b;
    if (!std::isfinite(x)) return {x, T(0)};
    constexpr int k = split_shift<T> + 1;   // max / 2^(s+1) < split_limit
    if (std::fabs(a) > split_limit<T>) {
        a = std::ldexp(a, -k);
        b = std::ldexp(b, k);
    } else if (std::fabs(b) > split_limit<T>) {
        a = std::ldexp(a, k);
        b = std::ldexp(b, -k);
    }
    // Within a factor (1 + 2^-(digits - s))^2 of max, a1 * b1 -- the halves
    // round away from zero -- can overflow although x does not (max * 1).
    // Then take the error of x / 2 from the larger operand halved: both
    // scalings by 2 are exact, and so is doubling the error back.
    const bool halve = std::fabs(x) > pow2<T>(std::numeric_limits<T>::max_exponent - 1);
    if (halve) {
        if (std::fabs(a) >= std::fabs(b)) a = std::ldexp(a, -1);
        else                              b = std::ldexp(b, -1);
    }
    const T xs = halve ? std::ldexp(x, -1) : x;
    const auto [a1, a2] = split(a);
    const auto [b1, b2] = split(b);
    const T y = a2 * b2 - (((xs - a1 * b1) - a2 * b1) - a1 * b2);
    return {x, halve ? std::ldexp(y, 1) : y};
}

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC pop_options
#endif

} // namespace mtl::math
