#pragma once
// MTL5 -- exact accumulator for binary32 and binary64 (#611)
//
// A Kulisch accumulator: a fixed-point two's-complement integer wide enough to
// hold any product of two finite T, and the sum of up to 2^64 of them, without
// rounding. Each product is formed exactly from the operands' integer
// significands and added in at its binary position; the round-out to a result
// type R rounds the whole sum ONCE, to nearest-even (subnormals, and overflow
// to infinity, included). So it is what a posit quire is to a posit, for IEEE
// float and double: configuration 3 of math/accumulator_traits.hpp without an
// external number library.
//
//   using exact = math::exact_accumulator<double>;
//   double d = mtl::dot<exact, double>(x, y);   // x . y, correctly rounded
//
// It plugs in through the ADL hook (math::adl_accumulator): the free functions
// below live next to the type, so every accumulator-aware operation in MTL5
// accepts it, and it provides accumulator_merge, so the threaded reductions do
// too -- and because the merge is exact, a parallel reduction gives the same
// bits for every partition of the terms. It is an IEEE element type with an
// exact accumulator, so math::splits_product_terms_v holds and the LDL^T paths
// feed it whole three-factor terms (#602).
//
// SIZE. binary32: 10 limbs, sizeof 88 bytes. binary64: 67 limbs, 544 bytes. Kernels
// that keep one accumulator per output entry hold n of them; see the workspace
// note in accumulator_traits.hpp.
//
// COST. Each add_product decomposes both operands, forms a 64x64 -> 128-bit
// product and adds it into three limbs, with a carry that only rarely runs
// further. It is a reference and an accuracy tool, not a fast path; an
// operation given a non-void accumulator also skips BLAS
// (interface::accumulator_allows_blas_v).
//
// NON-FINITE INPUTS follow IEEE: a NaN operand, or inf * 0, makes the sum NaN;
// an infinite product makes it that infinity; infinities of both signs make it
// NaN. The finite part is kept separately, so a finite sum is never disturbed
// by a test of the special state.
//
// Portable: only uint64 arithmetic, no __int128, no long double (which is 64,
// 80 or 128 bits depending on the platform).

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace mtl::math {

/// Is `T` a type the exact accumulator is defined for: binary32 or binary64.
template <typename T>
concept exact_accumulable =
    std::is_floating_point_v<T> && std::numeric_limits<T>::is_iec559 &&
    std::numeric_limits<T>::radix == 2 && std::numeric_limits<T>::digits <= 53;

/// Exact sum of products of `T`s (top of file). Trivially constructible: a
/// value-initialized `exact_accumulator<T>{}` is zero, and so is one cleared
/// with `accumulator_clear`.
template <typename T>
    requires exact_accumulable<T>
struct exact_accumulator {
    using lim = std::numeric_limits<T>;
    /// Binary exponent of bit 0: the smallest product of two subnormals.
    static constexpr int lsb = 2 * (lim::min_exponent - lim::digits);
    /// Bits for the largest product, 2^(2 * max_exponent), plus 64 carry bits.
    static constexpr int bits = 2 * lim::max_exponent - lsb + 64;
    static constexpr std::size_t limbs = static_cast<std::size_t>(bits + 63) / 64;

    /// Non-finite state, as bit flags: an infinity of each sign, and NaN.
    static constexpr std::uint8_t pos_inf = 1, neg_inf = 2, nan = 4;

    std::array<std::uint64_t, limbs> w;   // finite part: little-endian limbs, two's complement
    std::uint8_t special;                 // non-finite state (flags above)

    bool negative() const { return (w[limbs - 1] >> 63) != 0; }
};

namespace exact_detail {

/// |v| = m * 2^e with m odd (or v == 0, m == 0). Exact for every finite T.
template <typename T>
void decompose(T v, std::uint64_t& m, int& e) {
    m = 0;
    e = 0;
    if (v == T(0)) return;
    int ex = 0;
    const T fr = std::frexp(std::fabs(v), &ex);              // [0.5, 1)
    constexpr int p = std::numeric_limits<T>::digits;
    m = static_cast<std::uint64_t>(std::ldexp(fr, p));       // exact: < 2^p
    e = ex - p;
    const int tz = std::countr_zero(m);
    m >>= tz;
    e += tz;
}

/// 64 x 64 -> 128, as (hi, lo).
inline void mul128(std::uint64_t a, std::uint64_t b, std::uint64_t& hi, std::uint64_t& lo) {
    const std::uint64_t a0 = a & 0xffffffffu, a1 = a >> 32;
    const std::uint64_t b0 = b & 0xffffffffu, b1 = b >> 32;
    const std::uint64_t p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
    const std::uint64_t mid = (p00 >> 32) + (p01 & 0xffffffffu) + (p10 & 0xffffffffu);
    lo = (p00 & 0xffffffffu) | (mid << 32);
    hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
}

/// acc += sign * (hi:lo) * 2^off, modulo 2^(64 * limbs).
template <std::size_t L>
void add_shifted(std::array<std::uint64_t, L>& w, std::uint64_t hi, std::uint64_t lo,
                 int off, bool subtract) {
    // Spread the 128-bit value over three limbs starting at limb off / 64.
    const std::size_t li = static_cast<std::size_t>(off / 64);
    const int sh = off % 64;
    std::uint64_t v[3];
    v[0] = lo << sh;
    v[1] = sh ? (hi << sh) | (lo >> (64 - sh)) : hi;
    v[2] = sh ? hi >> (64 - sh) : 0;
    if (!subtract) {
        std::uint64_t carry = 0;
        for (std::size_t i = li; i < L; ++i) {
            const std::uint64_t add = (i - li < 3) ? v[i - li] : 0;
            if (i - li >= 3 && carry == 0) break;
            const std::uint64_t s1 = w[i] + add;
            const std::uint64_t c1 = s1 < add;
            const std::uint64_t s2 = s1 + carry;
            const std::uint64_t c2 = s2 < carry;
            w[i] = s2;
            carry = c1 | c2;
        }
    } else {
        std::uint64_t borrow = 0;
        for (std::size_t i = li; i < L; ++i) {
            const std::uint64_t sub = (i - li < 3) ? v[i - li] : 0;
            if (i - li >= 3 && borrow == 0) break;
            const std::uint64_t d1 = w[i] - sub;
            const std::uint64_t b1 = w[i] < sub;
            const std::uint64_t d2 = d1 - borrow;
            const std::uint64_t b2 = d1 < borrow;
            w[i] = d2;
            borrow = b1 | b2;
        }
    }
}

/// Round the nonnegative integer `w` times 2^lsb to R, once, to nearest-even.
template <typename R, std::size_t L>
R round_out(const std::array<std::uint64_t, L>& w, int lsb) {
    int top = -1;   // index of the leading one
    for (std::size_t i = L; i-- > 0;)
        if (w[i]) { top = static_cast<int>(i) * 64 + 63 - std::countl_zero(w[i]); break; }
    if (top < 0) return R(0);
    auto bit = [&](int i) -> std::uint64_t {
        return i < 0 ? 0 : (w[static_cast<std::size_t>(i / 64)] >> (i % 64)) & 1u;
    };
    using rl = std::numeric_limits<R>;
    const int E = top + lsb;                                   // value in [2^E, 2^(E+1))
    if (E >= rl::max_exponent) return rl::infinity();
    // Significant bits R keeps at this magnitude: digits, fewer when subnormal.
    const int emin = rl::min_exponent - 1;                     // smallest normal exponent
    const int k = E >= emin ? rl::digits : rl::digits - (emin - E);
    if (k < 0) return R(0);                                    // below half the least subnormal
    // Kept bits are top .. top-k+1; the round bit is top-k; the rest are sticky.
    std::uint64_t kept = 0;
    for (int i = top; i > top - k; --i) kept = (kept << 1) | bit(i);
    const std::uint64_t round = bit(top - k);
    bool sticky = false;
    for (int i = top - k - 1; i >= 0 && !sticky; --i) sticky = bit(i) != 0;
    if (round && (sticky || (kept & 1u))) ++kept;              // may carry to 2^k: exact
    return std::ldexp(static_cast<R>(kept), E - k + 1);        // overflows to inf correctly
}

/// May a `V` be fed into an `exact_accumulator<T>`: every V is exactly a T.
template <typename V, typename T>
concept feeds = exact_accumulable<V> &&
    std::numeric_limits<V>::digits <= std::numeric_limits<T>::digits &&
    std::numeric_limits<V>::max_exponent <= std::numeric_limits<T>::max_exponent &&
    std::numeric_limits<V>::min_exponent >= std::numeric_limits<T>::min_exponent;

} // namespace exact_detail

// ---- the ADL hook (accumulator_traits.hpp, adl_accumulator) ----

template <typename T>
void accumulator_clear(exact_accumulator<T>& a) {
    a.w.fill(0);
    a.special = 0;
}

template <typename T, typename V>
    requires exact_detail::feeds<V, T>
void accumulator_add_product(exact_accumulator<T>& a, const V& m, const V& v) {
    using A = exact_accumulator<T>;
    if (!std::isfinite(m) || !std::isfinite(v)) {
        if (std::isnan(m) || std::isnan(v) || m == V(0) || v == V(0))
            a.special |= A::nan;                               // NaN, or inf * 0
        else
            a.special |= (std::signbit(m) != std::signbit(v)) ? A::neg_inf : A::pos_inf;
        return;
    }
    std::uint64_t mm, mv;
    int em, ev;
    exact_detail::decompose(static_cast<T>(m), mm, em);
    exact_detail::decompose(static_cast<T>(v), mv, ev);
    if (mm == 0 || mv == 0) return;
    std::uint64_t hi, lo;
    exact_detail::mul128(mm, mv, hi, lo);
    exact_detail::add_shifted(a.w, hi, lo, em + ev - A::lsb,
                              std::signbit(m) != std::signbit(v));
}

template <typename T, typename V>
    requires exact_detail::feeds<V, T>
void accumulator_assign(exact_accumulator<T>& a, const V& v) {
    accumulator_clear(a);
    accumulator_add_product(a, v, V(1));
}

/// a += b, exactly: the finite parts add without rounding, the special states
/// combine. So a sum split across threads or processes and merged has the
/// bits of the sequential sum, for every partition.
template <typename T>
void accumulator_merge(exact_accumulator<T>& a, const exact_accumulator<T>& b) {
    std::uint64_t carry = 0;
    for (std::size_t i = 0; i < a.w.size(); ++i) {
        const std::uint64_t s1 = a.w[i] + b.w[i];
        const std::uint64_t c1 = s1 < b.w[i];
        const std::uint64_t s2 = s1 + carry;
        carry = c1 | (s2 < carry);
        a.w[i] = s2;
    }
    a.special |= b.special;
}

/// The single round-out, to nearest-even, of the exact sum to R (float or
/// double). An exactly zero sum is +0.
template <typename T, typename R>
    requires exact_accumulable<R>
R accumulator_value(const exact_accumulator<T>& a, std::type_identity<R>) {
    using A = exact_accumulator<T>;
    if (a.special) {
        const bool both = (a.special & A::pos_inf) && (a.special & A::neg_inf);
        if ((a.special & A::nan) || both) return std::numeric_limits<R>::quiet_NaN();
        return (a.special & A::pos_inf) ? std::numeric_limits<R>::infinity()
                                        : -std::numeric_limits<R>::infinity();
    }
    if (!a.negative()) return exact_detail::round_out<R>(a.w, A::lsb);
    auto mag = a.w;                                            // two's-complement negate
    std::uint64_t carry = 1;
    for (auto& x : mag) {
        x = ~x + carry;
        carry = (carry && x == 0) ? 1 : 0;
    }
    return -exact_detail::round_out<R>(mag, A::lsb);
}

} // namespace mtl::math
