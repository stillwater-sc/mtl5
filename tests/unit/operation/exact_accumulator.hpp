#pragma once
// A test-only exact accumulator for float and double: configuration 3 without
// Universal (#586).
//
// A Kulisch accumulator: a fixed-point two's-complement integer wide enough to
// hold any product of two finite T, and the sum of up to 2^64 of them, without
// rounding. Each product is formed exactly from the operands' integer
// significands and added in at its binary position; `value<R>` rounds the
// whole sum once, to nearest-even, to R (float or double, subnormals, overflow
// to infinity included). So it is what a quire is to MTL5 -- an exact sum of
// products with a single round-out -- and it plugs in the same way, through the
// ADL hook in accumulator_traits.hpp.
//
// It is the REFERENCE as well as configuration 3: `err` below measures a
// computed result's error exactly, as |exact - computed| rounded once to double.
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

namespace exact_test {

template <typename T>
    requires std::is_floating_point_v<T>
struct exact_acc {
    using lim = std::numeric_limits<T>;
    /// Binary exponent of bit 0: the smallest product of two subnormals.
    static constexpr int lsb = 2 * (lim::min_exponent - lim::digits);
    /// Bits for the largest product, 2^(2 * max_exponent), plus 64 carry bits.
    static constexpr int bits = 2 * lim::max_exponent - lsb + 64;
    static constexpr std::size_t limbs = static_cast<std::size_t>(bits + 63) / 64;

    std::array<std::uint64_t, limbs> w;   // little-endian limbs, two's complement

    bool negative() const { return (w[limbs - 1] >> 63) != 0; }
};

namespace detail {

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

} // namespace detail

// ---- the ADL hook (accumulator_traits.hpp, adl_accumulator) ----

template <typename T>
void accumulator_clear(exact_acc<T>& a) { a.w.fill(0); }

template <typename T, typename V>
    requires std::is_floating_point_v<V> && (sizeof(V) <= sizeof(T))
void accumulator_add_product(exact_acc<T>& a, const V& m, const V& v) {
    std::uint64_t mm, mv;
    int em, ev;
    detail::decompose(static_cast<T>(m), mm, em);
    detail::decompose(static_cast<T>(v), mv, ev);
    if (mm == 0 || mv == 0) return;
    std::uint64_t hi, lo;
    detail::mul128(mm, mv, hi, lo);
    detail::add_shifted(a.w, hi, lo, em + ev - exact_acc<T>::lsb,
                        std::signbit(m) != std::signbit(v));
}

template <typename T, typename V>
    requires std::is_floating_point_v<V> && (sizeof(V) <= sizeof(T))
void accumulator_assign(exact_acc<T>& a, const V& v) {
    accumulator_clear(a);
    accumulator_add_product(a, v, V(1));
}

template <typename T>
void accumulator_merge(exact_acc<T>& a, const exact_acc<T>& b) {
    std::uint64_t carry = 0;
    for (std::size_t i = 0; i < a.w.size(); ++i) {
        const std::uint64_t s1 = a.w[i] + b.w[i];
        const std::uint64_t c1 = s1 < b.w[i];
        const std::uint64_t s2 = s1 + carry;
        carry = c1 | (s2 < carry);
        a.w[i] = s2;
    }
}

template <typename T, typename R>
R accumulator_value(const exact_acc<T>& a, std::type_identity<R>) {
    if (!a.negative()) return detail::round_out<R>(a.w, exact_acc<T>::lsb);
    auto mag = a.w;                                            // two's-complement negate
    std::uint64_t carry = 1;
    for (auto& x : mag) {
        x = ~x + carry;
        carry = (carry && x == 0) ? 1 : 0;
    }
    return -detail::round_out<R>(mag, exact_acc<T>::lsb);
}

// ---- reference helpers ----

/// A cleared exact accumulator.
template <typename T>
exact_acc<T> zero() {
    exact_acc<T> a;
    accumulator_clear(a);
    return a;
}

/// |exact - computed|, rounded once to double. A non-finite `computed` has a
/// non-finite error (inf, or NaN, which fails every `<=` it meets); it is not
/// handed to `decompose`, whose integer conversion assumes a finite value.
template <typename T, typename C>
double err(const exact_acc<T>& exact, C computed) {
    if (!std::isfinite(computed))
        return std::isnan(computed) ? std::numeric_limits<double>::quiet_NaN()
                                    : std::numeric_limits<double>::infinity();
    exact_acc<T> d = exact;
    accumulator_add_product(d, static_cast<T>(-computed), T(1));
    return std::fabs(accumulator_value(d, std::type_identity<double>{}));
}

} // namespace exact_test
