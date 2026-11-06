#pragma once
// MTL5 -- exact power-of-two scaling, and the Givens rotation built on it (#671)
//
// A Givens rotation formed as r = sqrt(a*a + b*b) overflows once |a| passes
// sqrt(max) and underflows below sqrt(min), so it is correct on only about
// half of a type's exponent range. givens_rotation() scales a and b by an
// exact 2^-e first, where 2^e <= max(|a|, |b|) < 2^(e+1), so the squares sit
// near 1:
//
//   a' = a 2^-e,  b' = b 2^-e,  d = sqrt(a'^2 + b'^2)
//   c = a' / d,   s = b' / d,   r = d 2^e
//
// In a binary floating-point type the scaling rounds nothing. In a tapered
// type (posit, takum) it also keeps the arithmetic where the type is most
// precise, which LAPACK's threshold-based scalings (dlapy2, dlartg) do not:
// across the 65 number types measured in the Givens and Householder study
// (docs/algorithms/givens-householder-study.md), this was the only rotation
// that stayed within a few eps of orthogonal over each type's whole range.
//
// pow2_scaling<T> is the customization point. The built-in floating-point
// types use std::ilogb and std::ldexp. Any other type gets a fallback made of
// multiplications by exact powers of two, which a number library can replace
// by specializing the trait, for example with its own exact ldexp.

#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <vector>

#include <mtl/concepts/magnitude.hpp>

namespace mtl::detail {

/// T has an explicit or implicit conversion to double.
template <typename T>
concept converts_to_double = requires(const T& t) { static_cast<double>(t); };

} // namespace mtl::detail

namespace mtl::math {

/// exponent(m): the e with 2^e <= m < 2^(e+1), for a positive finite m.
/// scale(x, k): x * 2^k, exact whenever 2^k and the result are values of T.
///
/// The fallback works with two ladders of exact powers of two, 2^(2^i) and
/// 2^-(2^i), each rung the square of the one before, kept while the square is
/// exact, and built once per type. So both functions cost O(log |e|)
/// multiplications, not |e| (#680).
///
/// A tapered type (posit, takum) is the hard case: near maxpos and minpos its
/// values can lie farther apart than the largest exact rung, so m / 2^(2^i)
/// rounds, and can round back to m. A search that steps m toward [1, 2) by
/// powers of two, a binade or a rung at a time, then stalls or drifts (#680).
/// So a T that converts to double reads e from that conversion, which is
/// exact for any value inside double's range; the rungs only bring m into
/// that range, which a type with a wider range (cfloat<256,19>) needs, and
/// there the steps are exact. A T that does not convert to double is stepped
/// by the rungs to [1, 2), and a step that does not move m ends the search,
/// so it always terminates; it is exact where every step lands on a value of
/// T, as in a binary floating-point type. A type with coarser ends than its
/// rungs and no conversion should specialize the trait.
///
/// scale() multiplies x by 2^k once, so a tapered type rounds the result
/// once: 2^k taken from double where T converts and 2^k is a value of T,
/// otherwise formed from the ladder. Where 2^k itself is not a value of T
/// (2^1074 in an IEEE-like double, which only scales subnormals), x is
/// stepped by the rungs instead, largest first.
template <typename T>
struct pow2_scaling {
    static int exponent(T m) {
        const T zero(0), two(2);
        if (!(m > zero) || !(m - m == zero)) return 0;   // not positive and finite
        const auto& up = ladder(true);
        int e = 0;
        if constexpr (mtl::detail::converts_to_double<T>) {
            constexpr double hi = 0x1p1000, lo = 0x1p-1000;
            auto inside = [&](const T& y) {
                const double d = static_cast<double>(y);
                return d < hi && d > lo;
            };
            for (std::size_t i = up.size(); i-- > 0 && !inside(m);) {
                const int step = 1 << i;
                while (!(static_cast<double>(m) < hi)) {
                    const T next = m / up[i];
                    if (!(next < m) || !(static_cast<double>(next) > lo)) break;
                    m = next;
                    e += step;
                }
                while (!(static_cast<double>(m) > lo)) {
                    const T next = m * up[i];
                    if (!(m < next) || !(static_cast<double>(next) < hi)) break;
                    m = next;
                    e -= step;
                }
            }
            if (inside(m)) {
                const double d = static_cast<double>(m);
                int k = std::ilogb(d);
                // A T more precise than double can round up to 2^k from below.
                if (d == std::ldexp(1.0, k) && m < T(d)) --k;
                return e + k;
            }
        }
        for (std::size_t i = up.size(); i-- > 0;) {
            const int step = 1 << i;
            while (!(m < up[i])) {           // m >= 2^step
                const T next = m / up[i];
                if (!(next < m)) break;      // the step rounded back to m
                m = next;
                e += step;
            }
        }
        for (std::size_t i = up.size(); i-- > 0;) {
            const int step = 1 << i;
            while (true) {                   // m < 2^(1 - step)
                const T next = m * up[i];
                if (!(next < two) || !(m < next)) break;
                m = next;
                e -= step;
            }
        }
        return e;
    }

    static T scale(T x, int k) {
        if (k == 0) return x;
        if constexpr (mtl::detail::converts_to_double<T>) {
            if (k >= -1022 && k <= 1023) {
                const double pk = std::ldexp(1.0, k);
                const T p(pk);
                if (static_cast<double>(p) == pk) return x * p;
            }
        }
        const auto& rung = ladder(k > 0);
        const long long n = k > 0 ? static_cast<long long>(k) : -static_cast<long long>(k);
        // 2^k from the rungs, exact while undoing each step gives back what
        // it started from; overflow, saturation and underflow do not.
        T p(1);
        long long rest = n;
        bool exact = true;
        for (std::size_t i = rung.size(); i-- > 0 && exact;) {
            const long long step = 1LL << i;
            while (rest >= step) {
                const T next = p * rung[i];
                if (!(next / rung[i] == p)) { exact = false; break; }
                p = next;
                rest -= step;
            }
        }
        if (exact && rest == 0) return x * p;
        rest = n;
        for (std::size_t i = rung.size(); i-- > 0;) {
            const long long step = 1LL << i;
            while (rest >= step) {
                x = x * rung[i];
                rest -= step;
            }
        }
        return x;
    }

private:
    /// 2^(2^i) (up) or 2^-(2^i) (down) for i = 0, 1, ...: each rung the square
    /// of the one before, kept while the square is exact and moves away from
    /// 1. At most 31 rungs, so a step 2^i fits in an int.
    static const std::vector<T>& ladder(bool up) {
        static const std::vector<T> ladders[2] = {build(false), build(true)};
        return ladders[up ? 1 : 0];
    }

    static std::vector<T> build(bool up) {
        std::vector<T> r{up ? T(2) : T(0.5)};
        while (r.size() < 31) {
            const T p = r.back();
            const T q = p * p;
            const bool farther = up ? (q > p) : (q < p && q > T(0));
            if (!farther || !(q - q == T(0)) || !(q / p == p)) break;
            r.push_back(q);
        }
        return r;
    }
};

template <std::floating_point T>
struct pow2_scaling<T> {
    static int exponent(T m) { return std::ilogb(m); }
    static T scale(T x, int k) { return std::ldexp(x, k); }
};

namespace detail {

/// T is a binary IEEE-754 format, as its numeric_limits declare (#697). For
/// such a T, scaling by 2^-e is exact, so wherever a*a + b*b neither overflows
/// nor loses a term to underflow, the scaled rotation and the plain formula
/// give the same r, and givens_rotation can skip the scaling. They give the
/// same c and s too, except where c or s is itself subnormal: there the scaled
/// form rounds a 2^-e or b 2^-e before it divides, and the plain form rounds
/// the quotient once, so the plain result is the correctly rounded one. This is a property of
/// the format, not of the built-in types: a user-defined type that declares
/// is_iec559 gets the same fast path. A tapered type (posit, takum) does not
/// declare it, and keeps the scaled path, which keeps its arithmetic near 1,
/// where it is most precise.
template <typename T>
concept binary_ieee_limits = std::numeric_limits<T>::is_specialized &&
                             std::numeric_limits<T>::is_iec559 &&
                             std::numeric_limits<T>::radix == 2;

/// The magnitudes m = max(|a|, |b|) for which the plain formula is safe, as
/// the open interval (lo, hi):
///   hi = sqrt(max) / 4, so a*a + b*b <= 2 m^2 stays below max;
///   lo = 4 sqrt(min / eps), so m^2 > 16 min / eps, and any b*b too small to
///   be normal is below half an ulp of a*a and drops out of the sum exactly as
///   it does in the scaled form.
/// The factors of 4 cover the rounding of the square roots. Namespace-scope
/// constants, so a call pays no initialization guard. Read during another
/// translation unit's static initialization they may still be 0, which empties
/// the interval and sends every call down the scaled path, still correct.
template <typename T>
inline const T plain_givens_lo = [] {
    using std::sqrt;
    return T(4) * sqrt(std::numeric_limits<T>::min() / std::numeric_limits<T>::epsilon());
}();
template <typename T>
inline const T plain_givens_hi = [] {
    using std::sqrt;
    return sqrt(std::numeric_limits<T>::max()) / T(4);
}();

} // namespace detail

/// A plane rotation [c s; -s c] with c a + s b = r and -s a + c b = 0.
template <typename T>
struct givens_parameters {
    T c, s, r;
};

/// The rotation that maps (a, b) to (r, 0), with r = sqrt(a^2 + b^2) >= 0
/// formed without avoidable overflow or underflow (see the top of this file).
/// a = b = 0 gives the identity, c = 1, s = 0, r = 0.
///
/// A complex T (one whose magnitude type is not T itself), and an a or b that
/// is not finite, keep the unscaled formula r = sqrt(a*a + b*b), c = a/r,
/// s = b/r, so their results are what they were before the scaling existed.
///
/// A binary IEEE T (detail::binary_ieee_limits) whose larger magnitude lies
/// in (plain_givens_lo, plain_givens_hi) also takes the unscaled formula, which
/// there gives the scaled form's result without its exponent extraction and
/// three scalings (#697). For double the interval is about (2^-483, 2^510).
template <typename T>
givens_parameters<T> givens_rotation(const T& a, const T& b) {
    using std::abs;
    using std::sqrt;
    const T zero(0), one(1);
    auto unscaled = [&] {
        const T r = sqrt(a * a + b * b);
        if (r == zero) return givens_parameters<T>{one, zero, zero};
        return givens_parameters<T>{a / r, b / r, r};
    };
    if constexpr (!std::is_same_v<magnitude_t<T>, T>) {
        return unscaled();
    } else {
        const T aa = abs(a), ab = abs(b);
        const T m = aa < ab ? ab : aa;
        if constexpr (detail::binary_ieee_limits<T>) {
            if (m > detail::plain_givens_lo<T> && m < detail::plain_givens_hi<T>) {
                const T r = sqrt(a * a + b * b);
                return {a / r, b / r, r};
            }
        }
        if (m == zero) return {one, zero, zero};
        if (!(m - m == zero)) return unscaled();   // inf or NaN
        const int e = pow2_scaling<T>::exponent(m);
        const T a1 = pow2_scaling<T>::scale(a, -e);
        const T b1 = pow2_scaling<T>::scale(b, -e);
        const T d = sqrt(a1 * a1 + b1 * b1);
        return {a1 / d, b1 / d, pow2_scaling<T>::scale(d, e)};
    }
}

/// sqrt(a^2 + b^2), scaled as givens_rotation() scales it.
template <typename T>
T scaled_hypot(const T& a, const T& b) {
    return givens_rotation(a, b).r;
}

} // namespace mtl::math
