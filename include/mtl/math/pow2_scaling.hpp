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
// multiplications by 2 and 1/2, which a number library can replace by
// specializing the trait, for example with its own exact ldexp.

#include <cmath>
#include <concepts>
#include <type_traits>

#include <mtl/concepts/magnitude.hpp>

namespace mtl::math {

/// exponent(m): the e with 2^e <= m < 2^(e+1), for a positive finite m.
/// scale(x, k): x * 2^k, exact whenever the result is a normal value of T.
///
/// The fallback finds e by halving or doubling m toward [1, 2), which rounds
/// nothing in a binary type, at a cost of |e| multiplications. scale() forms
/// 2^k by the same steps from 1 and multiplies x by it once, so a tapered
/// type rounds the result once. Where 2^k itself is not a value of T (2^1074
/// in an IEEE-like double, which only scales subnormals), x is stepped
/// instead.
template <typename T>
struct pow2_scaling {
    static int exponent(T m) {
        const T one(1), two(2), half(0.5);
        int e = 0;
        while (m >= two) {
            const T next = m * half;
            if (!(next < m)) break;      // not finite
            m = next;
            ++e;
        }
        while (m < one) {
            const T next = m * two;
            if (!(next > m)) break;      // zero
            m = next;
            --e;
        }
        return e;
    }

    static T scale(T x, int k) {
        if (k == 0) return x;
        const T factor = k > 0 ? T(2) : T(0.5);
        const T inverse = k > 0 ? T(0.5) : T(2);
        const int n = k > 0 ? k : -k;
        // 2^k by n exact steps from 1. A step is exact when undoing it gives
        // back what it started from; overflow, saturation and underflow do not.
        T p(1);
        bool exact = true;
        for (int i = 0; i < n && exact; ++i) {
            const T next = p * factor;
            exact = (next * inverse == p);
            p = next;
        }
        if (exact) return x * p;
        for (int i = 0; i < n; ++i) x = x * factor;
        return x;
    }
};

template <std::floating_point T>
struct pow2_scaling<T> {
    static int exponent(T m) { return std::ilogb(m); }
    static T scale(T x, int k) { return std::ldexp(x, k); }
};

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
