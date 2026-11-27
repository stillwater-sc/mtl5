#pragma once
// MTL5 -- exact power-of-two rescaling of a Krylov step (issues 955, 958)
//
// With a preconditioner scaled by s, a preconditioned direction scales as 1/s,
// and a product of two such directions (CG's p^T A p, BiCG's p~^H A p,
// BiCGSTAB's t^T t) as 1/s^2: it leaves double's range near s = 2^+-512 while
// every other quantity in the step is still in range. The solvers form that
// product as usual, and only when it is not a normal number redo it on
// 2^-e times the direction, with 2^e <= max |v_i|. A power of two scales
// without rounding, so the step is undone exactly. A normal product takes the
// usual path, bit for bit.
#include <mtl/concepts/scalar.hpp>      // is_complex_v
#include <mtl/concepts/magnitude.hpp>   // magnitude_t
#include <mtl/math/pow2_scaling.hpp>
#include <cmath>
#include <limits>

namespace mtl::itl::detail {

/// Whether |d| is a normal number: neither zero, subnormal, infinite nor NaN.
template <typename Value>
bool is_normal_magnitude(const Value& d) {
    using mag_t = magnitude_t<Value>;
    using std::abs;
    const mag_t a = abs(d);
    return a >= std::numeric_limits<mag_t>::min() && a <= std::numeric_limits<mag_t>::max();
}

/// v * 2^k, exact also where 2^k itself is not representable (a subnormal v
/// scaled up by 2^1074, which as a factor would be inf). A complex value is
/// scaled part by part.
template <typename Value>
Value by_pow2(const Value& v, int k) {
    using mag_t = magnitude_t<Value>;
    if constexpr (is_complex_v<Value>)
        return Value(math::pow2_scaling<mag_t>::scale(v.real(), k),
                     math::pow2_scaling<mag_t>::scale(v.imag(), k));
    else
        return math::pow2_scaling<Value>::scale(v, k);
}

/// The e with 2^e <= max_i |v_i| < 2^(e+1); 0 for a zero or non-finite
/// maximum (NaN cannot win the max). std::ilogb, which the floating-point
/// specialization calls, gives INT_MIN-like and INT_MAX values there, and -e
/// would overflow. A zero v then gives a zero rescaled product, the solver's
/// breakdown; an infinite one a non-finite step.
template <typename Vec>
int max_abs_exponent(const Vec& v) {
    using value_type = typename Vec::value_type;
    using mag_t = magnitude_t<value_type>;
    using std::abs;
    mag_t vmax(0);
    for (typename Vec::size_type i = 0; i < v.size(); ++i) {
        const mag_t a = abs(v(i));
        if (a > vmax) vmax = a;
    }
    return (vmax > mag_t(0) && vmax <= std::numeric_limits<mag_t>::max())
               ? math::pow2_scaling<mag_t>::exponent(vmax) : 0;
}

/// out = 2^k v, value by value.
template <typename Vec, typename Out>
void scale_by_pow2(const Vec& v, int k, Out& out) {
    for (typename Vec::size_type i = 0; i < v.size(); ++i) out(i) = by_pow2(v(i), k);
}

} // namespace mtl::itl::detail
