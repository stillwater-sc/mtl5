#pragma once
// MTL5 -- Vector property predicates (#244, batch 1).
// is_zero, is_finite, has_nan, has_inf, is_normalized/is_unit, is_orthogonal_to.
//
// Structural checks (is_zero) use an ABSOLUTE tolerance defaulting to 0 (exact).
// The norm-based checks (is_normalized, is_orthogonal_to) use a RELATIVE
// tolerance defaulting to a small multiple of the element epsilon, since a
// computed norm/inner-product is rarely bit-exact.
#include <cmath>
#include <concepts>
#include <limits>
#include <type_traits>

#include <mtl/concepts/vector.hpp>
#include <mtl/concepts/magnitude.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/operation/dot.hpp>

namespace mtl {

namespace detail {

// Per-element floating-point classification that degrades gracefully: real
// floating types use <cmath>; complex checks both parts; a custom type that
// provides isfinite / isnan / isinf, found by ADL as a number library provides
// them (posit's NaR, an lns or cfloat NaN), uses its own (#703); every other
// (exact, integer, or custom without them) type is treated as finite, never
// NaN and never infinite. Before #703 a custom type always took that last
// branch, so has_nan missed a NaN that the type itself could report.
namespace classify_adl {
using std::isfinite;
using std::isinf;
using std::isnan;
template <typename T> concept has_isfinite = requires(const T& x) { { isfinite(x) } -> std::convertible_to<bool>; };
template <typename T> concept has_isnan    = requires(const T& x) { { isnan(x) } -> std::convertible_to<bool>; };
template <typename T> concept has_isinf    = requires(const T& x) { { isinf(x) } -> std::convertible_to<bool>; };
template <typename T> bool call_isfinite(const T& x) { return isfinite(x); }
template <typename T> bool call_isnan(const T& x) { return isnan(x); }
template <typename T> bool call_isinf(const T& x) { return isinf(x); }
} // namespace classify_adl

template <typename T>
bool scalar_is_finite(const T& x) {
    if constexpr (std::is_floating_point_v<T>) {
        using std::isfinite; return isfinite(x);
    } else if constexpr (requires { x.real(); x.imag(); }) {
        using std::isfinite; return isfinite(x.real()) && isfinite(x.imag());
    } else if constexpr (classify_adl::has_isfinite<T>) {
        return classify_adl::call_isfinite(x);
    } else {
        return true;
    }
}
template <typename T>
bool scalar_is_nan(const T& x) {
    if constexpr (std::is_floating_point_v<T>) {
        using std::isnan; return isnan(x);
    } else if constexpr (requires { x.real(); x.imag(); }) {
        using std::isnan; return isnan(x.real()) || isnan(x.imag());
    } else if constexpr (classify_adl::has_isnan<T>) {
        return classify_adl::call_isnan(x);
    } else {
        return false;
    }
}
template <typename T>
bool scalar_is_inf(const T& x) {
    if constexpr (std::is_floating_point_v<T>) {
        using std::isinf; return isinf(x);
    } else if constexpr (requires { x.real(); x.imag(); }) {
        using std::isinf; return isinf(x.real()) || isinf(x.imag());
    } else if constexpr (classify_adl::has_isinf<T>) {
        return classify_adl::call_isinf(x);
    } else {
        return false;
    }
}

} // namespace detail

/// Default relative tolerance for the norm-based predicates: 128 * eps.
template <typename Mag>
constexpr Mag default_norm_tol() {
    return Mag(128) * std::numeric_limits<Mag>::epsilon();
}

/// Every entry is within tol of zero (default tol 0: exactly zero). Written as
/// !(abs <= tol) so a NaN entry fails the predicate rather than being accepted.
template <Vector V>
bool is_zero(const V& v, magnitude_t<typename V::value_type> tol = magnitude_t<typename V::value_type>(0)) {
    using std::abs;
    for (typename V::size_type i = 0; i < v.size(); ++i)
        if (!(abs(v(i)) <= tol)) return false;
    return true;
}

/// Every entry is finite (no NaN, no +/-Inf).
template <Vector V>
bool is_finite(const V& v) {
    for (typename V::size_type i = 0; i < v.size(); ++i)
        if (!detail::scalar_is_finite(v(i))) return false;
    return true;
}

/// At least one entry is NaN.
template <Vector V>
bool has_nan(const V& v) {
    for (typename V::size_type i = 0; i < v.size(); ++i)
        if (detail::scalar_is_nan(v(i))) return true;
    return false;
}

/// At least one entry is +/-Inf.
template <Vector V>
bool has_inf(const V& v) {
    for (typename V::size_type i = 0; i < v.size(); ++i)
        if (detail::scalar_is_inf(v(i))) return true;
    return false;
}

/// Unit 2-norm within a relative tolerance (default 128 * eps).
template <Vector V>
bool is_normalized(const V& v,
                   magnitude_t<typename V::value_type> tol =
                       default_norm_tol<magnitude_t<typename V::value_type>>()) {
    using mag_t = magnitude_t<typename V::value_type>;
    using std::abs;
    return abs(two_norm(v) - mag_t(1)) <= tol;
}

/// Alias for is_normalized.
template <Vector V>
bool is_unit(const V& v,
             magnitude_t<typename V::value_type> tol =
                 default_norm_tol<magnitude_t<typename V::value_type>>()) {
    return is_normalized(v, tol);
}

/// Numerically orthogonal: |<u,v>| <= tol * ||u|| * ||v|| (relative; default
/// tol 128 * eps). Zero-length vectors are trivially orthogonal.
template <Vector V1, Vector V2>
bool is_orthogonal_to(const V1& u, const V2& v,
                      magnitude_t<typename V1::value_type> tol =
                          default_norm_tol<magnitude_t<typename V1::value_type>>()) {
    using mag_t = magnitude_t<typename V1::value_type>;
    using std::abs;
    const mag_t threshold = tol * two_norm(u) * two_norm(v);
    return abs(dot(u, v)) <= threshold;
}

} // namespace mtl
