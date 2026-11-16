#pragma once
// MTL5 -- Type projection and embedding for mixed-precision workflows
//
// project_onto<Target>(source): wider -> narrower (lossy)
//   Converts each element from a wider type to a narrower type.
//   Mathematical analogy: projecting from a higher-dimensional
//   representational space onto a lower-dimensional subspace.
//
// embed_into<Target>(source): narrower -> wider (lossless)
//   Converts each element from a narrower type to a wider type.
//   The value is exactly representable in the target type.
//   Mathematical analogy: embedding into a higher-dimensional space.
//
// Compile-time directional enforcement via concepts:
//   project_onto requires Target has fewer or equal digits than Source
//   embed_into requires Target has more or equal digits than Source
//
// Supported containers:
//   - dense_vector<T> -> dense_vector<U>
//   - dense2D<T> -> dense2D<U>

#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>
#include <mtl/concepts/scalar.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/mat/dense2D.hpp>

namespace mtl {

// ============================================================================
// Compile-time directional constraints
// ============================================================================

/// ProjectableOnto: Target has fewer (or equal) significant digits than Source.
/// This validates that project_onto is a narrowing (lossy) conversion.
template <typename Target, typename Source>
concept ProjectableOnto =
    std::numeric_limits<Target>::digits <= std::numeric_limits<Source>::digits;

/// EmbeddableInto: Target has more (or equal) significant digits than Source.
/// This validates that embed_into is a widening (lossless) conversion.
template <typename Target, typename Source>
concept EmbeddableInto =
    std::numeric_limits<Target>::digits >= std::numeric_limits<Source>::digits;

// ============================================================================
// Saturating cast for integer targets
// ============================================================================

namespace detail {

/// The integer types saturating_cast clamps: the integral types, and
/// __int128 / unsigned __int128 where the compiler has them. Under strict
/// -std=c++20 (MTL5's own build) std::is_integral is false for those, so a
/// cast from __int128 skipped the clamp and truncated (issue 865). Their
/// signedness and limits are given here too: with MSVC's standard library
/// (clang-cl) std::numeric_limits is not specialized for them, so is_signed
/// was false and min() and max() were 0 (#871, Windows Clang-CL).
template <typename T> struct sat_traits {
    static constexpr bool is_int    = std::is_integral_v<T>;
    static constexpr bool is_signed = std::numeric_limits<T>::is_signed;
    static constexpr T min() noexcept { return std::numeric_limits<T>::min(); }
    static constexpr T max() noexcept { return std::numeric_limits<T>::max(); }
};
template <typename T> struct sat_unsigned { using type = std::make_unsigned_t<T>; };
#if defined(__SIZEOF_INT128__)
__extension__ template <> struct sat_traits<unsigned __int128> {
    __extension__ typedef unsigned __int128 T;
    static constexpr bool is_int    = true;
    static constexpr bool is_signed = false;
    static constexpr T min() noexcept { return T(0); }
    static constexpr T max() noexcept { return ~T(0); }
};
__extension__ template <> struct sat_traits<__int128> {
    __extension__ typedef __int128 T;
    static constexpr bool is_int    = true;
    static constexpr bool is_signed = true;
    static constexpr T max() noexcept { return T(~static_cast<unsigned __int128>(0) >> 1); }
    static constexpr T min() noexcept { return -max() - 1; }
};
__extension__ template <> struct sat_unsigned<__int128> { __extension__ typedef unsigned __int128 type; };
#endif
template <typename T> inline constexpr bool is_sat_int_v = sat_traits<T>::is_int;
template <typename T> inline constexpr bool sat_signed_v = sat_traits<T>::is_signed;

/// a < b by value, for integral types of any signedness and width. With the
/// same signedness the usual conversions widen both to the wider type, which
/// preserves values; otherwise the signed side is tested for a negative value
/// first and compared, as unsigned, only when it is not.
template <typename A, typename B>
constexpr bool int_less(A a, B b) noexcept {
    if constexpr (sat_signed_v<A> == sat_signed_v<B>) {
        return a < b;
    } else if constexpr (sat_signed_v<A>) {
        return a < A(0) || static_cast<typename sat_unsigned<A>::type>(a) < b;
    } else {
        return b >= B(0) && a < static_cast<typename sat_unsigned<B>::type>(b);
    }
}

/// Convert a value to Target, clamping to [min, max] for integer types
/// to prevent undefined behavior from signed integer overflow.
/// Floating-point targets pass through without clamping.
///
/// Any non-integer source with numeric_limits is clamped, NaN to 0: a
/// bfloat16 or posit source was cast unclamped, because only
/// std::is_floating_point sources were (issue 887).
template <typename Target, typename Value>
Target saturating_cast(Value v) {
    if constexpr (is_sat_int_v<Target> && !is_sat_int_v<Value> &&
                  std::numeric_limits<Value>::is_specialized && !std::numeric_limits<Value>::is_integer) {
        using limit = sat_traits<Target>;
        if constexpr (std::is_floating_point_v<Value>) {
            if (std::isnan(v)) return Target{0};
        } else {
            if (v != v) return Target{0};
        }
        if (v >= static_cast<Value>(limit::max())) return limit::max();
        if (v <= static_cast<Value>(limit::min())) return limit::min();
    } else if constexpr (is_sat_int_v<Target> && is_sat_int_v<Value>) {
        // Compare by value, not in Value's type (#691): casting Target's limits
        // to Value wraps when the signedness differs, so uint32 -> int32 saw
        // INT32_MAX as a negative bound and int -> uint32 saw UINT32_MAX as -1.
        // int_less compares mixed signedness by value for every integral type.
        // It replaces std::cmp_less over a widening to (unsigned) long long,
        // which truncated __int128 (an integral type under gnu++20) before the
        // comparison (issue 865); std::cmp_less itself rejects bool, the
        // character types and __int128.
        using limit = sat_traits<Target>;
        if (int_less(limit::max(), v)) return limit::max();
        if (int_less(v, limit::min())) return limit::min();
    }
    return static_cast<Target>(v);
}

} // namespace detail

// ============================================================================
// project_onto<Target>: wider -> narrower (lossy)
// ============================================================================

/// Project a dense vector onto a narrower element type. An integer target
/// saturates, NaN to 0, as the scaled overloads do: a bare static_cast of an
/// out-of-range or NaN floating value is undefined behavior (issue 887).
template <Scalar Target, Scalar Source>
    requires ProjectableOnto<Target, Source>
vec::dense_vector<Target> project_onto(const vec::dense_vector<Source>& src) {
    vec::dense_vector<Target> dst(src.size());
    for (std::size_t i = 0; i < src.size(); ++i)
        dst(i) = detail::saturating_cast<Target>(src(i));
    return dst;
}

/// Scaled projection of a dense vector with saturating conversion.
/// dst(i) = saturating_cast<Target>(scale * src(i))
/// The scale factor maps the source dynamic range onto the target dynamic range,
/// essential for quantization workflows (e.g., float -> int8 with scale = 127).
/// Integer targets are clamped to [min, max] to prevent overflow UB.
template <Scalar Target, Scalar Source, Scalar ScaleType>
    requires ProjectableOnto<Target, Source>
vec::dense_vector<Target> project_onto(const vec::dense_vector<Source>& src, ScaleType scale) {
    vec::dense_vector<Target> dst(src.size());
    for (std::size_t i = 0; i < src.size(); ++i)
        dst(i) = detail::saturating_cast<Target>(scale * src(i));
    return dst;
}

/// Project a dense 2D matrix onto a narrower element type; integer targets
/// saturate (see the vector overload).
template <Scalar Target, Scalar Source>
    requires ProjectableOnto<Target, Source>
mat::dense2D<Target> project_onto(const mat::dense2D<Source>& src) {
    mat::dense2D<Target> dst(src.num_rows(), src.num_cols());
    for (std::size_t r = 0; r < src.num_rows(); ++r)
        for (std::size_t c = 0; c < src.num_cols(); ++c)
            dst(r, c) = detail::saturating_cast<Target>(src(r, c));
    return dst;
}

/// Scaled projection of a dense 2D matrix with saturating conversion.
template <Scalar Target, Scalar Source, Scalar ScaleType>
    requires ProjectableOnto<Target, Source>
mat::dense2D<Target> project_onto(const mat::dense2D<Source>& src, ScaleType scale) {
    mat::dense2D<Target> dst(src.num_rows(), src.num_cols());
    for (std::size_t r = 0; r < src.num_rows(); ++r)
        for (std::size_t c = 0; c < src.num_cols(); ++c)
            dst(r, c) = detail::saturating_cast<Target>(scale * src(r, c));
    return dst;
}

// ============================================================================
// embed_into<Target>: narrower -> wider (lossless)
// ============================================================================

/// Embed a dense vector into a wider element type.
template <Scalar Target, Scalar Source>
    requires EmbeddableInto<Target, Source>
vec::dense_vector<Target> embed_into(const vec::dense_vector<Source>& src) {
    vec::dense_vector<Target> dst(src.size());
    for (std::size_t i = 0; i < src.size(); ++i)
        dst(i) = static_cast<Target>(src(i));
    return dst;
}

/// Scaled embedding of a dense vector with saturating conversion.
/// Typically used with the inverse of the projection scale for dequantization
/// (e.g., int8 -> float with scale = 1.0/127).
template <Scalar Target, Scalar Source, Scalar ScaleType>
    requires EmbeddableInto<Target, Source>
vec::dense_vector<Target> embed_into(const vec::dense_vector<Source>& src, ScaleType scale) {
    vec::dense_vector<Target> dst(src.size());
    for (std::size_t i = 0; i < src.size(); ++i)
        dst(i) = detail::saturating_cast<Target>(scale * src(i));
    return dst;
}

/// Embed a dense 2D matrix into a wider element type.
template <Scalar Target, Scalar Source>
    requires EmbeddableInto<Target, Source>
mat::dense2D<Target> embed_into(const mat::dense2D<Source>& src) {
    mat::dense2D<Target> dst(src.num_rows(), src.num_cols());
    for (std::size_t r = 0; r < src.num_rows(); ++r)
        for (std::size_t c = 0; c < src.num_cols(); ++c)
            dst(r, c) = static_cast<Target>(src(r, c));
    return dst;
}

/// Scaled embedding of a dense 2D matrix with saturating conversion.
template <Scalar Target, Scalar Source, Scalar ScaleType>
    requires EmbeddableInto<Target, Source>
mat::dense2D<Target> embed_into(const mat::dense2D<Source>& src, ScaleType scale) {
    mat::dense2D<Target> dst(src.num_rows(), src.num_cols());
    for (std::size_t r = 0; r < src.num_rows(); ++r)
        for (std::size_t c = 0; c < src.num_cols(); ++c)
            dst(r, c) = detail::saturating_cast<Target>(scale * src(r, c));
    return dst;
}

} // namespace mtl
