#pragma once
// MTL5 -- Scalar concepts replacing MTL4 pseudo-concepts + boost::enable_if
#include <concepts>
#include <type_traits>
#include <complex>
#include <limits>

#include <mtl/concepts/magnitude.hpp>

namespace mtl {

/// The number system describes itself through std::numeric_limits (#838): its
/// epsilon, range, infinities and NaNs. Required of the MAGNITUDE type, because
/// the standard does not specialize numeric_limits for std::complex: a
/// std::complex<T> qualifies when T does.
///
/// Every MTL5 element type must meet it. Algorithms that need the precision or
/// the range of a type read them from numeric_limits, with no per-algorithm
/// fallback for a type that lacks them: a custom number type specializes
/// std::numeric_limits, as the Universal number systems do.
template <typename T>
concept HasNumericLimits = std::numeric_limits<magnitude_t<T>>::is_specialized;

/// A type that behaves like a scalar value (arithmetic or custom number type),
/// and describes itself through std::numeric_limits.
template <typename T>
concept Scalar = HasNumericLimits<T> && (std::is_arithmetic_v<T> || requires(T a, T b) {
    { a + b } -> std::convertible_to<T>;
    { a - b } -> std::convertible_to<T>;
    { a * b } -> std::convertible_to<T>;
    { -a }    -> std::convertible_to<T>;
    { T{0} };
});

/// A type that forms a mathematical field: division is defined AND invertible.
///
/// Integral types are excluded, and that exclusion is the point. `int` satisfies
/// the syntax -- `a / b` compiles and yields an `int` -- so a purely syntactic
/// Field admitted the integers, which are a RING, not a field: 3/2*2 == 2, and
/// only 1 and -1 have inverses. Every algorithm that says `requires Field` means
/// "I will divide and expect the quotient back", and on integers it silently
/// gets a truncated one. That is how `lu_factor(dense2D<int>)` used to compile
/// and return confident nonsense.
///
/// Custom number types (posit, LNS, fixpnt, rationals) are unaffected: they are
/// not std::is_integral and their division is the operation they advertise.
template <typename T>
concept Field = Scalar<T> && !std::is_integral_v<T> && requires(T a, T b) {
    { a / b } -> std::convertible_to<T>;
};

/// An ordered field (field with comparison operators)
template <typename T>
concept OrderedField = Field<T> && std::totally_ordered<T>;

/// Detect std::complex specializations
template <typename T>
struct is_complex : std::false_type {};

template <typename T>
struct is_complex<std::complex<T>> : std::true_type {};

/// Whether T is a std::complex, cv-qualifiers ignored (#838).
template <typename T>
inline constexpr bool is_complex_v = is_complex<std::remove_cv_t<T>>::value;

} // namespace mtl
