#pragma once
// MTL5 -- Scalar concepts replacing MTL4 pseudo-concepts + boost::enable_if
#include <concepts>
#include <type_traits>
#include <complex>

namespace mtl {

/// A type that behaves like a scalar value (arithmetic or custom number type)
template <typename T>
concept Scalar = (std::is_arithmetic_v<T> || requires(T a, T b) {
    { a + b } -> std::convertible_to<T>;
    { a - b } -> std::convertible_to<T>;
    { a * b } -> std::convertible_to<T>;
    { -a }    -> std::convertible_to<T>;
    { T{0} };
});

/// A type that forms a mathematical field (scalars with division)
template <typename T>
concept Field = Scalar<T> && requires(T a, T b) {
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

template <typename T>
inline constexpr bool is_complex_v = is_complex<T>::value;

} // namespace mtl
