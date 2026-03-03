#pragma once
// MTL5 — Magnitude trait + concept (replaces MTL4 Magnitude pseudo-concept)
#include <type_traits>
#include <complex>
#include <cmath>

namespace mtl {

/// Primary template: magnitude type is the type itself
template <typename T>
struct magnitude_trait {
    using type = T;
};

/// Specialization for std::complex
template <typename T>
struct magnitude_trait<std::complex<T>> {
    using type = T;
};

/// Alias for the magnitude type of T
template <typename T>
using magnitude_t = typename magnitude_trait<T>::type;

/// Concept: a type has a well-defined magnitude
template <typename T>
concept Magnitude = requires(T a) {
    { abs(a) } -> std::convertible_to<magnitude_t<T>>;
};

} // namespace mtl
