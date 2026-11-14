#pragma once
// MTL5 -- Magnitude trait + concept (replaces MTL4 Magnitude pseudo-concept)
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

/// Alias for the magnitude type of T, cv-qualifiers ignored (#838):
/// magnitude_t<const std::complex<double>> is double.
template <typename T>
using magnitude_t = typename magnitude_trait<std::remove_cv_t<T>>::type;

namespace detail::magnitude_adl {
using std::abs;
/// abs(a) as generic code calls it: std::abs for the built-in types, and the
/// number type's own abs found by ADL (#838: it was unqualified with no
/// `using std::abs`, so Magnitude<double> held only through C's ::abs(int)).
template <typename T>
concept has_abs = requires(T a) {
    { abs(a) } -> std::convertible_to<magnitude_t<T>>;
};
} // namespace detail::magnitude_adl

/// Concept: a type has a well-defined magnitude. An unsigned type is its own
/// magnitude: std::abs is ill-formed for unsigned types that do not promote to
/// int, so the abs functor returns them unchanged, and the concept agrees.
template <typename T>
concept Magnitude = std::is_unsigned_v<std::remove_cv_t<T>> || detail::magnitude_adl::has_abs<T>;

} // namespace mtl
