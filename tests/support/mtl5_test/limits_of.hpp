#pragma once
// mtl5_test::limits_of<T, N>: std::numeric_limits<N>, with T's values.
//
// MTL5's Scalar concept requires every element type to specialize
// std::numeric_limits (#838). A test's stand-in class type over a native N
// (a double in a struct, a float with a narrow range, ...) gets its
// specialization in one line:
//
//     template <> class std::numeric_limits<my_type>
//         : public mtl5_test::limits_of<my_type, double> {};
//
// T must be constructible from N. The constants are N's, except is_iec559,
// which is false; override a member in the specialization to model a narrower
// type.
#include <limits>

namespace mtl5_test {

template <typename T, typename N>
class limits_of {
    using L = std::numeric_limits<N>;

public:
    static constexpr bool is_specialized = true;
    static constexpr bool is_signed = L::is_signed;
    static constexpr bool is_integer = L::is_integer;
    static constexpr bool is_exact = L::is_exact;
    static constexpr bool has_infinity = L::has_infinity;
    static constexpr bool has_quiet_NaN = L::has_quiet_NaN;
    static constexpr bool has_signaling_NaN = L::has_signaling_NaN;
    static constexpr bool has_denorm_loss = false;
    static constexpr std::float_round_style round_style = L::round_style;
    // A class stand-in is not an IEEE hardware type: claiming is_iec559 would
    // send it down paths that assume one (math/pow2_scaling.hpp).
    static constexpr bool is_iec559 = false;
    static constexpr bool is_bounded = L::is_bounded;
    static constexpr bool is_modulo = L::is_modulo;
    static constexpr int digits = L::digits;
    static constexpr int digits10 = L::digits10;
    static constexpr int max_digits10 = L::max_digits10;
    static constexpr int radix = L::radix;
    static constexpr int min_exponent = L::min_exponent;
    static constexpr int min_exponent10 = L::min_exponent10;
    static constexpr int max_exponent = L::max_exponent;
    static constexpr int max_exponent10 = L::max_exponent10;
    static constexpr bool traps = L::traps;
    static constexpr bool tinyness_before = L::tinyness_before;

    static constexpr T min() noexcept { return T(L::min()); }
    static constexpr T max() noexcept { return T(L::max()); }
    static constexpr T lowest() noexcept { return T(L::lowest()); }
    static constexpr T epsilon() noexcept { return T(L::epsilon()); }
    static constexpr T round_error() noexcept { return T(L::round_error()); }
    static constexpr T infinity() noexcept { return T(L::infinity()); }
    static constexpr T quiet_NaN() noexcept { return T(L::quiet_NaN()); }
    static constexpr T signaling_NaN() noexcept { return T(L::signaling_NaN()); }
    static constexpr T denorm_min() noexcept { return T(L::denorm_min()); }
};

} // namespace mtl5_test
