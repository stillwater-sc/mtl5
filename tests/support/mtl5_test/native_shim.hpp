#pragma once
// MTL5 test support -- native_shim<N>: a user-defined number type that holds a
// native float or double and behaves exactly like it, for testing that MTL5
// stays generic over the number type (#703).
//
// MTL5 is used with posits, LNS, fixpnt, takum and other types that are not
// native floating point, but its tests run on float and double. Code that only
// works for a native type (a qualified std::sqrt(x), an is_floating_point_v
// gate, a mixed `x >= 0` comparison) passes them and breaks downstream.
// native_shim is a class, so:
//
//   - std::is_floating_point_v and std::is_arithmetic_v are false for it;
//   - it converts to float or double only explicitly, so a qualified
//     std::sqrt(x), std::abs(x), std::frexp(x, &e) or std::isnan(x) does not
//     compile; the math functions are hidden friends found by ADL, the way a
//     number library provides them, and MTL5 must call them unqualified after
//     `using std::f;`;
//   - it is constructed from a native arithmetic value only explicitly, T(0)
//     or T(0.5), and has no mixed operators: `x >= 0` or `2.0 * x` does not
//     compile. This is stricter than Universal's posit, which allows both, and
//     catches code that a type without them (or with different semantics for
//     them) would break on.
//
// It holds an N, so each operation rounds exactly as N does: an algorithm run
// on native_shim<float> should give float's bits, except where MTL5
// specializes a path for native types (BLAS, LAPACK, SIMD kernels) or the
// compiler contracts the two differently.
//
// Test-only: this lives under tests/support, not include/mtl, so it is not part
// of the public API and is not installed. std::complex<native_shim<N>> is
// unspecified by the standard ([complex.numbers.general]/2) and is not
// supported; MTL5's complex paths are exercised with std::complex<float|double>.
#include <cmath>
#include <compare>
#include <limits>
#include <ostream>
#include <type_traits>

namespace mtl5_test {

template <typename N>
class native_shim {
    static_assert(std::is_floating_point_v<N>, "native_shim wraps a native floating-point type");
    N v;   // no initializer: trivially constructible, like the types it stands in for

    static constexpr native_shim of(N x) noexcept {
        native_shim s;
        s.v = x;
        return s;
    }

public:
    using native_type = N;

    native_shim() = default;

    /// Explicit only: T(0), T(1), T(0.5) and T(n) work; `T x = 0.5;` does not.
    template <typename A>
        requires std::is_arithmetic_v<A>
    constexpr explicit native_shim(A a) noexcept : v(static_cast<N>(a)) {}

    constexpr explicit operator float() const noexcept { return static_cast<float>(v); }
    constexpr explicit operator double() const noexcept { return static_cast<double>(v); }
    constexpr explicit operator long double() const noexcept { return static_cast<long double>(v); }

    /// The value as N, for tests that compare against the native computation.
    constexpr N native() const noexcept { return v; }

    // Arithmetic, between two shims only.
    friend constexpr native_shim operator+(native_shim a, native_shim b) noexcept { return of(a.v + b.v); }
    friend constexpr native_shim operator-(native_shim a, native_shim b) noexcept { return of(a.v - b.v); }
    friend constexpr native_shim operator*(native_shim a, native_shim b) noexcept { return of(a.v * b.v); }
    friend constexpr native_shim operator/(native_shim a, native_shim b) noexcept { return of(a.v / b.v); }
    friend constexpr native_shim operator-(native_shim a) noexcept { return of(-a.v); }
    friend constexpr native_shim operator+(native_shim a) noexcept { return a; }
    constexpr native_shim& operator+=(native_shim o) noexcept { v += o.v; return *this; }
    constexpr native_shim& operator-=(native_shim o) noexcept { v -= o.v; return *this; }
    constexpr native_shim& operator*=(native_shim o) noexcept { v *= o.v; return *this; }
    constexpr native_shim& operator/=(native_shim o) noexcept { v /= o.v; return *this; }

    // Comparison, with N's semantics: a NaN is unordered and unequal to itself.
    friend constexpr bool operator==(native_shim a, native_shim b) noexcept { return a.v == b.v; }
    friend constexpr std::partial_ordering operator<=>(native_shim a, native_shim b) noexcept { return a.v <=> b.v; }

    // The math MTL5 calls through `using std::f; f(x);`, as hidden friends.
#define MTL5_TEST_SHIM_UNARY(f) \
    friend native_shim f(native_shim a) { return of(static_cast<N>(std::f(a.v))); }
    MTL5_TEST_SHIM_UNARY(abs)   MTL5_TEST_SHIM_UNARY(fabs)  MTL5_TEST_SHIM_UNARY(sqrt)
    MTL5_TEST_SHIM_UNARY(cbrt)  MTL5_TEST_SHIM_UNARY(exp)   MTL5_TEST_SHIM_UNARY(exp2)
    MTL5_TEST_SHIM_UNARY(log)   MTL5_TEST_SHIM_UNARY(log2)  MTL5_TEST_SHIM_UNARY(log10)
    MTL5_TEST_SHIM_UNARY(sin)   MTL5_TEST_SHIM_UNARY(cos)   MTL5_TEST_SHIM_UNARY(tan)
    MTL5_TEST_SHIM_UNARY(asin)  MTL5_TEST_SHIM_UNARY(acos)  MTL5_TEST_SHIM_UNARY(atan)
    MTL5_TEST_SHIM_UNARY(sinh)  MTL5_TEST_SHIM_UNARY(cosh)  MTL5_TEST_SHIM_UNARY(tanh)
    MTL5_TEST_SHIM_UNARY(asinh) MTL5_TEST_SHIM_UNARY(acosh) MTL5_TEST_SHIM_UNARY(atanh)
    MTL5_TEST_SHIM_UNARY(erf)   MTL5_TEST_SHIM_UNARY(erfc)  MTL5_TEST_SHIM_UNARY(floor)
    MTL5_TEST_SHIM_UNARY(ceil)  MTL5_TEST_SHIM_UNARY(round) MTL5_TEST_SHIM_UNARY(trunc)
#undef MTL5_TEST_SHIM_UNARY
    friend native_shim pow(native_shim a, native_shim b) { return of(std::pow(a.v, b.v)); }
    friend native_shim atan2(native_shim a, native_shim b) { return of(std::atan2(a.v, b.v)); }
    friend native_shim hypot(native_shim a, native_shim b) { return of(std::hypot(a.v, b.v)); }
    friend native_shim copysign(native_shim a, native_shim b) { return of(std::copysign(a.v, b.v)); }
    friend native_shim fmin(native_shim a, native_shim b) { return of(std::fmin(a.v, b.v)); }
    friend native_shim fmax(native_shim a, native_shim b) { return of(std::fmax(a.v, b.v)); }
    friend native_shim fma(native_shim a, native_shim b, native_shim c) { return of(std::fma(a.v, b.v, c.v)); }
    friend native_shim frexp(native_shim a, int* e) { return of(std::frexp(a.v, e)); }
    friend native_shim ldexp(native_shim a, int e) { return of(std::ldexp(a.v, e)); }
    friend int ilogb(native_shim a) { return std::ilogb(a.v); }
    friend bool isnan(native_shim a) { return std::isnan(a.v); }
    friend bool isinf(native_shim a) { return std::isinf(a.v); }
    friend bool isfinite(native_shim a) { return std::isfinite(a.v); }
    friend bool signbit(native_shim a) { return std::signbit(a.v); }

    friend std::ostream& operator<<(std::ostream& os, native_shim a) { return os << a.v; }
};

using float_shim  = native_shim<float>;
using double_shim = native_shim<double>;

} // namespace mtl5_test

/// N's limits, as values of the shim.
namespace std {
template <typename N>
class numeric_limits<mtl5_test::native_shim<N>> {
    using L = std::numeric_limits<N>;
    using S = mtl5_test::native_shim<N>;

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
    static constexpr bool is_iec559 = L::is_iec559;
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

    static constexpr S min() noexcept { return S(L::min()); }
    static constexpr S max() noexcept { return S(L::max()); }
    static constexpr S lowest() noexcept { return S(L::lowest()); }
    static constexpr S epsilon() noexcept { return S(L::epsilon()); }
    static constexpr S round_error() noexcept { return S(L::round_error()); }
    static constexpr S infinity() noexcept { return S(L::infinity()); }
    static constexpr S quiet_NaN() noexcept { return S(L::quiet_NaN()); }
    static constexpr S signaling_NaN() noexcept { return S(L::signaling_NaN()); }
    static constexpr S denorm_min() noexcept { return S(L::denorm_min()); }
};
} // namespace std
