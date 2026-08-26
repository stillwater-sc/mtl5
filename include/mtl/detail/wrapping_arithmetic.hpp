#pragma once
// MTL5 -- two's-complement wrapping integer arithmetic (#451).
//
// Integer lanes in mtl::simd are defined to WRAP: `+ - *` and `fma` return the
// exact mathematical result reduced mod 2^N. That is the native semantics of
// every target ISA and of Highway's integer ops, and it is what makes an integer
// reduction bit-identical across lane counts, backends and thread partitions
// (see simd/batch.hpp for the full contract).
//
// Plain C++ cannot express it directly. Signed integer overflow is UB, so a
// scalar tail written `s += a[i] * b[i]` over int32 is undefined exactly where
// the SIMD body it completes wraps and defines the result -- the two halves of
// one kernel disagreeing on whether an answer exists. Routing the scalar
// arithmetic through the unsigned counterpart makes both halves produce the SAME
// well-defined value: C++20 fixes two's complement, so the round trip through
// std::make_unsigned_t is exact and the narrowing cast back is modular rather
// than implementation-defined.
//
// These live here, outside simd/, because the generic element-wise loops in
// operation/ need them for the same reason the kernels do: an integer `mult` or
// `dot` that misses the SIMD fast path must still land on the documented answer
// rather than on UB. Floating-point and custom number types fall through every
// helper unchanged.
#include <cmath>
#include <type_traits>

namespace mtl::detail {

// The unsigned arithmetic below WRAPS ON PURPOSE. That is not undefined
// behavior -- unsigned overflow is defined as reduction mod 2^N, and this header
// exists to route signed arithmetic through it precisely for that guarantee.
//
// Clang's `-fsanitize=integer` nevertheless flags it, because that group bundles
// `unsigned-integer-overflow`, a lint for *accidental* wrap-around rather than a
// UB check. Left unexempted, the three helpers below trip it on their first
// overflowing multiply, which makes the whole sanitizer unusable on this project
// and hides the accidental wrap-around it would otherwise catch elsewhere. So the
// deliberate wrapping is exempted by name and everything else stays checked.
//
// Clang-only: `-fsanitize=integer` is a clang feature, and naming an unknown
// sanitizer in the attribute warns on GCC.
#if defined(__clang__)
#  define MTL5_WRAPS_ON_PURPOSE __attribute__((no_sanitize("unsigned-integer-overflow")))
#else
#  define MTL5_WRAPS_ON_PURPOSE
#endif


/// Integral types these helpers wrap.
///
/// `bool` is excluded: `std::make_unsigned_t<bool>` is ill-formed, and boolean
/// arithmetic promotes to `int` and cannot overflow, so it belongs on the plain
/// branch rather than in a modular one.
template <typename T>
inline constexpr bool is_wrapping_integer_v =
    std::is_integral_v<T> && !std::is_same_v<T, bool>;

/// The type the modular arithmetic is actually performed in: the unsigned
/// counterpart of `T`, or `unsigned int` when that is narrower than `int`.
///
/// This width promise is the whole correctness argument, and casting to
/// `make_unsigned_t<T>` alone does NOT deliver it. Integral promotion runs
/// before any arithmetic operator, and it promotes any type narrower than `int`
/// to *signed* `int` -- so for `T = int16_t`, `static_cast<uint16_t>(-1) *
/// static_cast<uint16_t>(-1)` is `65535 * 65535` evaluated in `int`, which
/// overflows and is undefined. The unsigned cast meant to prevent UB
/// reintroduced it for every type narrower than `int`. Multiplying in
/// `common_type_t<U, unsigned>` keeps the operation unsigned, where overflow is
/// defined as reduction, and the result is then narrowed back to `U`.
template <typename T>
using wrap_op_t = std::common_type_t<std::make_unsigned_t<T>, unsigned int>;

template <typename T>
MTL5_WRAPS_ON_PURPOSE
constexpr T wrap_add(T a, T b) noexcept {
    if constexpr (is_wrapping_integer_v<T>) {
        using U = std::make_unsigned_t<T>;
        using P = wrap_op_t<T>;
        return static_cast<T>(static_cast<U>(static_cast<P>(static_cast<U>(a)) +
                                             static_cast<P>(static_cast<U>(b))));
    } else {
        return a + b;
    }
}

template <typename T>
MTL5_WRAPS_ON_PURPOSE
constexpr T wrap_sub(T a, T b) noexcept {
    if constexpr (is_wrapping_integer_v<T>) {
        using U = std::make_unsigned_t<T>;
        using P = wrap_op_t<T>;
        return static_cast<T>(static_cast<U>(static_cast<P>(static_cast<U>(a)) -
                                             static_cast<P>(static_cast<U>(b))));
    } else {
        return a - b;
    }
}

template <typename T>
MTL5_WRAPS_ON_PURPOSE
constexpr T wrap_mul(T a, T b) noexcept {
    if constexpr (is_wrapping_integer_v<T>) {
        using U = std::make_unsigned_t<T>;
        using P = wrap_op_t<T>;
        return static_cast<T>(static_cast<U>(static_cast<P>(static_cast<U>(a)) *
                                             static_cast<P>(static_cast<U>(b))));
    } else {
        return a * b;
    }
}

/// `acc + a * b` for the generic element-wise loops: wrapping when the result
/// type is integral, the plain expression otherwise.
///
/// The generic path is where an integer `mult` lands whenever the native kernel
/// is not selected -- which is the DEFAULT build, since MTL5_NATIVE_FAST_GEMM is
/// off -- and where a non-contiguous integer vector lands in `dot`. Both must
/// agree with the SIMD path bit for bit, and neither may be undefined.
///
/// The operands are converted to `Result` BEFORE multiplying, which is what
/// makes the integer case exact mod 2^N rather than merely defined: it is the
/// product that overflows first, not the sum. Narrowing first loses nothing,
/// because reduction mod 2^N is a ring homomorphism -- (a mod 2^N)(b mod 2^N)
/// and (a b) mod 2^N agree.
///
/// The OPERANDS must be integral too, not just `Result`. Converting them first
/// is only lossless between integers: with a floating-point operand and an
/// integral result -- `mult(dense2D<double>, dense_vector<double>,
/// dense_vector<int>)` -- it would truncate each factor before multiplying and
/// turn 2.5 * 2.5 into 2 * 2, where the plain expression accumulates in the
/// result type and rounds each term once. Those mixed cases keep the original
/// expression.
///
/// The operand test is `std::is_integral_v`, NOT `is_wrapping_integer_v`. The
/// difference is `bool`, and it matters in the direction that is easy to get
/// backwards: `bool` must be excluded as a RESULT type, because
/// `std::make_unsigned_t<bool>` is ill-formed, but it must be INCLUDED as an
/// operand, because converting it to an integral `Result` is exact (0 or 1) and
/// excluding it would push the whole expression onto the plain branch --
/// reinstating the signed-overflow UB this helper exists to remove.
/// `dot(dense_vector<bool>, dense_vector<int32_t>)` is that case.
template <typename Result, typename A, typename B>
constexpr Result generic_fma(Result acc, const A& a, const B& b) {
    if constexpr (is_wrapping_integer_v<Result> &&
                  std::is_integral_v<A> && std::is_integral_v<B>) {
        return wrap_add(acc, wrap_mul(static_cast<Result>(a), static_cast<Result>(b)));
    } else {
        return acc + a * b;
    }
}

/// `acc + a` for the generic element-wise loops: wrapping when the result type
/// is integral, the plain expression otherwise (#461).
///
/// `generic_fma` for the loops that accumulate a term they did not form by
/// multiplying -- `one_norm`'s `acc += abs(v(i))`, the column and row sums in the
/// matrix norms, and the scatter in the transposed sparse matvec, where the
/// product was computed at a different point from the addition.
///
/// The same operand rule as `generic_fma`, for the same reason: `A` is tested
/// with `std::is_integral_v`, which admits `bool`, because converting a bool to
/// an integral `Result` is exact and excluding it would push the expression onto
/// the plain branch and reinstate the UB.
template <typename Result, typename A>
constexpr Result generic_add(Result acc, const A& a) {
    if constexpr (is_wrapping_integer_v<Result> && std::is_integral_v<A>) {
        return wrap_add(acc, static_cast<Result>(a));
    } else {
        return acc + a;
    }
}

/// |a|, wrapping, for the norm accumulations (#461).
///
/// `std::abs` IS UB at the minimum of a signed type -- there is no positive
/// `-2^(N-1)` to return -- so `one_norm(dense_vector<int32_t>)` over full-range
/// data is undefined before any accumulation happens. That is a separate defect
/// from the `+=` this pass is fixing, and it is reached by exactly the test the
/// fix requires: full-range operands include the minimum.
///
/// The wrapping answer is the minimum itself, since 2^(N-1) reduced mod 2^N is
/// -2^(N-1). Expressed as `wrap_sub(0, a)` so the narrow types get the same
/// promotion handling as everywhere else in this header -- writing the unsigned
/// negation inline would reintroduce the int-promotion trap for int8/int16.
///
/// Non-integral types keep `using std::abs; abs(a)`, so the ADL-found `abs` of a
/// custom number type is still what gets called.
template <typename T>
constexpr auto generic_abs(const T& a) {
    if constexpr (is_wrapping_integer_v<T>) {
        return a < T{0} ? wrap_sub(T{0}, a) : a;
    } else {
        using std::abs;
        return abs(a);
    }
}

/// `a * b` delivered in `Result`: wrapping when `Result` is integral, the plain
/// expression otherwise (#461). The multiply-only counterpart, for `scale`'s
/// `*it *= alpha`.
///
/// NOTE the difference from `generic_fma`, which deliberately does NOT convert on
/// its non-integral branch: here the conversion is already part of the operation
/// being replaced. `*it *= alpha` is `*it = static_cast<T>(*it * alpha)` by the
/// definition of compound assignment, so returning `Result` preserves the
/// existing rounding rather than adding one.
template <typename Result, typename A, typename B>
constexpr Result generic_mul(const A& a, const B& b) {
    if constexpr (is_wrapping_integer_v<Result> &&
                  std::is_integral_v<A> && std::is_integral_v<B>) {
        return wrap_mul(static_cast<Result>(a), static_cast<Result>(b));
    } else {
        return a * b;
    }
}

} // namespace mtl::detail
