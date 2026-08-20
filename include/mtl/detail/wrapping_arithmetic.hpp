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
#include <type_traits>

namespace mtl::detail {

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

} // namespace mtl::detail
