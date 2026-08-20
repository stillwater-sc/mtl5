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

template <typename T>
constexpr T wrap_add(T a, T b) noexcept {
    if constexpr (std::is_integral_v<T>) {
        using U = std::make_unsigned_t<T>;
        return static_cast<T>(static_cast<U>(static_cast<U>(a) + static_cast<U>(b)));
    } else {
        return a + b;
    }
}

template <typename T>
constexpr T wrap_sub(T a, T b) noexcept {
    if constexpr (std::is_integral_v<T>) {
        using U = std::make_unsigned_t<T>;
        return static_cast<T>(static_cast<U>(static_cast<U>(a) - static_cast<U>(b)));
    } else {
        return a - b;
    }
}

template <typename T>
constexpr T wrap_mul(T a, T b) noexcept {
    if constexpr (std::is_integral_v<T>) {
        using U = std::make_unsigned_t<T>;
        return static_cast<T>(static_cast<U>(static_cast<U>(a) * static_cast<U>(b)));
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
/// product that overflows first, not the sum.
template <typename Result, typename A, typename B>
constexpr Result generic_fma(Result acc, const A& a, const B& b) {
    if constexpr (std::is_integral_v<Result>) {
        return wrap_add(acc, wrap_mul(static_cast<Result>(a), static_cast<Result>(b)));
    } else {
        return acc + a * b;
    }
}

} // namespace mtl::detail
