#pragma once
// MTL5 -- keep the interleaved row accumulators out of GCC's SLP vectorizer (#776)
//
// The generic gemv, trmv and forward substitution sum W rows per pass, one
// accumulator per row (#749, #772). On a row-major A, GCC's SLP vectorizer
// packs the W float accumulators into vector registers, filling each vector
// from W scalar loads a whole row apart. The packing costs more than the
// vector add saves. On an i7-12700K (n = 4096, one core), f32 gemv at the
// default W = 6 took 3.04 ms with it and 2.72 ms without; under -march=native,
// 3.79 against 2.87 ms.
//
// keep_scalar(acc) is an empty asm statement that claims to read and write
// the accumulator in a register of its own. It emits no instruction, but GCC
// can no longer see a lane of a vector in it, so it leaves the W chains
// scalar. Rounding is untouched: the value goes in and comes out unchanged.
//
// It applies only where it was measured to help (keep_scalar_v):
//   - GCC on x86-64 or AArch64. Clang does not pack these chains the same way
//     (f32 gemv 2.62 ms at W = 6 with no barrier), and other compilers and
//     targets are untested.
//   - float. For double, the packing is close to neutral (2-3% faster with it
//     on in a plain build), so double keeps it.
//   - A row-major A. In a column-major A the W rows of one column are
//     contiguous, the packed loads are cheap, and the barrier made f32 and f64
//     gemv 10-15% slower at W = 6.
//   - W >= 4, where it was measured to help. One chain has nothing to pack,
//     and under -march=native the barrier doubled the time of the one-row loop
//     and made f32 trmv at W = 2 1.7x slower. Narrower widths only finish the
//     few rows a wider block leaves over. At W = 4 it is not a clean win
//     either: under -march=native f32 gemv got 9% faster but trmv and the
//     forward substitution 8% slower, while a plain build gained 12-24% in
//     all three. No float default is 4 where the barrier applies.
//
// The bits are checked by op_test_trmv_trsv_interleaved, whose row-major
// float cases at W >= 4 take this path in a GCC build.

#include <cstddef>
#include <type_traits>

#include <mtl/tag/orientation.hpp>

namespace mtl::detail {

/// Whether M is a row-major matrix type.
template <typename M>
constexpr bool keep_scalar_row_major() {
    if constexpr (requires { typename M::orientation; })
        return std::is_same_v<typename M::orientation, tag::row_major>;
    else
        return false;
}

/// Whether the W accumulators of type T over a matrix of type M need
/// keep_scalar (see the top of this file).
template <typename T, typename M, std::size_t W>
inline constexpr bool keep_scalar_v =
#if defined(__GNUC__) && !defined(__clang__) && (defined(__x86_64__) || defined(__aarch64__))
    W >= 4 && std::is_same_v<T, float> && keep_scalar_row_major<M>();
#else
    false;
#endif

/// An empty asm that keeps v in a scalar register of its own, when Enable.
template <bool Enable, typename T>
inline void keep_scalar(T& v) {
    if constexpr (Enable) {
#if defined(__GNUC__) && !defined(__clang__) && defined(__x86_64__)
        asm("" : "+x"(v));
#elif defined(__GNUC__) && !defined(__clang__) && defined(__aarch64__)
        asm("" : "+w"(v));
#endif
    }
    static_cast<void>(v);
}

} // namespace mtl::detail
