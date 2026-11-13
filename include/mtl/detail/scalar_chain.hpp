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
//     all three. That was x86-64.
//   - On AArch64, gemv only (keep_scalar_triangular_v). The barrier is what
//     made GCC's float trmv and forward substitution 5.6-7.2x slower at W = 4
//     than at 2 there (#795). An A/B with and without it (generic build,
//     n = 4096, one core, median of 3 runs of bench_interleave_rows --reps 21):
//
//                                 with the barrier    without it
//         Jetson Orin, GCC 11
//           trmv f32, W = 4         50.04 ms            6.31 ms
//           trisolve f32, W = 4     39.45 ms            5.58 ms
//           gemv f32, W = 4          8.23 ms            8.34 ms
//         Ventuno Q, GCC 13
//           trmv f32, W = 4         31.66 ms            3.12 ms
//           trisolve f32, W = 4     26.22 ms            2.99 ms
//           gemv f32, W = 4          5.51 ms            9.14 ms
//
//     Without it, W = 4 is the triangular kernels' fastest width on both
//     boards, 13-36% faster than the 2 they were held at before. gemv keeps
//     the barrier: on the Ventuno it is 1.66x faster with it. The upper trmv
//     wasn't timed (the sweep runs the lower one) but has the same inner
//     loop, so it follows. On x86-64 nothing changes: all three keep it, as
//     measured there.
//
// The bits are checked by op_test_trmv_trsv_interleaved, whose row-major
// float cases at W >= 4 take this path in a GCC build (gemv only, on AArch64).

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

/// The same, for the trmv and forward-substitution kernels: as keep_scalar_v
/// on x86-64, never on AArch64, where it made them 5.6-7.2x slower (#795).
template <typename T, typename M, std::size_t W>
inline constexpr bool keep_scalar_triangular_v =
#if defined(__aarch64__)
    false;
#else
    keep_scalar_v<T, M, W>;
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
