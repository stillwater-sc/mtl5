#pragma once
// MTL5 -- vectorized level-1 kernels over raw contiguous storage (#86, epic #82).
//
// These are the native (no external BLAS) L1 building blocks: dot, sum-of-
// squares (for two_norm), axpy, and scal. Each is written ONCE against the SIMD
// abstraction (mtl::simd::batch<T>, #83) so it targets SSE/AVX/AVX-512/NEON by
// build flags. Reductions use several independent accumulators to hide the FP
// add/FMA latency (the dependent-accumulator stall that limits a single-acc
// loop), then a horizontal reduce and a scalar tail. L1 is bandwidth-bound, so
// the goal is to reach the memory ceiling, not peak FLOPs.
//
// Unaligned loads/stores are used so these work on any contiguous span
// (sub-vectors, views); on modern cores aligned vs unaligned cost is
// negligible. The packed-panel GEMM path (#89) uses aligned loads instead.
//
// Lane types are whatever batch<T> supports (mtl::simd::is_lane_v): float,
// double, int32_t, uint32_t. On integer lanes every kernel here is EXACT mod
// 2^32 -- the accumulators, the horizontal reduce and the scalar tail all
// commute, because wrapping addition is associative. So the integer results are
// bit-identical across lane counts and backends, and the scalar tails go
// through the wrapping helpers rather than raw `+`/`*` so that a tail overflow
// is defined rather than UB. Widening accumulation (int8/int16 into int32) is
// #451 phases 2-3 and is not offered here.
//
// ACCUMULATOR COUNT IS A FLOATING-POINT DECISION
//
// The four independent accumulators exist to hide FP add/FMA latency: an FMA
// retires in 3-5 cycles, so a single accumulator serializes the loop on that
// chain. Integer lanes have no such problem -- the loop-carried dependency is
// an integer add (1 cycle), while the body needs two loads and a multiply, so
// the loop is throughput-bound long before it is latency-bound. The integer
// reductions therefore run ONE accumulator, and the hand unroll is skipped.
//
// That is also what keeps them correct today. On the scalar-fallback backend
// (W == 1) the hand unroll is four interleaved scalar chains, and the
// autovectorizer has to fuse them back into one contiguous reduction -- legal
// for integers, because their addition reassociates, and NOT attempted for
// floats, because FP addition does not. GCC 13 performs that fusion and gets
// the bookkeeping wrong: at -O3 -march=x86-64-v3 it vectorizes the first chain
// with 8-wide contiguous loads (advancing 8 elements per iteration) while
// leaving the other three chains advancing 4, so they cover half the range and
// the result is silently wrong. Measured on GCC 13.3: 176 of the first 299
// compile-time-constant lengths, int32_t and uint32_t only, float and double
// never. Clang is unaffected.
//
// The single-accumulator form is not a workaround chosen to dodge that -- it is
// the right shape for integers on its own terms, and it happens to be the shape
// the compiler already handles, because there is nothing left to un-unroll.
// tests/unit/simd/test_integer_lanes.cpp sweeps compile-time-constant lengths
// against both the runtime-length result and a closed-form reference, which is
// how the miscompile was caught and what will catch the next one.

#include <cstddef>
#include <type_traits>

#include <mtl/simd/batch.hpp>

namespace mtl::simd {

/// dot product: sum_i a[i]*b[i]  (real float/double, or int32/uint32 mod 2^32).
template <typename T>
T reduce_dot(const T* a, const T* b, std::size_t n) {
    static_assert(is_lane_v<T>);
    using B = batch<T>;
    constexpr std::size_t W = B::size;
    B a0{}, a1{}, a2{}, a3{};                // default ctor zero-initializes
    std::size_t i = 0;
    // Four-accumulator body: floating lanes only (see the header note). On
    // integer lanes a1..a3 stay zero and fold out of the combine exactly.
    if constexpr (std::is_floating_point_v<T>) {
        constexpr std::size_t U = 4;         // independent accumulators
        constexpr std::size_t step = W * U;
        for (; i + step <= n; i += step) {
            a0 = fma(B::load_unaligned(a + i),           B::load_unaligned(b + i),           a0);
            a1 = fma(B::load_unaligned(a + i + W),       B::load_unaligned(b + i + W),       a1);
            a2 = fma(B::load_unaligned(a + i + 2 * W),   B::load_unaligned(b + i + 2 * W),   a2);
            a3 = fma(B::load_unaligned(a + i + 3 * W),   B::load_unaligned(b + i + 3 * W),   a3);
        }
    }
    for (; i + W <= n; i += W)
        a0 = fma(B::load_unaligned(a + i), B::load_unaligned(b + i), a0);
    T s = reduce_add((a0 + a1) + (a2 + a3));
    for (; i < n; ++i)                       // scalar tail, fused as the body is
        s = scalar_fma(a[i], b[i], s);
    return s;
}

/// Widening dot product: sum_i a[i]*b[i] with `Narrow` elements (e.g. float)
/// accumulated in a wider `Wide` (e.g. double). Each load widens `Wide`-lane-
/// count narrow values to `Wide` lanes, then FMAs into `Wide` accumulators -- so
/// mixed-precision dot keeps the wide accumulator's accuracy at SIMD speed,
/// instead of the scalar policy fallback. (On the scalar batch backend this is
/// the correct scalar widening loop.)
/// (Integer overload below; this one is the floating-point case.)
template <typename Wide, typename Narrow>
    requires std::is_floating_point_v<Wide>
Wide reduce_dot_widen(const Narrow* a, const Narrow* b, std::size_t n) {
    static_assert(std::is_floating_point_v<Narrow>);
    static_assert(sizeof(Narrow) < sizeof(Wide), "reduce_dot_widen requires Narrow < Wide");
    using B = batch<Wide>;
    constexpr std::size_t W = B::size;
    constexpr std::size_t U = 4;             // independent accumulators
    constexpr std::size_t step = W * U;
    B a0{}, a1{}, a2{}, a3{};
    std::size_t i = 0;
    for (; i + step <= n; i += step) {
        a0 = fma(B::load_widen(a + i),           B::load_widen(b + i),           a0);
        a1 = fma(B::load_widen(a + i + W),       B::load_widen(b + i + W),       a1);
        a2 = fma(B::load_widen(a + i + 2 * W),   B::load_widen(b + i + 2 * W),   a2);
        a3 = fma(B::load_widen(a + i + 3 * W),   B::load_widen(b + i + 3 * W),   a3);
    }
    for (; i + W <= n; i += W)
        a0 = fma(B::load_widen(a + i), B::load_widen(b + i), a0);
    Wide s = reduce_add((a0 + a1) + (a2 + a3));
    for (; i < n; ++i)                       // tail, fused as the body is
        s = scalar_fma(static_cast<Wide>(a[i]), static_cast<Wide>(b[i]), s);
    return s;
}

/// Widening INTEGER dot product: 16-bit operands accumulated in 32 bits, via the
/// hardware's widening multiply-accumulate (#451 phase 2).
///
/// `int16 -> int32` and `uint16 -> uint32`. One instruction does the widening,
/// the multiply and the accumulate -- `vpmaddwd` on x86, `SMLAL`/`UMLAL` on
/// NEON -- so this is the first kernel here whose ISA support is the point
/// rather than an implementation detail.
///
/// THE OVERFLOW CONTRACT
///
/// Two separate questions, with very different answers.
///
///   * The PRODUCTS are always exact. |-2^15 * -2^15| = 2^30 fits an int32 with
///     a bit to spare, so no individual term is ever lost.
///   * The SUM wraps, and it wraps SOON. Full-range operands consume 31 of the
///     accumulator's 32 bits per term, so the worst case (every term the same
///     sign, both operands at the limit) overflows at k = 3. Random signed data
///     does better -- the terms random-walk rather than march, so |sum| grows
///     like sqrt(k) -- but only to roughly k = 36 before it is more likely than
///     not to have exceeded 2^31.
///
/// So this is NOT the "accumulate 65 000 terms safely" case; that is int8 into
/// int32 (phase 3), where a product needs 15 bits and leaves 16 of headroom, and
/// it is why the VNNI and SDOT instructions are shaped the way they are. What
/// makes 16-bit accumulation useful is small operands, not short vectors: at b
/// bits of operand magnitude the headroom is about 2^(31-2b) terms, so 8-bit
/// values in int16 storage give ~2^15 terms and 4-bit values give ~2^23.
///
/// When the sum does exceed the accumulator it WRAPS, exactly as everywhere else
/// in mtl::simd -- the result is the true sum reduced mod 2^32, never a trap,
/// never a saturation, never undefined. Callers who need the full range should
/// widen their storage, not hope. tests/unit/simd/test_int_accumulation.cpp pins
/// the measured envelope for the 64-bit case.
///
/// LANE ORDER IS UNSPECIFIED AND THAT IS FINE. The hardware instruction is free
/// to permute which product lands in which accumulator lane, and Highway
/// promises only that the total is right. For a floating accumulator that would
/// make the result ISA-dependent; here it is unobservable, because addition mod
/// 2^32 is associative and commutative. Phase 0's contract is what lets this
/// kernel use the instruction without a caveat.
/// 8-BIT OPERANDS (#451 phase 3) are the same function with a different
/// instruction and a very different headroom, so the two live together here.
///
///   16-bit: pairwise widen (`vpmaddwd`), 2 products per 32-bit lane
///    8-bit: quad widen (VNNI `vpdpbusd`, NEON `SDOT`), 4 products per lane
///
/// And the headroom is where the 8-bit case earns "the one that matters":
///
///   | pairing        | max |product| | terms before the sum can overflow |
///   |----------------|----------------|-----------------------------------|
///   | int16 x int16  | 2^30           | 3 worst case, ~36 random          |
///   |  int8 x int8   | 2^14           | ~131 000                          |
///   | uint8 x  int8  | ~2^15          | ~65 000                           |
///   | uint8 x uint8  | ~2^16          | ~66 000                           |
///
/// Four to five orders of magnitude, from the same 32-bit accumulator, purely
/// because the operands are half as wide and the product therefore a quarter
/// the size. This is why the quantized-inference instructions are 8-bit and not
/// 16-bit, and why phase 2's kernel is useful only for small operands while this
/// one is useful for real vectors.
///
/// THE SIGNEDNESS ASYMMETRY IS THE HARDWARE'S. VNNI implements `unsigned x
/// signed` -- unsigned activations against signed weights -- and symmetric
/// `i8 x i8` only from AVX10.2. Before that Highway emulates the symmetric form
/// with two `dpbusd` calls plus a shift and subtract, roughly three times the
/// work. `(int8, uint8)` is rejected rather than silently reordered: the dot is
/// symmetric, so swapping the arguments gets the native instruction, and a
/// compile error saying so beats a quiet detour through the emulation.
template <typename Wide, typename NA, typename NB = NA>
    requires std::is_integral_v<Wide>
Wide reduce_dot_widen(const NA* a, const NB* b, std::size_t n) {
    static_assert(is_widenable_v<NA> || is_quad_widenable_v<NA>,
                  "reduce_dot_widen widens 8- or 16-bit integers");
    using B = batch<Wide>;
    std::size_t i = 0;
    Wide s{};
    // Guarded so a bad Narrow reports ONE diagnostic: the accumulator traits are
    // undefined for it, and a discarded if-constexpr branch is not instantiated.
    if constexpr (is_quad_widenable_v<NA>) {
        static_assert(is_quad_widenable_v<NB>, "both operands must be 8-bit");
        static_assert(QuadPair<NA, NB>,
                      "unsupported 8-bit pairing -- the hardware op takes "
                      "(i8,i8), (u8,i8) or (u8,u8); for (i8,u8) swap the "
                      "arguments, which a dot product permits");
        // Guarded for the same reason as the 16-bit case below: quad_accumulator
        // is undefined for a rejected pairing, so naming it unconditionally
        // would bury the assertion above under an "incomplete type" error.
        if constexpr (QuadPair<NA, NB>) {
            static_assert(std::is_same_v<Wide, quad_accumulator_t<NA, NB>>,
                          "i8*i8 and u8*i8 accumulate in int32, u8*u8 in uint32");
        }
        constexpr std::size_t S = B::quad_step;
        B acc{};
        for (; i + S <= n; i += S) B::quad_dot_accumulate(a + i, b + i, acc);
        s = reduce_add(acc);
    } else {
        static_assert(std::is_same_v<NA, NB>,
                      "16-bit widening takes two operands of the same type");
        static_assert(is_widenable_v<NA>, "reduce_dot_widen widens 16-bit integers");
        if constexpr (is_widenable_v<NA>) {
            static_assert(std::is_same_v<Wide, widen_accumulator_t<NA>>,
                          "int16 accumulates in int32, uint16 in uint32");
        }
        constexpr std::size_t S = B::widen_step;
        B s0{}, s1{};                        // s1 MUST start zeroed -- see batch.hpp
        for (; i + S <= n; i += S) B::widen_dot_accumulate(a + i, b + i, s0, s1);
        s = reduce_add(s0 + s1);             // the one defined observation
    }
    for (; i < n; ++i)                       // scalar tail, same wrapping rules
        s = mtl::detail::wrap_add(
            s, mtl::detail::wrap_mul(static_cast<Wide>(a[i]), static_cast<Wide>(b[i])));
    return s;
}

/// sum of squares: sum_i a[i]*a[i]  (two_norm computes sqrt of this).
template <typename T>
T reduce_sum_squares(const T* a, std::size_t n) {
    static_assert(is_lane_v<T>);
    using B = batch<T>;
    constexpr std::size_t W = B::size;
    B a0{}, a1{}, a2{}, a3{};
    std::size_t i = 0;
    if constexpr (std::is_floating_point_v<T>) {   // see reduce_dot / header note
        constexpr std::size_t U = 4;
        constexpr std::size_t step = W * U;
        for (; i + step <= n; i += step) {
            B v0 = B::load_unaligned(a + i);          a0 = fma(v0, v0, a0);
            B v1 = B::load_unaligned(a + i + W);      a1 = fma(v1, v1, a1);
            B v2 = B::load_unaligned(a + i + 2 * W);  a2 = fma(v2, v2, a2);
            B v3 = B::load_unaligned(a + i + 3 * W);  a3 = fma(v3, v3, a3);
        }
    }
    for (; i + W <= n; i += W) { B v = B::load_unaligned(a + i); a0 = fma(v, v, a0); }
    T s = reduce_add((a0 + a1) + (a2 + a3));
    for (; i < n; ++i) s = scalar_fma(a[i], a[i], s);   // fused as the body is
    return s;
}

/// axpy: y[i] += alpha*x[i].
template <typename T>
void axpy(T alpha, const T* x, T* y, std::size_t n) {
    static_assert(is_lane_v<T>);
    using B = batch<T>;
    constexpr std::size_t W = B::size;
    const B va(alpha);
    std::size_t i = 0;
    for (; i + W <= n; i += W) {
        B r = fma(va, B::load_unaligned(x + i), B::load_unaligned(y + i));
        r.store_unaligned(y + i);
    }
    for (; i < n; ++i) y[i] = scalar_fma(alpha, x[i], y[i]);   // fused as the body is
}

/// scal: x[i] *= alpha.
template <typename T>
void scal(T alpha, T* x, std::size_t n) {
    static_assert(is_lane_v<T>);
    using B = batch<T>;
    constexpr std::size_t W = B::size;
    const B va(alpha);
    std::size_t i = 0;
    for (; i + W <= n; i += W)
        (va * B::load_unaligned(x + i)).store_unaligned(x + i);
    for (; i < n; ++i) x[i] = mtl::detail::wrap_mul(x[i], alpha);
}

} // namespace mtl::simd
