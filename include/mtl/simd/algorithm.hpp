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

#include <cstddef>
#include <type_traits>

#include <mtl/simd/batch.hpp>

namespace mtl::simd {

/// dot product: sum_i a[i]*b[i]  (real float/double).
template <typename T>
T reduce_dot(const T* a, const T* b, std::size_t n) {
    static_assert(std::is_floating_point_v<T>);
    using B = batch<T>;
    constexpr std::size_t W = B::size;
    constexpr std::size_t U = 4;             // independent accumulators
    constexpr std::size_t step = W * U;
    B a0{}, a1{}, a2{}, a3{};                // default ctor zero-initializes
    std::size_t i = 0;
    for (; i + step <= n; i += step) {
        a0 = fma(B::load_unaligned(a + i),           B::load_unaligned(b + i),           a0);
        a1 = fma(B::load_unaligned(a + i + W),       B::load_unaligned(b + i + W),       a1);
        a2 = fma(B::load_unaligned(a + i + 2 * W),   B::load_unaligned(b + i + 2 * W),   a2);
        a3 = fma(B::load_unaligned(a + i + 3 * W),   B::load_unaligned(b + i + 3 * W),   a3);
    }
    for (; i + W <= n; i += W)
        a0 = fma(B::load_unaligned(a + i), B::load_unaligned(b + i), a0);
    T s = reduce_add((a0 + a1) + (a2 + a3));
    for (; i < n; ++i) s += a[i] * b[i];     // scalar tail
    return s;
}

/// sum of squares: sum_i a[i]*a[i]  (two_norm computes sqrt of this).
template <typename T>
T reduce_sum_squares(const T* a, std::size_t n) {
    static_assert(std::is_floating_point_v<T>);
    using B = batch<T>;
    constexpr std::size_t W = B::size;
    constexpr std::size_t U = 4;
    constexpr std::size_t step = W * U;
    B a0{}, a1{}, a2{}, a3{};
    std::size_t i = 0;
    for (; i + step <= n; i += step) {
        B v0 = B::load_unaligned(a + i);          a0 = fma(v0, v0, a0);
        B v1 = B::load_unaligned(a + i + W);      a1 = fma(v1, v1, a1);
        B v2 = B::load_unaligned(a + i + 2 * W);  a2 = fma(v2, v2, a2);
        B v3 = B::load_unaligned(a + i + 3 * W);  a3 = fma(v3, v3, a3);
    }
    for (; i + W <= n; i += W) { B v = B::load_unaligned(a + i); a0 = fma(v, v, a0); }
    T s = reduce_add((a0 + a1) + (a2 + a3));
    for (; i < n; ++i) s += a[i] * a[i];
    return s;
}

/// axpy: y[i] += alpha*x[i].
template <typename T>
void axpy(T alpha, const T* x, T* y, std::size_t n) {
    static_assert(std::is_floating_point_v<T>);
    using B = batch<T>;
    constexpr std::size_t W = B::size;
    const B va(alpha);
    std::size_t i = 0;
    for (; i + W <= n; i += W) {
        B r = fma(va, B::load_unaligned(x + i), B::load_unaligned(y + i));
        r.store_unaligned(y + i);
    }
    for (; i < n; ++i) y[i] += alpha * x[i];
}

/// scal: x[i] *= alpha.
template <typename T>
void scal(T alpha, T* x, std::size_t n) {
    static_assert(std::is_floating_point_v<T>);
    using B = batch<T>;
    constexpr std::size_t W = B::size;
    const B va(alpha);
    std::size_t i = 0;
    for (; i + W <= n; i += W)
        (va * B::load_unaligned(x + i)).store_unaligned(x + i);
    for (; i < n; ++i) x[i] *= alpha;
}

} // namespace mtl::simd
