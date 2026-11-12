#pragma once
// MTL5 -- optimized native GEMV (#87, epic #82, Phase 1).
//
// y := A * x. GEMV is memory-bandwidth-bound (A is O(mn) bytes read once and is
// the bottleneck), so there is no packing: the wins are an orientation-aware
// loop order that reads A unit-stride, SIMD + FMA over the contiguous axis, and
// register-blocking the *reused* vector to cut its traffic.
//
//   row-major A  -> dot-of-rows: block MR rows so each loaded x batch is reused
//                   MR times (x traffic / MR); one horizontal reduce per row.
//   col-major A  -> axpy-of-columns over an L1-resident strip of y: walk down
//                   each column contiguously, accumulating (broadcast x[j]) *
//                   A[strip, j] into the strip, four columns per pass (#804).
//
// A(i,j) is read from its orientation's contiguous layout:
//   row-major  A(i,j) = A[i*lda + j]   (lda = n, rows contiguous)
//   col-major  A(i,j) = A[j*lda + i]   (lda = m, columns contiguous)

#include <cstddef>

#include <mtl/concepts/scalar.hpp>
#include <mtl/simd/algorithm.hpp>   // reduce_dot for the row remainder
#include <mtl/simd/batch.hpp>

namespace mtl::detail {

/// y[m] := A[m x n] * x[n], A row-major (rows contiguous, lda = n).
template <typename T>
    requires mtl::Scalar<T>
void gemv_rowmajor(std::size_t m, std::size_t n, const T* A, std::size_t lda,
                   const T* x, T* y) {
    using B = simd::batch<T>;
    constexpr std::size_t W = B::size;
    constexpr std::size_t MR = 4;                 // rows per register block

    std::size_t i = 0;
    for (; i + MR <= m; i += MR) {
        const T* a0 = A + (i + 0) * lda;
        const T* a1 = A + (i + 1) * lda;
        const T* a2 = A + (i + 2) * lda;
        const T* a3 = A + (i + 3) * lda;
        B acc0, acc1, acc2, acc3;                 // batch() zero-initializes
        std::size_t j = 0;
        for (; j + W <= n; j += W) {
            const B xb = B::load_unaligned(x + j);  // loaded once, reused MR times
            acc0 = fma(B::load_unaligned(a0 + j), xb, acc0);
            acc1 = fma(B::load_unaligned(a1 + j), xb, acc1);
            acc2 = fma(B::load_unaligned(a2 + j), xb, acc2);
            acc3 = fma(B::load_unaligned(a3 + j), xb, acc3);
        }
        T s0 = reduce_add(acc0), s1 = reduce_add(acc1),
          s2 = reduce_add(acc2), s3 = reduce_add(acc3);
        for (; j < n; ++j) {                      // scalar tail over columns
            const T xj = x[j];
            // wrap_* so an integer tail wraps like the SIMD body above rather
            // than being signed-overflow UB (see simd/batch.hpp); identity on
            // floating lanes.
            s0 = mtl::detail::wrap_add(s0, mtl::detail::wrap_mul(a0[j], xj));
            s1 = mtl::detail::wrap_add(s1, mtl::detail::wrap_mul(a1[j], xj));
            s2 = mtl::detail::wrap_add(s2, mtl::detail::wrap_mul(a2[j], xj));
            s3 = mtl::detail::wrap_add(s3, mtl::detail::wrap_mul(a3[j], xj));
        }
        y[i + 0] = s0; y[i + 1] = s1; y[i + 2] = s2; y[i + 3] = s3;
    }
    for (; i < m; ++i)                            // remaining rows (< MR)
        y[i] = simd::reduce_dot<T>(A + i * lda, x, n);
}

/// y[m] := A[m x n] * x[n], A col-major (columns contiguous, lda = m).
///
/// The rows are swept in STRIPS of S rows (16 KB of y, so the strip stays in
/// L1), and each strip walks down the columns, reading S contiguous elements of
/// each (#804). The kernel before it held only 4 batches of y in registers
/// (16 rows of f64 under AVX2) and swept all n columns for them, so every load
/// came from a different column: at n = 4096 each strip touched 4096 cache lines
/// on 4096 pages, and native-fast ran 2.2-5x slower than the generic build's
/// column sweep (#777), which already had this shape.
///
/// Four columns are taken per pass, so the strip of y is loaded and stored once
/// per four columns rather than once per column. Each y[i] still sees the same
/// FMA chain, in ascending column order from zero, as the register-strip kernel
/// gave it, so the result is bit-identical to that kernel, and independent of
/// how the caller partitions the rows across threads.
template <typename T>
    requires mtl::Scalar<T>
void gemv_colmajor(std::size_t m, std::size_t n, const T* A, std::size_t lda,
                   const T* x, T* y) {
    using B = simd::batch<T>;
    constexpr std::size_t W = B::size;
    // Rows per strip: 16 KB of y, a whole number of batches.
    constexpr std::size_t S = (16384 / sizeof(T)) / W * W > 0 ? (16384 / sizeof(T)) / W * W : W;

    const std::size_t mv = m / W * W;             // rows the SIMD strips cover
    for (std::size_t i0 = 0; i0 < mv; i0 += S) {
        const std::size_t i1 = i0 + S < mv ? i0 + S : mv;
        for (std::size_t i = i0; i < i1; i += W) B().store_unaligned(y + i);
        std::size_t j = 0;
        for (; j + 4 <= n; j += 4) {              // four columns per pass
            const B x0(x[j + 0]), x1(x[j + 1]), x2(x[j + 2]), x3(x[j + 3]);
            const T* c0 = A + (j + 0) * lda;
            const T* c1 = A + (j + 1) * lda;
            const T* c2 = A + (j + 2) * lda;
            const T* c3 = A + (j + 3) * lda;
            for (std::size_t i = i0; i < i1; i += W) {
                B acc = B::load_unaligned(y + i);
                acc = fma(x0, B::load_unaligned(c0 + i), acc);
                acc = fma(x1, B::load_unaligned(c1 + i), acc);
                acc = fma(x2, B::load_unaligned(c2 + i), acc);
                acc = fma(x3, B::load_unaligned(c3 + i), acc);
                acc.store_unaligned(y + i);
            }
        }
        for (; j < n; ++j) {                      // remaining columns (< 4)
            const B xj(x[j]);
            const T* c = A + j * lda;
            for (std::size_t i = i0; i < i1; i += W)
                fma(xj, B::load_unaligned(c + i), B::load_unaligned(y + i)).store_unaligned(y + i);
        }
    }
    for (std::size_t i = mv; i < m; ++i) {        // remaining rows (< W), scalar
        T s = T(0);
        for (std::size_t j = 0; j < n; ++j)
            s = mtl::detail::wrap_add(s, mtl::detail::wrap_mul(A[j * lda + i], x[j]));
        y[i] = s;
    }
}

/// t[m] := A(r0 : r0+m, c0 : c0+k) * x[k] for a sub-panel of an n-column
/// (row-major) or n-row (column-major) dense matrix at A: the off-diagonal
/// panel of the blocked trmv and trsv (#747). k may be 0 (t is then zero).
template <typename T>
    requires mtl::Scalar<T>
void gemv_panel(bool row_major, std::size_t ld, const T* A, std::size_t r0, std::size_t c0,
                std::size_t m, std::size_t k, const T* x, T* t) {
    if (row_major)
        gemv_rowmajor<T>(m, k, A + r0 * ld + c0, ld, x, t);
    else
        gemv_colmajor<T>(m, k, A + c0 * ld + r0, ld, x, t);
}

} // namespace mtl::detail
