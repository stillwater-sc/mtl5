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
//   col-major A  -> axpy-of-columns: hold a strip of y in registers across all
//                   columns and accumulate (broadcast x[j]) * A[:,j]; y is then
//                   written exactly once.
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
            s0 += a0[j] * xj; s1 += a1[j] * xj; s2 += a2[j] * xj; s3 += a3[j] * xj;
        }
        y[i + 0] = s0; y[i + 1] = s1; y[i + 2] = s2; y[i + 3] = s3;
    }
    for (; i < m; ++i)                            // remaining rows (< MR)
        y[i] = simd::reduce_dot<T>(A + i * lda, x, n);
}

/// y[m] := A[m x n] * x[n], A col-major (columns contiguous, lda = m).
template <typename T>
    requires mtl::Scalar<T>
void gemv_colmajor(std::size_t m, std::size_t n, const T* A, std::size_t lda,
                   const T* x, T* y) {
    using B = simd::batch<T>;
    constexpr std::size_t W = B::size;
    constexpr std::size_t UB = 4;                 // y batches resident per strip
    constexpr std::size_t SB = UB * W;            // rows per strip

    std::size_t i = 0;
    for (; i + SB <= m; i += SB) {                // wide resident y-strip
        B acc0, acc1, acc2, acc3;                 // y-strip in registers, zeroed
        for (std::size_t j = 0; j < n; ++j) {
            const B xj(x[j]);
            const T* col = A + j * lda + i;
            acc0 = fma(xj, B::load_unaligned(col + 0 * W), acc0);
            acc1 = fma(xj, B::load_unaligned(col + 1 * W), acc1);
            acc2 = fma(xj, B::load_unaligned(col + 2 * W), acc2);
            acc3 = fma(xj, B::load_unaligned(col + 3 * W), acc3);
        }
        acc0.store_unaligned(y + i + 0 * W);
        acc1.store_unaligned(y + i + 1 * W);
        acc2.store_unaligned(y + i + 2 * W);
        acc3.store_unaligned(y + i + 3 * W);
    }
    for (; i + W <= m; i += W) {                  // single-batch strips
        B acc;
        for (std::size_t j = 0; j < n; ++j)
            acc = fma(B(x[j]), B::load_unaligned(A + j * lda + i), acc);
        acc.store_unaligned(y + i);
    }
    for (; i < m; ++i) {                          // remaining rows (< W), scalar
        T s = T(0);
        for (std::size_t j = 0; j < n; ++j) s += A[j * lda + i] * x[j];
        y[i] = s;
    }
}

} // namespace mtl::detail
