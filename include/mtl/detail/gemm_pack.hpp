#pragma once
// MTL5 -- GEMM packing routines (#89, epic #82, Phase 2).
//
// Copy A and B blocks into contiguous, aligned, SIMD-friendly buffers laid out
// in the exact order the register micro-kernel (#88) streams them. This is the
// single biggest non-register performance lever in a GotoBLAS/BLIS-style GEMM:
// the hot inner loop then reads unit-stride, cache-line- and page-friendly
// bytes, which slashes TLB misses and lets the hardware prefetcher run.
//
// Two micro-panel layouts, matching the micro-kernel's contract exactly:
//
//   pack_A  -- MR-row column-major panels.   Ap[panel*MR*k + p*MR + i] == A(i,p)
//   pack_B  -- NR-col row-major panels.       Bp[panel*NR*k + p*NR + j] == B(p,j)
//
// where i/j are panel-local indices and p in [0,k). The micro-kernel consumes a
// single mr x kc A-panel as Ap[p*MR+i] and a kc x nr B-panel as Bp[p*NR+j].
//
// Generic strides (BLIS-style): the source is given by a base pointer plus a row
// stride `rs` and column stride `cs`, so row-major (rs=ld, cs=1), column-major
// (rs=1, cs=ld), and transposed views all pack with NO special-casing -- just
// pass the strides of the view. A(i,j) is read from `A[i*rs + j*cs]`.
//
// Edge padding: a trailing partial block (rows < MR, or cols < NR) is zero-padded
// up to the full tile, so the micro-kernel only ever sees full MR x NR tiles and
// needs no inner-loop size checks. Padded multiplies contribute 0 to C.
//
// Buffers are written sequentially front-to-back; the caller allocates them once
// (via the aligned allocator) and reuses them across the blocking loops. Use
// packed_A_size / packed_B_size to size them.

#include <cstddef>
#include <type_traits>

namespace mtl::detail {

/// Elements needed to pack an m x k A block into MR-row panels (rows padded to MR).
constexpr std::size_t packed_A_size(std::size_t m, std::size_t k, std::size_t MR) {
    return ((m + MR - 1) / MR) * MR * k;
}

/// Elements needed to pack a k x n B block into NR-col panels (cols padded to NR).
constexpr std::size_t packed_B_size(std::size_t k, std::size_t n, std::size_t NR) {
    return ((n + NR - 1) / NR) * NR * k;
}

/// Pack an m x k block of A into MR-row, column-major micro-panels.
///
/// Layout: panels are laid out contiguously by row-block; within panel q (rows
/// [q*MR, q*MR+MR)), storage is column-major over the kc dimension --
///     Ap[q*MR*k + p*MR + i] = (q*MR+i < m) ? A(q*MR+i, p) : 0
/// for p in [0,k), i in [0,MR). This is exactly what gemm_microkernel reads as
/// Ap[p*MR + i]. `Ap` must hold packed_A_size(m,k,MR) elements.
template <typename T, std::size_t MR>
void pack_A(const T* A, std::ptrdiff_t rs, std::ptrdiff_t cs,
            std::size_t m, std::size_t k, T* Ap) {
    static_assert(MR > 0, "MR must be positive");
    std::size_t dst = 0;
    for (std::size_t i0 = 0; i0 < m; i0 += MR) {
        const std::size_t mr = (m - i0 < MR) ? (m - i0) : MR;  // rows in this panel
        for (std::size_t p = 0; p < k; ++p) {
            const T* col = A + static_cast<std::ptrdiff_t>(i0) * rs
                             + static_cast<std::ptrdiff_t>(p) * cs;
            for (std::size_t i = 0; i < mr; ++i)
                Ap[dst++] = col[static_cast<std::ptrdiff_t>(i) * rs];
            for (std::size_t i = mr; i < MR; ++i)
                Ap[dst++] = T(0);  // zero-pad trailing rows up to MR
        }
    }
}

/// Pack a k x n block of B into NR-col, row-major micro-panels.
///
/// Layout: panels are laid out contiguously by column-block; within panel q
/// (cols [q*NR, q*NR+NR)), storage is row-major over the kc dimension --
///     Bp[q*NR*k + p*NR + j] = (q*NR+j < n) ? B(p, q*NR+j) : 0
/// for p in [0,k), j in [0,NR). This is exactly what gemm_microkernel reads as
/// Bp[p*NR + j]. `Bp` must hold packed_B_size(k,n,NR) elements.
template <typename T, std::size_t NR>
void pack_B(const T* B, std::ptrdiff_t rs, std::ptrdiff_t cs,
            std::size_t k, std::size_t n, T* Bp) {
    static_assert(NR > 0, "NR must be positive");
    std::size_t dst = 0;
    for (std::size_t j0 = 0; j0 < n; j0 += NR) {
        const std::size_t nr = (n - j0 < NR) ? (n - j0) : NR;  // cols in this panel
        for (std::size_t p = 0; p < k; ++p) {
            const T* row = B + static_cast<std::ptrdiff_t>(p) * rs
                             + static_cast<std::ptrdiff_t>(j0) * cs;
            for (std::size_t j = 0; j < nr; ++j)
                Bp[dst++] = row[static_cast<std::ptrdiff_t>(j) * cs];
            for (std::size_t j = nr; j < NR; ++j)
                Bp[dst++] = T(0);  // zero-pad trailing cols up to NR
        }
    }
}

} // namespace mtl::detail
