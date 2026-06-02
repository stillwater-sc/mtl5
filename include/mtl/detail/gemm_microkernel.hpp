#pragma once
// MTL5 -- register-blocked GEMM micro-kernel (#88, epic #82, Phase 2).
//
// The innermost GEMM kernel: an MR x NR microtile of C is held entirely in
// vector registers while kc successive rank-1 updates are accumulated from a
// packed A micro-panel and a packed B micro-panel. Written once over the SIMD
// abstraction (mtl::simd::batch<T>, #83); MR and NR are compile-time template
// parameters (from mtl::simd::blocking, #85) so the kc loop unrolls and the C
// microtile maps cleanly to registers -- letting the compiler do what hand-
// written assembly does. (BLIS micro-kernel / Eigen gebp.)
//
// Packed layouts (produced by the packing step, #89):
//   Ap -- column-major mr-row micro-panel:  Ap[p*MR + i] == A(i, p)
//   Bp -- row-major    nr-col micro-panel:  Bp[p*NR + j] == B(p, j)
// C is a row-major MR x NR tile with leading dimension ldc, updated in place:
//   C(i, j) += sum_{p < kc} A(i, p) * B(p, j),   0 <= i < MR, 0 <= j < NR.
// (Accumulate form: the caller zeroes or pre-scales C. NR is the vectorized
// dimension and must be a multiple of the SIMD width.)

#include <cstddef>

#include <mtl/simd/batch.hpp>

namespace mtl::detail {

template <typename T, std::size_t MR, std::size_t NR>
void gemm_microkernel(std::size_t kc, const T* Ap, const T* Bp,
                      T* C, std::size_t ldc) {
    using B = simd::batch<T>;
    constexpr std::size_t W = B::size;
    static_assert(NR % W == 0, "NR (the vectorized dimension) must be a multiple of the SIMD width");
    constexpr std::size_t NB = NR / W;   // batch-columns spanning the NR lanes

    // C microtile: MR rows x NB batch-columns, register-resident, zeroed.
    B c[MR][NB];

    for (std::size_t p = 0; p < kc; ++p) {
        // Load this depth's B micro-panel row (NR values = NB batches).
        B b[NB];
        for (std::size_t jb = 0; jb < NB; ++jb)
            b[jb] = B::load_unaligned(Bp + p * NR + jb * W);

        // Rank-1 update: each A(i,p) (broadcast) times the B row, into C.
        const T* ap = Ap + p * MR;
        for (std::size_t i = 0; i < MR; ++i) {
            const B a_i(ap[i]);
            for (std::size_t jb = 0; jb < NB; ++jb)
                c[i][jb] = fma(a_i, b[jb], c[i][jb]);
        }
    }

    // Flush the microtile into C (C += tile).
    for (std::size_t i = 0; i < MR; ++i)
        for (std::size_t jb = 0; jb < NB; ++jb) {
            T* cptr = C + i * ldc + jb * W;
            (B::load_unaligned(cptr) + c[i][jb]).store_unaligned(cptr);
        }
}

} // namespace mtl::detail
