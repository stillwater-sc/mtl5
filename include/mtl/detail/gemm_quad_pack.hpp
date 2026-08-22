#pragma once
// MTL5 -- QUAD-INTERLEAVED GEMM packing, for the VNNI / SDOT micro-kernel
// (#451 phase 5).
//
// The ordinary packing (gemm_pack.hpp) lays each micro-panel out one k value at
// a time, because the micro-kernel it feeds does one rank-1 update per k:
//
//   pack_A   Ap[panel*MR*k + p*MR + i] == A(i, p)      -- MR-row, column-major
//   pack_B   Bp[panel*NR*k + p*NR + j] == B(p, j)      -- NR-col, row-major
//
// The quad multiply-accumulate consumes FOUR k values per instruction: lane j of
// the accumulator receives sum_{q<4} A(i, p+q) * B(p+q, j). So the four k values
// belonging to one (row, lane) must sit in four ADJACENT bytes -- and that is a
// different layout, not a different traversal of the same one. Interleave k in
// groups of four, innermost:
//
//   pack_A_quad  Ap[panel*MR*KP + g*(MR*4) + i*4 + q] == A(i, 4g+q)
//   pack_B_quad  Bp[panel*NR*KP + g*(NR*4) + j*4 + q] == B(4g+q, j)
//
// with g the k-group index and KP = k rounded up to a multiple of four. The
// micro-kernel then reads A as a 4-byte BROADCAST unit at `i*4` and B as an
// ordinary vector load at `jb*4*W` -- because `j*4 + q` over a run of W columns
// is exactly the 4*W narrow lanes one instruction takes.
//
// Everything else is inherited from gemm_pack.hpp deliberately: generic (rs, cs)
// strides so any orientation packs without special-casing, zero padding of the
// ragged row/column edge so the micro-kernel only ever sees full MR x NR tiles,
// and independent NR-column panels at computable offsets so a team of threads
// can pack one B block cooperatively without changing a single packed byte.
//
// The k TAIL is zero-padded too, which the pairwise layout never had to do: k is
// not a multiple of four in general, and a zero operand contributes a zero
// product, so the padding is free of both special cases and arithmetic effect.

#include <cstddef>

#include <mtl/concepts/scalar.hpp>
#include <mtl/detail/gemm_pack.hpp>

namespace mtl::detail {

/// k rounded up to a whole number of 4-value groups -- the depth the quad
/// layout actually stores, and the depth the micro-kernel steps.
constexpr std::size_t quad_depth(std::size_t k) { return ((k + 3) / 4) * 4; }

/// Elements needed to pack an m x k A block into quad-interleaved MR-row panels.
constexpr std::size_t packed_A_quad_size(std::size_t m, std::size_t k, std::size_t MR) {
    return ((m + MR - 1) / MR) * MR * quad_depth(k);
}

/// Elements needed to pack a k x n B block into quad-interleaved NR-col panels.
constexpr std::size_t packed_B_quad_size(std::size_t k, std::size_t n, std::size_t NR) {
    return ((n + NR - 1) / NR) * NR * quad_depth(k);
}

/// The A-panel size for whichever layout the (accumulator, operand) pair selects.
/// A single call site in the nest then covers both kernels.
template <bool Quad>
constexpr std::size_t packed_A_size_for(std::size_t m, std::size_t k, std::size_t MR) {
    if constexpr (Quad) return packed_A_quad_size(m, k, MR);
    else                return packed_A_size(m, k, MR);
}

/// The B-panel size for whichever layout the pair selects; see packed_A_size_for.
template <bool Quad>
constexpr std::size_t packed_B_size_for(std::size_t k, std::size_t n, std::size_t NR) {
    if constexpr (Quad) return packed_B_quad_size(k, n, NR);
    else                return packed_B_size(k, n, NR);
}

/// Pack an m x k block of A into quad-interleaved MR-row micro-panels.
///
///     Ap[q*MR*KP + g*MR*4 + i*4 + t] = (q*MR+i < m && 4g+t < k) ? A(q*MR+i, 4g+t) : 0
///
/// for g in [0, KP/4), i in [0,MR), t in [0,4), KP = quad_depth(k). `Ap` must
/// hold packed_A_quad_size(m,k,MR) elements. The `i*4 + t` run is the 4-byte
/// unit gemm_quad_microkernel broadcasts.
template <typename T, std::size_t MR>
    requires mtl::Scalar<T>
void pack_A_quad(const T* A, std::ptrdiff_t rs, std::ptrdiff_t cs,
                 std::size_t m, std::size_t k, T* Ap) {
    static_assert(MR > 0, "MR must be positive");
    const std::size_t KP = quad_depth(k);
    std::size_t dst = 0;
    for (std::size_t i0 = 0; i0 < m; i0 += MR) {
        const std::size_t mr = (m - i0 < MR) ? (m - i0) : MR;   // rows in this panel
        for (std::size_t p = 0; p < KP; p += 4) {
            for (std::size_t i = 0; i < MR; ++i) {
                // The whole index is formed INSIDE the guard. Hoisting a `row`
                // pointer out of it would compute A + (i0+i)*rs for a padded
                // row, which is past the end of the source object -- undefined
                // even though the value is never loaded.
                for (std::size_t t = 0; t < 4; ++t)
                    Ap[dst++] = (i < mr && p + t < k)
                        ? A[static_cast<std::ptrdiff_t>(i0 + i) * rs +
                            static_cast<std::ptrdiff_t>(p + t) * cs]
                        : T(0);   // ragged rows and the k tail both pad with zero
            }
        }
    }
}

/// Pack panels [q0, q1) of a k x n B block into quad-interleaved NR-col
/// micro-panels, each written at the offset it would occupy in a whole pack:
///
///     Bp[q*NR*KP + g*NR*4 + j*4 + t] = (q*NR+j < n && 4g+t < k) ? B(4g+t, q*NR+j) : 0
///
/// `Bp` must hold packed_B_quad_size(k,n,NR) elements. Panel destinations are
/// disjoint and computable, and packing is pure data movement, so a cooperative
/// split across a thread team produces byte-identical output -- which is what
/// keeps the threaded quad GEMM bit-identical to the serial one.
template <typename T, std::size_t NR>
    requires mtl::Scalar<T>
void pack_B_quad_panels(const T* B, std::ptrdiff_t rs, std::ptrdiff_t cs,
                        std::size_t k, std::size_t n, T* Bp,
                        std::size_t q0, std::size_t q1) {
    static_assert(NR > 0, "NR must be positive");
    const std::size_t KP = quad_depth(k);
    const std::size_t npanels = (n + NR - 1) / NR;
    if (q1 > npanels) q1 = npanels;
    for (std::size_t q = q0; q < q1; ++q) {
        const std::size_t j0 = q * NR;
        const std::size_t nr = (n - j0 < NR) ? (n - j0) : NR;   // cols in this panel
        std::size_t dst = q * NR * KP;                          // this panel's slot
        for (std::size_t p = 0; p < KP; p += 4) {
            for (std::size_t j = 0; j < NR; ++j) {
                // Index formed inside the guard; see pack_A_quad.
                for (std::size_t t = 0; t < 4; ++t)
                    Bp[dst++] = (j < nr && p + t < k)
                        ? B[static_cast<std::ptrdiff_t>(p + t) * rs +
                            static_cast<std::ptrdiff_t>(j0 + j) * cs]
                        : T(0);
            }
        }
    }
}

template <typename T, std::size_t NR>
    requires mtl::Scalar<T>
void pack_B_quad(const T* B, std::ptrdiff_t rs, std::ptrdiff_t cs,
                 std::size_t k, std::size_t n, T* Bp) {
    pack_B_quad_panels<T, NR>(B, rs, cs, k, n, Bp, 0, (n + NR - 1) / NR);
}

} // namespace mtl::detail
