#pragma once
// MTL5 -- GEMM micro-kernel built on the QUAD MULTIPLY-ACCUMULATE (#451 phase 5).
//
// VNNI's `vpdpbusd` and NEON's `SDOT`/`UDOT` fold four 8-bit products into one
// 32-bit accumulator lane per instruction. Every kernel in MTL5 up to now used
// them the way a DOT product does -- two vectors walked in step -- and the
// integer GEMM (#468) did not use them at all: it promoted narrow operands into
// int32 lanes and did an ordinary multiply-add, one k value at a time.
//
// That measurement is what motivated this kernel. On a machine without the
// instruction, narrowing the operand of a GEMM buys exactly nothing (Xeon
// E5-2420 v2, n=512: gemm_i8_i32 13.3 GOP/s against gemm_i32's 13.5 and fp32's
// 18.8), because a GEMM packs each operand once and then reads it from cache
// O(n) times -- its width in memory stops mattering. So everything an int8 GEMM
// can offer over fp32 has to come from the instruction, and using the
// instruction requires this kernel.
//
// WHAT CHANGES, relative to gemm_microkernel:
//
//   * FOUR k VALUES PER INSTRUCTION instead of one. The kc loop steps by 4.
//   * The A operand is a BROADCAST QUAD, not a broadcast scalar: the four bytes
//     A(i, p..p+3) splatted across the register, so lane j of the result gets
//     sum_{q<4} A(i,p+q) * B(p+q, j).
//   * The panels are quad-interleaved (gemm_quad_pack.hpp), which is what puts
//     those four bytes adjacent in the first place.
//   * The operand types may DIFFER. `(uint8, int8)` is VNNI's native shape --
//     unsigned activations against signed weights -- and the reason the
//     instruction exists; symmetric `i8 x i8` is native only from AVX10.2 and is
//     emulated below it. See simd::quad_accumulator.
//
// WHAT DOES NOT CHANGE: the C microtile is still MR x NR int32 accumulators held
// in registers across the whole kc panel, MR and NR still come from
// simd::default_blocking<TC>, and the arithmetic is still exact mod 2^32 -- so a
// quad GEMM is bit-identical across lane counts, backends and thread partitions,
// which no floating-point GEMM can claim.
//
// The instruction-per-load ratio is the practical win. Per MR x NR tile the
// pairwise kernel issues MR broadcasts and NR/W loads for ONE k step; this one
// issues the same MR broadcasts and NR/W loads for FOUR, so the same
// multiply-accumulate throughput arrives with a quarter of the operand traffic
// into the register file.
//
// A KNOWN, UNMEASURED LEVER: kc is still derived for a TC-wide B micro-panel.
// derive_blocking sizes it so `kc x nr` fills about half of L1 at
// sizeof(TC) bytes per element, but the panel this kernel reads holds 8-bit
// operands -- a QUARTER of that, so the real occupancy is ~1/8 of L1 rather than
// ~1/2. kc could therefore be several times larger before the panel stops
// fitting. The same mismatch already applies to the widen-on-load int8 path, and
// it is left alone deliberately: #430 and #453 both found that moving a
// cache-derived parameter moves BLOCK COUNTS the thread partition is sensitive
// to, and that the analytical sizing lost to the shipped constants on real
// hardware more often than it won. So this gets its own measurement, on a
// machine with the instruction, rather than riding along with the kernel.

#include <cstddef>
#include <type_traits>

#include <mtl/simd/batch.hpp>

namespace mtl::detail {

/// Which micro-kernel the blocked nest drives.
///
/// An EXPLICIT choice, never inferred from the element types, because an
/// (i8, i8) pair is valid input to both and inference would silently replace the
/// widen-on-load path the moment this kernel existed. Both are correct; they
/// differ only in speed, and which one a build measured is not recoverable from
/// a timing afterwards. So the caller says which, and the default is the one
/// that was already there.
enum class gemm_kernel {
    widening,   ///< same-type, or narrow operands widened on load: one k value per multiply-add
    quad,       ///< the hardware quad multiply-accumulate: four k values per instruction
};

/// Does the (accumulator, A operand, B operand) triple ADMIT the quad kernel?
/// Necessary for `gemm_kernel::quad`, never sufficient to select it -- see above.
///
/// A function rather than a variable template because `quad_accumulator_t` is
/// undefined for a rejected pairing -- notably `(int8, uint8)`, which is absent
/// on purpose so that a caller holding it is told to swap rather than handed the
/// emulated path silently. Naming the trait unconditionally would turn that
/// deliberate absence into an "incomplete type" error at every use site.
template <typename TC, typename TA, typename TB>
constexpr bool is_quad_gemm() {
    if constexpr (simd::QuadPair<TA, TB>)
        return std::is_same_v<TC, simd::quad_accumulator_t<TA, TB>>;
    else
        return false;
}

/// Accumulate an MR x NR tile of C from quad-interleaved packed panels:
///
///     C(i, j) += alpha * sum_{p < kp} A(i, p) * B(p, j)
///
/// `kp` is the panel depth, a multiple of four (`quad_depth(kc)`); the packing
/// zero-pads the k tail up to it, and a zero operand contributes a zero product,
/// so no size check reaches the inner loop. `Ap` and `Bp` are laid out by
/// pack_A_quad / pack_B_quad. C is row-major with leading dimension `ldc`, and
/// this is the ACCUMULATE form -- the caller zeroes or pre-scales it.
///
/// ALPHA IS APPLIED HERE, on the way out, and not folded into the packed A panel
/// the way the pairwise nest does it. Folding is wrong for this kernel by
/// construction: the panel holds 8-bit operands, so `alpha * A` would be
/// truncated back into a byte before it ever reached the accumulator. Scaling
/// the int32 tile instead is exact (mod 2^32, like everything here) and costs
/// one multiply per C element per kc block, which is O(1/kc) of the work.
template <typename TC, typename TA, typename TB, std::size_t MR, std::size_t NR>
void gemm_quad_microkernel(std::size_t kp, const TA* Ap, const TB* Bp,
                           TC* C, std::size_t ldc, TC alpha) {
    using B32 = simd::batch<TC>;
    static_assert(is_quad_gemm<TC, TA, TB>(),
                  "the quad micro-kernel takes an 8-bit operand pair the hardware "
                  "op accepts -- (i8,i8), (u8,i8) or (u8,u8) -- accumulated in its "
                  "matching 32-bit type");
    constexpr std::size_t W = B32::size;
    static_assert(NR % W == 0, "NR (the vectorized dimension) must be a multiple of the SIMD width");
    constexpr std::size_t NB = NR / W;   // batch-columns spanning the NR lanes

    // C microtile: MR rows x NB batch-columns, register-resident, zeroed.
    B32 c[MR][NB];

    const std::size_t ngroups = kp / 4;
    for (std::size_t g = 0; g < ngroups; ++g) {
        const TA* ag = Ap + g * (MR * 4);   // MR quads of A, one per tile row
        const TB* bg = Bp + g * (NR * 4);   // NR quads of B, one per tile column

        // jb outermost so the B pointer is loop-invariant across the MR
        // accumulations that consume it -- the same reason the pairwise kernel
        // hoists its B row into registers before the rank-1 update.
        for (std::size_t jb = 0; jb < NB; ++jb) {
            // Columns [jb*W, jb*W + W) occupy 4*W adjacent narrow values, which
            // is exactly one instruction's right operand.
            const TB* bptr = bg + jb * (W * 4);
            for (std::size_t i = 0; i < MR; ++i)
                B32::template quad_dot_broadcast_accumulate<TA, TB>(ag + i * 4, bptr, c[i][jb]);
        }
    }

    // Flush the microtile into C (C += alpha * tile).
    const bool scale = !(alpha == TC(1));
    const B32 va(alpha);
    for (std::size_t i = 0; i < MR; ++i)
        for (std::size_t jb = 0; jb < NB; ++jb) {
            TC* cptr = C + i * ldc + jb * W;
            const B32 acc = scale ? va * c[i][jb] : c[i][jb];
            (B32::load_unaligned(cptr) + acc).store_unaligned(cptr);
        }
}

} // namespace mtl::detail
