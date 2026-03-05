#pragma once
// MTL5 — Predefined recursion masks for Morton Z-order and related curves
// Port from MTL4: boost/numeric/mtl/recursion/predefined_masks.hpp
//                + boost/numeric/mtl/recursion/bit_masking.hpp
// Key changes: constexpr functions replace Boost.MPL template metaprograms

#include <cstddef>

namespace mtl::recursion {

// ── Bit manipulation helpers ───────────────────────────────────────────

/// Create a mask with the lowest N bits set
constexpr unsigned long lsb_mask(unsigned n) {
    if (n >= 64) return ~0UL;
    return (1UL << n) - 1UL;
}

/// Row-major mask for block size K: interleaved bit pattern
/// For K=1: 0xAAAA... (even bits set — row index bits)
/// For K=2: 0xCCCC... (pairs of bits for row)
constexpr unsigned long row_major_mask(unsigned k) {
    unsigned long mask = 0;
    for (unsigned i = 0; i + k < 64; i += 2 * k) {
        mask |= (lsb_mask(k) << (i + k));
    }
    return mask;
}

/// Column-major mask for block size K: complement of row-major
constexpr unsigned long col_major_mask(unsigned k) {
    return ~row_major_mask(k) & ~0UL;
}

// ── Morton Z-order masks ───────────────────────────────────────────────

/// Morton Z-order mask: alternating bits for row/column interleaving
/// Bit 0 = column, Bit 1 = row, Bit 2 = column, Bit 3 = row, ...
inline constexpr unsigned long morton_z_mask = 0x5555'5555'5555'5555UL;

/// Complement of morton_z_mask (row bits in Z-order)
inline constexpr unsigned long morton_mask   = ~morton_z_mask;

// ── Doppled (block-interleaved) masks at various block sizes ───────────

/// Doppled row-major: blocks of 2 bits for row indices
inline constexpr unsigned long doppled_2_row_mask  = row_major_mask(1);
inline constexpr unsigned long doppled_2_col_mask  = col_major_mask(1);

/// Doppled with block size 4
inline constexpr unsigned long doppled_4_row_mask  = row_major_mask(2);
inline constexpr unsigned long doppled_4_col_mask  = col_major_mask(2);

/// Doppled with block size 16
inline constexpr unsigned long doppled_16_row_mask = row_major_mask(4);
inline constexpr unsigned long doppled_16_col_mask = col_major_mask(4);

/// Doppled with block size 32
inline constexpr unsigned long doppled_32_row_mask = row_major_mask(5);
inline constexpr unsigned long doppled_32_col_mask = col_major_mask(5);

/// Doppled with block size 64
inline constexpr unsigned long doppled_64_row_mask = row_major_mask(6);
inline constexpr unsigned long doppled_64_col_mask = col_major_mask(6);

/// Doppled with block size 128
inline constexpr unsigned long doppled_128_row_mask = row_major_mask(7);
inline constexpr unsigned long doppled_128_col_mask = col_major_mask(7);

// ── Shark-tooth masks (hybrid block + Z-order) ────────────────────────

/// Shark-tooth: Z-order at fine granularity, row-major at coarse
inline constexpr unsigned long shark_z_row_mask = morton_z_mask;
inline constexpr unsigned long shark_z_col_mask = morton_mask;

} // namespace mtl::recursion
