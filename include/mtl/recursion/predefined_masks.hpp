#pragma once
// MTL5 -- Predefined recursion masks for Morton Z-order and related curves
// Port from MTL4: boost/numeric/mtl/recursion/predefined_masks.hpp
//                + boost/numeric/mtl/recursion/bit_masking.hpp
// Key changes: constexpr functions replace Boost.MPL template metaprograms
//
// NOTE: Uses std::uint64_t throughout -- unsigned long is only 32 bits on
// Windows (LLP64 data model), so shifting beyond 31 would be UB there.

#include <cstdint>
#include <stdexcept>

#include <mtl/detail/exceptions.hpp>

namespace mtl::recursion {

// -- Bit manipulation helpers -------------------------------------------

/// Create a mask with the lowest N bits set
constexpr std::uint64_t lsb_mask(unsigned n) {
    if (n >= 64) return ~std::uint64_t{0};
    return (std::uint64_t{1} << n) - std::uint64_t{1};
}

/// Row-major mask for block size K: interleaved bit pattern, the bits
/// [i + k, i + 2k) for i = 0, 2k, 4k, ... below 64.
/// For K=1: 0xAAAA... (odd bits set -- row index bits)
/// For K=2: 0xCCCC... (pairs of bits for row)
/// For K >= 32 there is one block, clipped at bit 63 (K = 33: bits 33..63);
/// for K >= 64 none (0).
/// K = 0 has no blocks and the loop did not advance: it is rejected, which
/// in a constant expression is a compile error (issue 877).
constexpr std::uint64_t row_major_mask(unsigned k) {
    if (k == 0) {
        ::mtl::detail::throw_or_abort(std::invalid_argument("row_major_mask: block size k must be at least 1"));
        return 0;
    }
    std::uint64_t mask = 0;
    for (unsigned i = 0; i + k < 64; i += 2 * k) {
        mask |= (lsb_mask(k) << (i + k));
    }
    return mask;
}

/// Column-major mask for block size K: complement of row-major
constexpr std::uint64_t col_major_mask(unsigned k) {
    return ~row_major_mask(k);
}

// -- Morton Z-order masks -----------------------------------------------

/// Morton Z-order mask: alternating bits for row/column interleaving
/// Bit 0 = column, Bit 1 = row, Bit 2 = column, Bit 3 = row, ...
inline constexpr std::uint64_t morton_z_mask = UINT64_C(0x5555555555555555);

/// Complement of morton_z_mask (row bits in Z-order)
inline constexpr std::uint64_t morton_mask   = ~morton_z_mask;

// -- Doppled (block-interleaved) masks at various block sizes -----------

/// Doppled row-major: blocks of 2 bits for row indices
inline constexpr std::uint64_t doppled_2_row_mask  = row_major_mask(1);
inline constexpr std::uint64_t doppled_2_col_mask  = col_major_mask(1);

/// Doppled with block size 4
inline constexpr std::uint64_t doppled_4_row_mask  = row_major_mask(2);
inline constexpr std::uint64_t doppled_4_col_mask  = col_major_mask(2);

/// Doppled with block size 16
inline constexpr std::uint64_t doppled_16_row_mask = row_major_mask(4);
inline constexpr std::uint64_t doppled_16_col_mask = col_major_mask(4);

/// Doppled with block size 32
inline constexpr std::uint64_t doppled_32_row_mask = row_major_mask(5);
inline constexpr std::uint64_t doppled_32_col_mask = col_major_mask(5);

/// Doppled with block size 64
inline constexpr std::uint64_t doppled_64_row_mask = row_major_mask(6);
inline constexpr std::uint64_t doppled_64_col_mask = col_major_mask(6);

/// Doppled with block size 128
inline constexpr std::uint64_t doppled_128_row_mask = row_major_mask(7);
inline constexpr std::uint64_t doppled_128_col_mask = col_major_mask(7);

// The shark-tooth masks (shark_z_row_mask, shark_z_col_mask) were removed in
// issue 877: described as "hybrid block + Z-order", they were plain Morton
// aliases, with shark_z_row_mask = 0x5555... -- the column bits by this
// file's convention, against doppled_2_row_mask = 0xAAAA... -- and nothing
// used them.

} // namespace mtl::recursion
