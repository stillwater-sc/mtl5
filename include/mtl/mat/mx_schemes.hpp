#pragma once
// MTL5 -- OCP microscaling (MX) block schemes (#629).
//
// The OCP Microscaling Formats specification (v1.0) defines a block as k = 32
// elements sharing one E8M0 scale X, each element P_i decoding to X * P_i. Two
// of its element types are here, as BlockSchemes for block_matrix_view:
//
//   mxint8   32 x INT8  (two's complement, implicit scale 2^-6: value q / 64)
//   mxfp4    32 x FP4 E2M1 (+-{0, 0.5, 1, 1.5, 2, 3, 4, 6}; no Inf or NaN)
//
// THE BYTE LAYOUT IS OURS TO STATE. The specification defines the encodings
// but not how a block is laid out in memory, so each scheme says so below.
// Both put the scale byte first. mxfp4 follows GGML's block_mxfp4 (llama.cpp's
// GGML_TYPE_MXFP4, used by gpt-oss GGUF files), so such data can be viewed in
// place: element j is the LOW nibble of byte 1 + j and element j + 16 its HIGH
// nibble, for j < 16. GGML has no MXINT8 type; mxint8 is the scale byte
// followed by the 32 elements in order.
//
// THE SCALE. E8M0 is a bare biased exponent: X = 2^(e - 127) for e in
// [0, 254], and e = 0xFF is NaN, which makes every element of the block NaN.
// The decode is to float, which covers that range at the low end (2^-127 is a
// float subnormal) but not quite at the top: under X = 2^127, the largest
// scale, any element of magnitude 2 or more exceeds FLT_MAX (just under 2^128)
// and decodes to infinity. For mxfp4 that is the codes 2, 3, 4 and 6; for
// mxint8 it is q = -128 alone, which is exactly -2.
//
// MTL5 stays Universal-free, so these are decoded by hand rather than through
// Universal's mxfloat. They are small enough to check exhaustively, and
// tests/unit/mat/test_block_matrix_view.cpp does.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace mtl::mx {

/// The value of an E8M0 scale byte: 2^(e - 127), or NaN for 0xFF.
inline float e8m0_to_float(std::uint8_t e) noexcept {
    if (e == 0xFF) return std::numeric_limits<float>::quiet_NaN();
    return std::ldexp(1.0f, int(e) - 127);
}

/// The value of an FP4 E2M1 code (low four bits of `code`): sign, two exponent
/// bits with bias 1, one mantissa bit, and a subnormal at exponent 0.
inline float e2m1_to_float(std::uint8_t code) noexcept {
    static constexpr float magnitude[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float m = magnitude[code & 0x7];
    return (code & 0x8) ? -m : m;
}

/// MXINT8: a scale byte, then 32 two's-complement int8 elements. 33 bytes.
struct mxint8 {
    static constexpr std::size_t block_size  = 32;
    static constexpr std::size_t block_bytes = 1 + 32;

    static float decode_element(const std::byte* src, std::size_t j) noexcept {
        const auto e = static_cast<std::uint8_t>(src[0]);
        if (e == 0xFF) return std::numeric_limits<float>::quiet_NaN();
        std::int8_t q;
        std::memcpy(&q, src + 1 + j, 1);
        // q * 2^-6 * 2^(e - 127), in one exact ldexp: q has 8 significant
        // bits, and the smallest nonzero result, 2^-133, is a float subnormal.
        return std::ldexp(static_cast<float>(q), int(e) - 127 - 6);
    }

    static void decode(const std::byte* src, float* dst) noexcept {
        for (std::size_t j = 0; j < block_size; ++j) dst[j] = decode_element(src, j);
    }
};

/// MXFP4: a scale byte, then 16 bytes of packed E2M1 elements in GGML's
/// block_mxfp4 order (element j low nibble, element j + 16 high nibble of
/// byte 1 + j). 17 bytes.
struct mxfp4 {
    static constexpr std::size_t block_size  = 32;
    static constexpr std::size_t block_bytes = 1 + 16;

    static float decode_element(const std::byte* src, std::size_t j) noexcept {
        const auto e = static_cast<std::uint8_t>(src[0]);
        if (e == 0xFF) return std::numeric_limits<float>::quiet_NaN();
        const auto byte = static_cast<std::uint8_t>(src[1 + (j % 16)]);
        const std::uint8_t code = j < 16 ? (byte & 0x0F) : (byte >> 4);
        return std::ldexp(e2m1_to_float(code), int(e) - 127);
    }

    static void decode(const std::byte* src, float* dst) noexcept {
        for (std::size_t j = 0; j < block_size; ++j) dst[j] = decode_element(src, j);
    }
};

} // namespace mtl::mx
