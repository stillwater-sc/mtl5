#pragma once
// Shared test helper: a minimal PNG decoder for the subset the MTL5 writer emits
// (8-bit grayscale / RGB, filter 0, stored DEFLATE blocks). Verifies the chunk
// sequence, every chunk CRC-32, and the Adler-32, then returns the pixel grid.
// Used by test_png.cpp and test_spy.cpp to validate output beyond its signature.
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include <mtl/io/png.hpp>

namespace mtl_test {

inline std::vector<std::uint8_t> read_file(const std::filesystem::path& p) {
    std::ifstream in(p, std::ios::binary);
    return std::vector<std::uint8_t>((std::istreambuf_iterator<char>(in)),
                                     std::istreambuf_iterator<char>());
}

inline std::uint32_t rd_be(const std::vector<std::uint8_t>& b, std::size_t i) {
    return (std::uint32_t(b[i]) << 24) | (std::uint32_t(b[i + 1]) << 16) |
           (std::uint32_t(b[i + 2]) << 8) | std::uint32_t(b[i + 3]);
}

struct DecodedPng {
    std::size_t w = 0, h = 0;
    int channels = 0;
    std::vector<std::uint8_t> pixels;   // row-major, w*h*channels bytes
};

// Parse + validate; failures trip Catch REQUIREs (call from within a TEST_CASE).
// Enforces chunk ordering (IHDR first and unique, IDAT before IEND, IEND empty
// and final with no trailing bytes) and bounds-checks every field before reading
// it, so a malformed writer output asserts rather than being accepted or crashing.
inline DecodedPng decode_png(const std::vector<std::uint8_t>& f) {
    using mtl::io::detail::crc32_add;
    using mtl::io::detail::adler32;

    static const std::uint8_t sig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    REQUIRE(f.size() >= 8);
    for (int i = 0; i < 8; ++i) REQUIRE(f[i] == sig[i]);

    DecodedPng out;
    std::vector<std::uint8_t> idat;
    std::size_t pos = 8;
    bool seen_ihdr = false, seen_idat = false, seen_iend = false, first = true;
    while (pos < f.size()) {
        REQUIRE(!seen_iend);                              // nothing may follow IEND
        REQUIRE(pos + 8 <= f.size());                     // length + type
        const std::uint32_t len = rd_be(f, pos);
        const std::string ctype(reinterpret_cast<const char*>(&f[pos + 4]), 4);
        const std::size_t data_at = pos + 8;
        REQUIRE(data_at + static_cast<std::size_t>(len) + 4 <= f.size());  // data + CRC

        std::uint32_t crc = 0xFFFFFFFFu;
        crc = crc32_add(crc, &f[pos + 4], 4);
        if (len) crc = crc32_add(crc, &f[data_at], len);
        crc ^= 0xFFFFFFFFu;
        REQUIRE(crc == rd_be(f, data_at + len));

        if (first) { REQUIRE(ctype == "IHDR"); first = false; }

        if (ctype == "IHDR") {
            REQUIRE(!seen_ihdr);                          // unique
            REQUIRE(len == 13);
            seen_ihdr = true;
            out.w = rd_be(f, data_at);
            out.h = rd_be(f, data_at + 4);
            REQUIRE(f[data_at + 8] == 8);                 // bit depth
            const std::uint8_t color = f[data_at + 9];
            REQUIRE((color == 0 || color == 2));
            out.channels = (color == 0) ? 1 : 3;
        } else {
            REQUIRE(seen_ihdr);                           // IHDR precedes all others
            if (ctype == "IDAT") {
                seen_idat = true;
                idat.insert(idat.end(), &f[data_at], &f[data_at] + len);
            } else if (ctype == "IEND") {
                REQUIRE(len == 0);                        // IEND carries no data
                seen_iend = true;
            }
        }
        pos = data_at + len + 4;
    }
    REQUIRE(seen_ihdr);
    REQUIRE(seen_idat);
    REQUIRE(seen_iend);
    REQUIRE(pos == f.size());                             // no trailing bytes

    // Inflate the stored DEFLATE blocks; bounds-check then verify Adler-32.
    std::vector<std::uint8_t> raw;
    REQUIRE(idat.size() >= 2);                            // zlib header
    std::size_t p = 2;
    for (;;) {
        REQUIRE(p + 1 <= idat.size());                    // block header byte
        const std::uint8_t hdr = idat[p++];
        REQUIRE((hdr & 0x06) == 0);                       // BTYPE == 00 (stored)
        REQUIRE(p + 4 <= idat.size());                    // LEN + NLEN
        const std::uint32_t blen = std::uint32_t(idat[p]) | (std::uint32_t(idat[p + 1]) << 8);
        const std::uint32_t nlen = std::uint32_t(idat[p + 2]) | (std::uint32_t(idat[p + 3]) << 8);
        REQUIRE((blen ^ 0xFFFFu) == nlen);
        p += 4;
        REQUIRE(p + blen <= idat.size());                 // payload
        raw.insert(raw.end(), &idat[p], &idat[p] + blen);
        p += blen;
        if (hdr & 0x01) break;                            // BFINAL
    }
    REQUIRE(p + 4 <= idat.size());                        // Adler-32
    REQUIRE(rd_be(idat, p) == adler32(raw.data(), raw.size()));
    REQUIRE(p + 4 == idat.size());                        // no trailing zlib bytes

    // Strip per-scanline filter bytes (all filter 0 = None).
    const std::size_t stride = out.w * static_cast<std::size_t>(out.channels);
    REQUIRE(raw.size() == out.h * (1 + stride));
    out.pixels.reserve(out.h * stride);
    for (std::size_t y = 0; y < out.h; ++y) {
        REQUIRE(raw[y * (1 + stride)] == 0);
        out.pixels.insert(out.pixels.end(),
                          &raw[y * (1 + stride) + 1], &raw[y * (1 + stride) + 1] + stride);
    }
    return out;
}

} // namespace mtl_test
