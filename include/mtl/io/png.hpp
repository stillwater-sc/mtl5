#pragma once
// MTL5 -- Minimal, dependency-free PNG writer (#252, batch 1).
//
// Implements the PNG container from first principles -- no libpng, no zlib:
//   * 8-byte signature + length-tagged chunks (IHDR / IDAT / IEND), each with a
//     CRC-32 (PNG polynomial) over its type+data.
//   * IDAT carries a zlib stream whose DEFLATE payload uses only *stored*
//     (uncompressed) blocks, followed by an Adler-32 checksum. Correct framing
//     and the two checksums are all a conforming decoder needs -- no compressor.
//   * 8-bit grayscale (color type 0) and 8-bit RGB (color type 2); each scanline
//     is prefixed with filter byte 0 (None), per the PNG spec.
//
// PNG multi-byte integers are big-endian; DEFLATE stored-block LEN/NLEN are
// little-endian. Bytes are written through a binary std::ofstream.
#include <array>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mtl::io {

namespace detail {

/// CRC-32 lookup table (reflected polynomial 0xEDB88320), built once.
inline const std::array<std::uint32_t, 256>& crc32_table() {
    static const std::array<std::uint32_t, 256> table = [] {
        std::array<std::uint32_t, 256> t{};
        for (std::uint32_t n = 0; n < 256; ++n) {
            std::uint32_t c = n;
            for (int k = 0; k < 8; ++k)
                c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            t[n] = c;
        }
        return t;
    }();
    return table;
}

/// Fold bytes into a running CRC-32 register (internal, pre-inverted form:
/// start from 0xFFFFFFFF and XOR 0xFFFFFFFF once at the end).
inline std::uint32_t crc32_add(std::uint32_t crc, const std::uint8_t* buf, std::size_t len) {
    const auto& t = crc32_table();
    for (std::size_t i = 0; i < len; ++i)
        crc = t[(crc ^ buf[i]) & 0xFFu] ^ (crc >> 8);
    return crc;
}

/// Adler-32 (zlib checksum) of a byte buffer.
inline std::uint32_t adler32(const std::uint8_t* data, std::size_t len) {
    constexpr std::uint32_t MOD = 65521u;   // largest prime < 2^16
    std::uint32_t a = 1, b = 0;
    std::size_t i = 0;
    while (i < len) {
        // 5552 is the most iterations before (a,b) can overflow 32 bits.
        const std::size_t n = std::min<std::size_t>(len - i, 5552);
        for (std::size_t j = 0; j < n; ++j) { a += data[i + j]; b += a; }
        a %= MOD; b %= MOD;
        i += n;
    }
    return (b << 16) | a;
}

inline void put_u32_be(std::vector<std::uint8_t>& v, std::uint32_t x) {
    v.push_back(std::uint8_t((x >> 24) & 0xFF));
    v.push_back(std::uint8_t((x >> 16) & 0xFF));
    v.push_back(std::uint8_t((x >> 8) & 0xFF));
    v.push_back(std::uint8_t(x & 0xFF));
}

/// Wrap raw bytes into a zlib stream of DEFLATE *stored* blocks (no compression).
inline std::vector<std::uint8_t> zlib_store(const std::vector<std::uint8_t>& raw) {
    std::vector<std::uint8_t> out;
    out.push_back(0x78);   // zlib CMF: CM=8 (deflate), CINFO=7 (32K window)
    out.push_back(0x01);   // zlib FLG: FCHECK so (CMF*256+FLG) % 31 == 0, no dict
    const std::size_t n = raw.size();
    std::size_t i = 0;
    if (n == 0) {                          // one empty, final stored block
        out.push_back(0x01);
        out.push_back(0x00); out.push_back(0x00);   // LEN  = 0
        out.push_back(0xFF); out.push_back(0xFF);   // NLEN = ~0
    }
    while (i < n) {
        const std::size_t block = std::min<std::size_t>(n - i, 65535);
        const bool final = (i + block >= n);
        out.push_back(final ? 0x01 : 0x00);         // BFINAL + BTYPE=00 (stored)
        const std::uint16_t len  = static_cast<std::uint16_t>(block);
        const std::uint16_t nlen = static_cast<std::uint16_t>(~len);
        out.push_back(std::uint8_t(len & 0xFF));  out.push_back(std::uint8_t((len >> 8) & 0xFF));
        out.push_back(std::uint8_t(nlen & 0xFF)); out.push_back(std::uint8_t((nlen >> 8) & 0xFF));
        out.insert(out.end(), raw.begin() + i, raw.begin() + i + block);
        i += block;
    }
    put_u32_be(out, adler32(raw.data(), raw.size()));
    return out;
}

/// Write one PNG chunk: length (BE) + type + data + CRC-32(type+data).
inline void write_chunk(std::ofstream& os, const char type[4],
                        const std::vector<std::uint8_t>& data) {
    const std::uint32_t n = static_cast<std::uint32_t>(data.size());
    const std::uint8_t len[4] = {
        std::uint8_t((n >> 24) & 0xFF), std::uint8_t((n >> 16) & 0xFF),
        std::uint8_t((n >> 8) & 0xFF),  std::uint8_t(n & 0xFF)
    };
    os.write(reinterpret_cast<const char*>(len), 4);
    os.write(type, 4);
    if (!data.empty())
        os.write(reinterpret_cast<const char*>(data.data()),
                 static_cast<std::streamsize>(data.size()));

    std::uint32_t crc = 0xFFFFFFFFu;
    crc = crc32_add(crc, reinterpret_cast<const std::uint8_t*>(type), 4);
    if (!data.empty()) crc = crc32_add(crc, data.data(), data.size());
    crc ^= 0xFFFFFFFFu;
    const std::uint8_t crcb[4] = {
        std::uint8_t((crc >> 24) & 0xFF), std::uint8_t((crc >> 16) & 0xFF),
        std::uint8_t((crc >> 8) & 0xFF),  std::uint8_t(crc & 0xFF)
    };
    os.write(reinterpret_cast<const char*>(crcb), 4);
}

/// Core writer: `channels` is 1 (grayscale) or 3 (RGB); `pixels` is row-major,
/// w*h*channels bytes, top row first.
inline void write_png(const std::filesystem::path& path, const std::uint8_t* pixels,
                      std::size_t w, std::size_t h, int channels) {
    if (w == 0 || h == 0)
        throw std::runtime_error("write_png: zero image dimension");
    if (channels != 1 && channels != 3)
        throw std::runtime_error("write_png: channels must be 1 (gray) or 3 (RGB)");

    std::ofstream os(path, std::ios::binary);
    if (!os)
        throw std::runtime_error("write_png: cannot open file: " + path.string());

    static const std::uint8_t signature[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    os.write(reinterpret_cast<const char*>(signature), 8);

    std::vector<std::uint8_t> ihdr;
    put_u32_be(ihdr, static_cast<std::uint32_t>(w));
    put_u32_be(ihdr, static_cast<std::uint32_t>(h));
    ihdr.push_back(8);                                  // bit depth
    ihdr.push_back(channels == 1 ? std::uint8_t(0) : std::uint8_t(2)); // 0 gray, 2 RGB
    ihdr.push_back(0);                                  // compression method (deflate)
    ihdr.push_back(0);                                  // filter method
    ihdr.push_back(0);                                  // interlace (none)
    write_chunk(os, "IHDR", ihdr);

    // Assemble the raw scanline stream: one filter byte (0 = None) per row.
    const std::size_t stride = w * static_cast<std::size_t>(channels);
    std::vector<std::uint8_t> raw;
    raw.reserve(h * (1 + stride));
    for (std::size_t y = 0; y < h; ++y) {
        raw.push_back(0);
        const std::uint8_t* row = pixels + y * stride;
        raw.insert(raw.end(), row, row + stride);
    }
    write_chunk(os, "IDAT", zlib_store(raw));
    write_chunk(os, "IEND", {});

    if (!os)
        throw std::runtime_error("write_png: write failed: " + path.string());
}

} // namespace detail

/// Write an 8-bit grayscale PNG. `pixels` is w*h bytes, row-major, top row first.
inline void write_png_gray(const std::filesystem::path& path,
                           const std::uint8_t* pixels, std::size_t w, std::size_t h) {
    detail::write_png(path, pixels, w, h, 1);
}

/// Write an 8-bit RGB PNG. `rgb` is w*h*3 bytes (R,G,B per pixel), row-major.
inline void write_png_rgb(const std::filesystem::path& path,
                          const std::uint8_t* rgb, std::size_t w, std::size_t h) {
    detail::write_png(path, rgb, w, h, 3);
}

} // namespace mtl::io
