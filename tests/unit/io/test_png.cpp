// Tests for the from-first-principles PNG writer (#252, batch 1).
// Validates CRC-32 / Adler-32 against known vectors, then round-trips written
// PNGs by parsing the container, verifying every chunk CRC and the Adler-32,
// decoding the stored DEFLATE blocks, and comparing pixels.
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <mtl/io/png.hpp>

using namespace mtl::io;

namespace {

std::uint32_t crc32_of(const std::string& s) {
    std::uint32_t c = 0xFFFFFFFFu;
    c = detail::crc32_add(c, reinterpret_cast<const std::uint8_t*>(s.data()), s.size());
    return c ^ 0xFFFFFFFFu;
}

std::vector<std::uint8_t> read_file(const std::filesystem::path& p) {
    std::ifstream in(p, std::ios::binary);
    return std::vector<std::uint8_t>((std::istreambuf_iterator<char>(in)),
                                     std::istreambuf_iterator<char>());
}

std::uint32_t rd_be(const std::vector<std::uint8_t>& b, std::size_t i) {
    return (std::uint32_t(b[i]) << 24) | (std::uint32_t(b[i + 1]) << 16) |
           (std::uint32_t(b[i + 2]) << 8) | std::uint32_t(b[i + 3]);
}

// Minimal PNG decoder for the subset this writer emits (8-bit gray/RGB, filter 0,
// stored DEFLATE blocks). Verifies chunk CRCs and Adler-32; returns pixels.
struct Decoded {
    std::size_t w = 0, h = 0;
    int channels = 0;
    std::vector<std::uint8_t> pixels;
};

Decoded decode_png(const std::vector<std::uint8_t>& f) {
    static const std::uint8_t sig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    REQUIRE(f.size() > 8);
    for (int i = 0; i < 8; ++i) REQUIRE(f[i] == sig[i]);

    Decoded out;
    std::vector<std::uint8_t> idat;
    std::size_t pos = 8;
    bool seen_iend = false;
    while (pos + 8 <= f.size()) {
        const std::uint32_t len = rd_be(f, pos);
        const char* type = reinterpret_cast<const char*>(&f[pos + 4]);
        const std::string ctype(type, 4);
        const std::size_t data_at = pos + 8;
        REQUIRE(data_at + len + 4 <= f.size());

        // Verify chunk CRC over type+data.
        std::uint32_t crc = 0xFFFFFFFFu;
        crc = detail::crc32_add(crc, &f[pos + 4], 4);
        if (len) crc = detail::crc32_add(crc, &f[data_at], len);
        crc ^= 0xFFFFFFFFu;
        REQUIRE(crc == rd_be(f, data_at + len));

        if (ctype == "IHDR") {
            out.w = rd_be(f, data_at);
            out.h = rd_be(f, data_at + 4);
            REQUIRE(f[data_at + 8] == 8);                 // bit depth
            const std::uint8_t color = f[data_at + 9];
            out.channels = (color == 0) ? 1 : 3;
            REQUIRE((color == 0 || color == 2));
        } else if (ctype == "IDAT") {
            idat.insert(idat.end(), &f[data_at], &f[data_at] + len);
        } else if (ctype == "IEND") {
            seen_iend = true;
        }
        pos = data_at + len + 4;
    }
    REQUIRE(seen_iend);

    // Decode the zlib stream: skip 2-byte header, parse stored blocks, check adler.
    std::vector<std::uint8_t> raw;
    std::size_t p = 2;
    for (;;) {
        const std::uint8_t hdr = idat[p++];
        REQUIRE((hdr & 0x06) == 0);                        // BTYPE == 00 (stored)
        const std::uint32_t blen = std::uint32_t(idat[p]) | (std::uint32_t(idat[p + 1]) << 8);
        const std::uint32_t nlen = std::uint32_t(idat[p + 2]) | (std::uint32_t(idat[p + 3]) << 8);
        REQUIRE((blen ^ 0xFFFFu) == nlen);                 // LEN/NLEN complement
        p += 4;
        raw.insert(raw.end(), &idat[p], &idat[p] + blen);
        p += blen;
        if (hdr & 0x01) break;                             // BFINAL
    }
    const std::uint32_t adler = rd_be(idat, p);
    REQUIRE(adler == detail::adler32(raw.data(), raw.size()));

    // Strip per-scanline filter bytes (all filter 0).
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

std::filesystem::path tmp(const std::string& name) {
    return std::filesystem::temp_directory_path() / name;
}

} // namespace

TEST_CASE("crc32 and adler32 match known vectors", "[io][png]") {
    // Standard CRC-32 of "123456789".
    REQUIRE(crc32_of("123456789") == 0xCBF43926u);
    // Standard Adler-32 of "Wikipedia".
    const std::string w = "Wikipedia";
    REQUIRE(detail::adler32(reinterpret_cast<const std::uint8_t*>(w.data()), w.size())
            == 0x11E60398u);
    // Adler-32 of the empty string is 1.
    REQUIRE(detail::adler32(nullptr, 0) == 1u);
}

TEST_CASE("grayscale PNG round-trips", "[io][png]") {
    const std::size_t w = 4, h = 3;
    std::vector<std::uint8_t> px(w * h);
    for (std::size_t i = 0; i < px.size(); ++i) px[i] = std::uint8_t(i * 20);

    const auto path = tmp("mtl5_png_gray.png");
    write_png_gray(path, px.data(), w, h);
    const auto dec = decode_png(read_file(path));

    REQUIRE(dec.w == w);
    REQUIRE(dec.h == h);
    REQUIRE(dec.channels == 1);
    REQUIRE(dec.pixels == px);
    std::filesystem::remove(path);
}

TEST_CASE("RGB PNG round-trips", "[io][png]") {
    const std::size_t w = 2, h = 2;
    std::vector<std::uint8_t> rgb = {
        255, 0, 0,   0, 255, 0,
        0, 0, 255,   10, 20, 30
    };
    const auto path = tmp("mtl5_png_rgb.png");
    write_png_rgb(path, rgb.data(), w, h);
    const auto dec = decode_png(read_file(path));

    REQUIRE(dec.channels == 3);
    REQUIRE(dec.w == w);
    REQUIRE(dec.h == h);
    REQUIRE(dec.pixels == rgb);
    std::filesystem::remove(path);
}

TEST_CASE("1x1 PNG", "[io][png]") {
    std::uint8_t one = 128;
    const auto path = tmp("mtl5_png_1x1.png");
    write_png_gray(path, &one, 1, 1);
    const auto dec = decode_png(read_file(path));
    REQUIRE(dec.w == 1);
    REQUIRE(dec.h == 1);
    REQUIRE(dec.pixels.size() == 1);
    REQUIRE(dec.pixels[0] == 128);
    std::filesystem::remove(path);
}

TEST_CASE("large PNG spans multiple stored blocks", "[io][png]") {
    // 300x300 gray -> raw = 300*(1+300) = 90300 bytes > 65535 -> multiple blocks.
    const std::size_t w = 300, h = 300;
    std::vector<std::uint8_t> px(w * h);
    for (std::size_t i = 0; i < px.size(); ++i) px[i] = std::uint8_t((i * 7) & 0xFF);

    const auto path = tmp("mtl5_png_large.png");
    write_png_gray(path, px.data(), w, h);
    const auto dec = decode_png(read_file(path));

    REQUIRE(dec.w == w);
    REQUIRE(dec.h == h);
    REQUIRE(dec.pixels == px);
    std::filesystem::remove(path);
}

TEST_CASE("invalid dimensions throw", "[io][png]") {
    std::uint8_t dummy = 0;
    REQUIRE_THROWS(write_png_gray(tmp("mtl5_png_bad.png"), &dummy, 0, 1));
    REQUIRE_THROWS(write_png_gray(tmp("mtl5_png_bad.png"), &dummy, 1, 0));
}
