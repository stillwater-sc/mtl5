// Tests for the from-first-principles PNG writer (#252, batch 1).
// Validates CRC-32 / Adler-32 against known vectors, then round-trips written
// PNGs by parsing the container, verifying every chunk CRC and the Adler-32,
// decoding the stored DEFLATE blocks, and comparing pixels.
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <mtl/io/png.hpp>
#include "png_decode.hpp"

using namespace mtl::io;
using mtl_test::decode_png;
using mtl_test::read_file;

namespace {

std::uint32_t crc32_of(const std::string& s) {
    std::uint32_t c = 0xFFFFFFFFu;
    c = detail::crc32_add(c, reinterpret_cast<const std::uint8_t*>(s.data()), s.size());
    return c ^ 0xFFFFFFFFu;
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
