// Tests for the sparse-matrix spy visualizations (#252, batch 2).
// Assert on the deterministic binning grid (build_spy_grid) across CRS / COO /
// ELL / dense, then confirm each spy variant emits a valid PNG.
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <vector>

#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/coordinate2D.hpp>
#include <mtl/mat/ell_matrix.hpp>
#include <mtl/io/spy.hpp>
#include "png_decode.hpp"

using namespace mtl;
using mtl::io::spy_options;
using mtl_test::decode_png;
using mtl_test::read_file;

namespace {

std::filesystem::path tmp(const std::string& name) {
    return std::filesystem::temp_directory_path() / name;
}

// Build a 4x4 tridiagonal matrix in COO (then convertible to CRS/ELL).
mat::coordinate2D<double> tridiag_coo(std::size_t n) {
    mat::coordinate2D<double> A(n, n);
    for (std::size_t i = 0; i < n; ++i) {
        A.insert(i, i, 2.0);
        if (i + 1 < n) { A.insert(i, i + 1, -1.0); A.insert(i + 1, i, -1.0); }
    }
    return A;
}

} // namespace

TEST_CASE("build_spy_grid: 1:1 mapping marks the right cells", "[io][spy]") {
    // 4x4 tridiagonal, no down-sampling -> W=H=4, one cell per entry.
    auto A = tridiag_coo(4);
    auto g = io::detail::build_spy_grid(A, 1024);
    REQUIRE(g.W == 4);
    REQUIRE(g.H == 4);
    // A tridiagonal has non-zeros exactly on |i-j| <= 1.
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j) {
            const bool expect = (i == j) || (i + 1 == j) || (j + 1 == i);
            REQUIRE((g.counts[i * 4 + j] > 0) == expect);
        }
    // Diagonal magnitude 2, off-diagonal 1.
    REQUIRE(g.maxabs[0 * 4 + 0] == 2.0);
    REQUIRE(g.maxabs[0 * 4 + 1] == 1.0);
}

TEST_CASE("for_each_nonzero agrees across CRS / COO / ELL / dense", "[io][spy]") {
    auto coo = tridiag_coo(5);
    auto crs = coo.compress();
    mat::ell_matrix<double> ell(crs);

    mat::dense2D<double> dense(5, 5);
    for (std::size_t i = 0; i < 5; ++i)
        for (std::size_t j = 0; j < 5; ++j) dense(i, j) = crs(i, j);

    auto g_coo   = io::detail::build_spy_grid(coo, 1024);
    auto g_crs   = io::detail::build_spy_grid(crs, 1024);
    auto g_ell   = io::detail::build_spy_grid(ell, 1024);
    auto g_dense = io::detail::build_spy_grid(dense, 1024);

    REQUIRE(g_crs.counts == g_coo.counts);
    REQUIRE(g_ell.counts == g_coo.counts);
    REQUIRE(g_dense.counts == g_coo.counts);
    REQUIRE(g_crs.maxabs == g_dense.maxabs);
    REQUIRE(g_ell.maxabs == g_dense.maxabs);
}

TEST_CASE("build_spy_grid: down-sampling bins non-zeros", "[io][spy]") {
    // 100x100 dense-diagonal, max_pixels=10 -> 10x10 grid, diagonal cells only,
    // each diagonal cell holds 10 non-zeros.
    mat::coordinate2D<double> D(100, 100);
    for (std::size_t i = 0; i < 100; ++i) D.insert(i, i, 1.0);
    auto g = io::detail::build_spy_grid(D, 10);
    REQUIRE(g.W == 10);
    REQUIRE(g.H == 10);
    for (std::size_t i = 0; i < 10; ++i)
        for (std::size_t j = 0; j < 10; ++j)
            REQUIRE(g.counts[i * 10 + j] == (i == j ? 10u : 0u));
}

TEST_CASE("spy emits a valid PNG whose pixels match the pattern", "[io][spy]") {
    // 5x5 tridiagonal, spy -> 5x5 grayscale; decode and check black marks land
    // exactly on |i-j| <= 1, white elsewhere.
    auto A = tridiag_coo(5);
    const auto path = tmp("mtl5_spy.png");
    io::spy(A, path);

    const auto dec = decode_png(read_file(path));   // verifies CRCs + Adler-32
    REQUIRE(dec.w == 5);
    REQUIRE(dec.h == 5);
    REQUIRE(dec.channels == 1);
    for (std::size_t i = 0; i < 5; ++i)
        for (std::size_t j = 0; j < 5; ++j) {
            const bool mark = (i == j) || (i + 1 == j) || (j + 1 == i);
            REQUIRE(dec.pixels[i * 5 + j] == (mark ? 0 : 255));
        }
    std::filesystem::remove(path);
}

TEST_CASE("spy_magnitude / spy_density emit valid RGB PNGs", "[io][spy]") {
    auto A = tridiag_coo(8);
    const auto p2 = tmp("mtl5_spy_mag.png");
    const auto p3 = tmp("mtl5_spy_den.png");

    io::spy_magnitude(A, p2, {64, /*log_scale=*/true});
    io::spy_density(A, p3);

    const auto d2 = decode_png(read_file(p2));   // full structural validation
    const auto d3 = decode_png(read_file(p3));
    REQUIRE(d2.channels == 3);
    REQUIRE(d3.channels == 3);
    REQUIRE(d2.w == 8); REQUIRE(d2.h == 8);
    REQUIRE(d3.w == 8); REQUIRE(d3.h == 8);

    std::filesystem::remove(p2);
    std::filesystem::remove(p3);
}

TEST_CASE("spy skips explicitly-stored zeros", "[io][spy]") {
    // A COO with an explicit stored zero must not produce a mark there, matching
    // the dense (value != 0) semantics.
    mat::coordinate2D<double> A(2, 2);
    A.insert(0, 0, 1.0);
    A.insert(1, 1, 0.0);   // explicit zero -> not a structural non-zero
    auto g = io::detail::build_spy_grid(A, 1024);
    REQUIRE(g.counts[0] == 1);   // (0,0)
    REQUIRE(g.counts[3] == 0);   // (1,1) skipped
}

TEST_CASE("spy rejects a zero-dimension matrix", "[io][spy]") {
    mat::coordinate2D<double> Z(0, 0);
    REQUIRE_THROWS(io::spy(Z, tmp("mtl5_spy_bad.png")));
}

// ---------------------------------------------------------------------------
// #843: the coloring scales, option edge cases, shapes and value types.

namespace {

/// The RGB the colormap gives t.
std::vector<std::uint8_t> color(double t) {
    std::uint8_t r, g, b;
    io::detail::colormap(t, r, g, b);
    return {r, g, b};
}

std::vector<std::uint8_t> pixel(const mtl_test::DecodedPng& d, std::size_t x, std::size_t y) {
    const std::size_t o = (y * d.w + x) * 3;
    return {d.pixels[o], d.pixels[o + 1], d.pixels[o + 2]};
}

} // namespace

TEST_CASE("spy_magnitude: linear and log10 coloring", "[io][spy]") {
    // 1x3 with |values| 1, 10, 100: linear t = 0.01, 0.1, 1; log t = 0, 0.5, 1.
    mat::dense2D<double> A(1, 3);
    A(0, 0) = 1.0; A(0, 1) = -10.0; A(0, 2) = 100.0;
    const auto p = tmp("mtl5_spy_mag_scale.png");

    io::spy_magnitude(A, p);
    auto lin = decode_png(read_file(p));
    REQUIRE((lin.w == 3 && lin.h == 1));
    REQUIRE(pixel(lin, 0, 0) == color(0.01));
    REQUIRE(pixel(lin, 1, 0) == color(0.1));
    REQUIRE(pixel(lin, 2, 0) == color(1.0));

    io::spy_magnitude(A, p, {1024, /*log_scale=*/true});
    auto lg = decode_png(read_file(p));
    REQUIRE(pixel(lg, 0, 0) == color(0.0));
    REQUIRE(pixel(lg, 1, 0) == color(0.5));
    REQUIRE(pixel(lg, 2, 0) == color(1.0));
    std::filesystem::remove(p);
}

TEST_CASE("spy_density: linear and log10 coloring of binned counts", "[io][spy]") {
    // 4x4 binned into 2x2 cells: counts 4 (full block), 1, 0, 2.
    mat::coordinate2D<double> A(4, 4);
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 2; ++j) A.insert(i, j, 1.0);   // cell (0,0): 4
    A.insert(0, 3, 1.0);                                           // cell (1,0): 1
    A.insert(3, 3, 1.0); A.insert(2, 2, 1.0);                      // cell (1,1): 2
    const auto p = tmp("mtl5_spy_den_scale.png");

    io::spy_density(A, p, {2, false});
    auto lin = decode_png(read_file(p));
    REQUIRE((lin.w == 2 && lin.h == 2));
    REQUIRE(pixel(lin, 0, 0) == color(1.0));
    REQUIRE(pixel(lin, 1, 0) == color(0.25));
    REQUIRE(pixel(lin, 0, 1) == std::vector<std::uint8_t>{255, 255, 255});   // empty cell
    REQUIRE(pixel(lin, 1, 1) == color(0.5));

    io::spy_density(A, p, {2, true});
    auto lg = decode_png(read_file(p));
    REQUIRE(pixel(lg, 0, 0) == color(1.0));
    REQUIRE(pixel(lg, 1, 0) == color(std::log10(2.0) / std::log10(5.0)));
    REQUIRE(pixel(lg, 1, 1) == color(std::log10(3.0) / std::log10(5.0)));
    std::filesystem::remove(p);
}

TEST_CASE("spy: max_pixels == 0, 1x1, non-square and float values", "[io][spy]") {
    auto A = tridiag_coo(4);
    REQUIRE_THROWS(io::spy(A, tmp("mtl5_spy_zero_px.png"), {0, false}));

    const auto p = tmp("mtl5_spy_shapes.png");
    mat::dense2D<float> one(1, 1);
    one(0, 0) = 3.5f;
    io::spy(one, p);
    auto d1 = decode_png(read_file(p));
    REQUIRE((d1.w == 1 && d1.h == 1 && d1.pixels[0] == 0));

    // 2x5, so the image is 5 wide and 2 high; float values through spy_magnitude.
    mat::dense2D<float> R(2, 5);
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 5; ++j) R(i, j) = (j == 4 && i == 1) ? 2.0f : 0.0f;
    io::spy_magnitude(R, p);
    auto d2 = decode_png(read_file(p));
    REQUIRE((d2.w == 5 && d2.h == 2));
    REQUIRE(pixel(d2, 4, 1) == color(1.0));
    REQUIRE(pixel(d2, 0, 0) == std::vector<std::uint8_t>{255, 255, 255});
    std::filesystem::remove(p);
}

TEST_CASE("scale_index: no overflow for huge logical dimensions", "[io][spy]") {
    // idx * span overflows 64 bits here; the result must still be exact.
    const std::size_t dim = std::size_t(1) << 62;
    REQUIRE(io::detail::scale_index(dim / 2, 1000, dim) == 500);
    REQUIRE(io::detail::scale_index(dim - 1, 1000, dim) == 999);
    REQUIRE(io::detail::scale_index(0, 1000, dim) == 0);
    // Clamped to span - 1.
    REQUIRE(io::detail::scale_index(9, 3, 9) == 2);
}
