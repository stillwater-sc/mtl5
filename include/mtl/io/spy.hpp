#pragma once
// MTL5 -- MATLAB-style sparse-matrix visualizations (#252, batch 2).
//
// spy()           binary non-zero pattern       (MATLAB spy)
// spy_magnitude() per-cell max |a_ij| colormap   (imagesc(abs(A)))
// spy_density()   per-cell non-zero count colormap (clustering under downsample)
//
// Renders to PNG via the dependency-free writer in <mtl/io/png.hpp>. Works on
// any matrix through for_each_nonzero: efficient structural-non-zero traversal
// for compressed2D / coordinate2D / ell_matrix, and an (i,j) scan for dense2D
// and other Matrix types. Matrices larger than the target image are down-sampled
// by binning non-zeros into pixel cells.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <vector>

#include <mtl/concepts/magnitude.hpp>
#include <mtl/math/identity.hpp>
#include <mtl/io/png.hpp>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/coordinate2D.hpp>
#include <mtl/mat/ell_matrix.hpp>

namespace mtl::io {

/// Rendering options shared by the spy variants.
struct spy_options {
    std::size_t max_pixels = 1024;   ///< longest image edge; larger matrices down-sample
    bool        log_scale  = false;  ///< magnitude/density coloring on a log scale
};

// -- Structural non-zero traversal: f(row, col, value) --------------------

/// Generic fallback: scan (i,j) and report entries that differ from zero.
/// Used for dense2D and any Matrix type without a specialized overload.
template <typename M, typename F>
void for_each_nonzero(const M& A, F&& f) {
    using value_type = typename M::value_type;
    const auto zero = math::zero<value_type>();
    for (std::size_t i = 0; i < A.num_rows(); ++i)
        for (std::size_t j = 0; j < A.num_cols(); ++j) {
            const value_type v = A(i, j);
            if (v != zero) f(std::size_t(i), std::size_t(j), v);
        }
}

/// CRS: iterate the three arrays directly (O(nnz)).
template <typename Value, typename Parameters, typename F>
void for_each_nonzero(const mat::compressed2D<Value, Parameters>& A, F&& f) {
    const auto& starts  = A.ref_major();
    const auto& indices = A.ref_minor();
    const auto& data    = A.ref_data();
    for (std::size_t i = 0; i < A.num_rows(); ++i)
        for (std::size_t k = starts[i]; k < starts[i + 1]; ++k)
            f(i, std::size_t(indices[k]), data[k]);
}

/// COO: iterate the stored triplets (O(nnz)).
template <typename Value, typename Parameters, typename F>
void for_each_nonzero(const mat::coordinate2D<Value, Parameters>& A, F&& f) {
    for (const auto& [r, c, v] : A.ref_entries())
        f(std::size_t(r), std::size_t(c), v);
}

/// ELLPACK: iterate the packed rows up to the invalid sentinel (O(nnz)).
template <typename Value, typename Parameters, typename F>
void for_each_nonzero(const mat::ell_matrix<Value, Parameters>& A, F&& f) {
    using ell = mat::ell_matrix<Value, Parameters>;
    const auto& indices = A.ref_indices();
    const auto& data    = A.ref_data();
    const std::size_t w = A.max_width();
    for (std::size_t i = 0; i < A.num_rows(); ++i)
        for (std::size_t k = 0; k < w; ++k) {
            const auto idx = indices[i * w + k];
            if (idx == ell::invalid) break;
            f(i, std::size_t(idx), data[i * w + k]);
        }
}

namespace detail {

/// Binned grid: for each of W*H pixel cells, the non-zero count and the largest
/// |value| that fell in it.
template <typename Mag>
struct spy_grid {
    std::size_t W = 0, H = 0;
    std::vector<std::size_t> counts;   ///< non-zeros per cell
    std::vector<Mag>         maxabs;   ///< max |value| per cell
};

template <typename M>
spy_grid<magnitude_t<typename M::value_type>>
build_spy_grid(const M& A, std::size_t max_pixels) {
    using value_type = typename M::value_type;
    using mag_t = magnitude_t<value_type>;
    using std::abs;
    const std::size_t m = A.num_rows(), n = A.num_cols();
    if (m == 0 || n == 0)
        throw std::runtime_error("spy: matrix has a zero dimension");
    if (max_pixels == 0)
        throw std::runtime_error("spy: max_pixels must be > 0");

    spy_grid<mag_t> g;
    g.W = std::min(n, max_pixels);
    g.H = std::min(m, max_pixels);
    g.counts.assign(g.W * g.H, 0);
    g.maxabs.assign(g.W * g.H, mag_t(0));

    for_each_nonzero(A, [&](std::size_t i, std::size_t j, const value_type& v) {
        std::size_t px = (j * g.W) / n;   // map column -> [0, W)
        std::size_t py = (i * g.H) / m;   // map row    -> [0, H)
        if (px >= g.W) px = g.W - 1;
        if (py >= g.H) py = g.H - 1;
        const std::size_t c = py * g.W + px;
        ++g.counts[c];
        const mag_t a = abs(v);
        if (a > g.maxabs[c]) g.maxabs[c] = a;
    });
    return g;
}

/// Compact viridis-like colormap: t in [0,1] -> (r,g,b), 8-bit.
inline void colormap(double t, std::uint8_t& r, std::uint8_t& gc, std::uint8_t& b) {
    static const double stop[5][3] = {
        { 68,   1,  84}, { 59,  82, 139}, { 33, 145, 140},
        { 94, 201,  98}, {253, 231,  37}
    };
    if (t < 0) t = 0; else if (t > 1) t = 1;
    const double x = t * 4.0;
    const int i = std::min(3, static_cast<int>(x));
    const double f = x - i;
    auto lerp = [&](int ch) { return stop[i][ch] + f * (stop[i + 1][ch] - stop[i][ch]); };
    r  = static_cast<std::uint8_t>(lerp(0) + 0.5);
    gc = static_cast<std::uint8_t>(lerp(1) + 0.5);
    b  = static_cast<std::uint8_t>(lerp(2) + 0.5);
}

} // namespace detail

/// Binary non-zero pattern: black mark on white where the matrix has a non-zero
/// (MATLAB `spy`). Grayscale PNG.
template <typename M>
void spy(const M& A, const std::filesystem::path& path, const spy_options& opt = {}) {
    const auto g = detail::build_spy_grid(A, opt.max_pixels);
    std::vector<std::uint8_t> px(g.W * g.H, 255);   // white background
    for (std::size_t c = 0; c < px.size(); ++c)
        if (g.counts[c] > 0) px[c] = 0;             // black mark
    write_png_gray(path, px.data(), g.W, g.H);
}

/// Non-zero pattern colored by per-cell max |a_ij| (analog of imagesc(abs(A))).
/// Linear by default, log10 with opt.log_scale. RGB PNG; empty cells are white.
template <typename M>
void spy_magnitude(const M& A, const std::filesystem::path& path, const spy_options& opt = {}) {
    using mag_t = magnitude_t<typename M::value_type>;
    const auto g = detail::build_spy_grid(A, opt.max_pixels);

    mag_t gmax = mag_t(0), gmin_pos = mag_t(0);
    bool have_min = false;
    for (std::size_t c = 0; c < g.counts.size(); ++c) {
        if (g.counts[c] == 0) continue;
        const mag_t a = g.maxabs[c];
        if (a > gmax) gmax = a;
        if (a > mag_t(0) && (!have_min || a < gmin_pos)) { gmin_pos = a; have_min = true; }
    }
    const double lgmin = have_min ? std::log10(static_cast<double>(gmin_pos)) : 0.0;
    const double lgmax = gmax > mag_t(0) ? std::log10(static_cast<double>(gmax)) : 0.0;

    std::vector<std::uint8_t> rgb(g.W * g.H * 3, 255);   // white background
    for (std::size_t c = 0; c < g.counts.size(); ++c) {
        if (g.counts[c] == 0) continue;
        double t = 0.0;
        const double a = static_cast<double>(g.maxabs[c]);
        if (opt.log_scale && have_min && lgmax > lgmin && a > 0.0)
            t = (std::log10(a) - lgmin) / (lgmax - lgmin);
        else if (gmax > mag_t(0))
            t = a / static_cast<double>(gmax);
        detail::colormap(t, rgb[c * 3], rgb[c * 3 + 1], rgb[c * 3 + 2]);
    }
    write_png_rgb(path, rgb.data(), g.W, g.H);
}

/// Non-zero pattern colored by per-cell non-zero *count* -- reveals clustering
/// once a large matrix is down-sampled. Linear by default, log10 with
/// opt.log_scale. RGB PNG; empty cells are white.
template <typename M>
void spy_density(const M& A, const std::filesystem::path& path, const spy_options& opt = {}) {
    const auto g = detail::build_spy_grid(A, opt.max_pixels);

    std::size_t cmax = 0;
    for (std::size_t v : g.counts) cmax = std::max(cmax, v);

    std::vector<std::uint8_t> rgb(g.W * g.H * 3, 255);   // white background
    for (std::size_t c = 0; c < g.counts.size(); ++c) {
        if (g.counts[c] == 0) continue;
        double t = 0.0;
        if (cmax > 0) {
            if (opt.log_scale && cmax > 1)
                t = std::log10(1.0 + static_cast<double>(g.counts[c])) /
                    std::log10(1.0 + static_cast<double>(cmax));
            else
                t = static_cast<double>(g.counts[c]) / static_cast<double>(cmax);
        }
        detail::colormap(t, rgb[c * 3], rgb[c * 3 + 1], rgb[c * 3 + 2]);
    }
    write_png_rgb(path, rgb.data(), g.W, g.H);
}

} // namespace mtl::io
