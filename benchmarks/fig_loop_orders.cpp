// fig_loop_orders -- figures for docs/architecture/accuracy-architecture.md (#596)
//
// Which part of an n x n matrix an LU factorization reads and writes at one
// step, in the right-looking and the left-looking form. Written with mtl::io::write_png_rgb,
// one square cell per element; the page's captions give the colours.
//
//   g++ -std=c++20 -O2 -I<mtl5>/include fig_loop_orders.cpp && ./a.out <out-dir>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <vector>

#include <mtl/io/png.hpp>

namespace {

using rgb = std::array<std::uint8_t, 3>;

constexpr rgb kGrid{232, 232, 236};
constexpr rgb kUntouched{255, 255, 255};   // still A, not yet touched
constexpr rgb kDone{200, 200, 208};        // finished factor, not used at this step
constexpr rgb kRead{31, 119, 180};         // read at this step
constexpr rgb kWritten{214, 96, 39};       // written (rounded into storage) at this step

struct canvas {
    std::size_t w, h;
    std::vector<std::uint8_t> px;
    canvas(std::size_t w_, std::size_t h_) : w(w_), h(h_), px(w_ * h_ * 3, 255) {}
    void set(std::size_t x, std::size_t y, rgb c) {
        if (x >= w || y >= h) return;
        auto* p = &px[(y * w + x) * 3];
        p[0] = c[0]; p[1] = c[1]; p[2] = c[2];
    }
    void rect(std::size_t x0, std::size_t y0, std::size_t ww, std::size_t hh, rgb c) {
        for (std::size_t y = y0; y < y0 + hh; ++y)
            for (std::size_t x = x0; x < x0 + ww; ++x) set(x, y, c);
    }
    void save(const std::filesystem::path& p) const { mtl::io::write_png_rgb(p, px.data(), w, h); }
};

/// Panels side by side: each an n x n matrix whose cell (i, j) has colour
/// color(panel, i, j), `cell` pixels per element.
canvas draw(std::size_t panels, std::size_t n, std::size_t cell,
            const std::function<rgb(std::size_t, std::size_t, std::size_t)>& color) {
    const std::size_t pad = 2, gap = 3 * cell, side = n * cell + 1;
    canvas cv(panels * side + (panels - 1) * gap + 2 * pad, side + 2 * pad);
    for (std::size_t p = 0; p < panels; ++p) {
        const std::size_t x0 = pad + p * (side + gap);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                cv.rect(x0 + j * cell + 1, pad + i * cell + 1, cell - 1, cell - 1, color(p, i, j));
        for (std::size_t i = 0; i <= n; ++i) cv.rect(x0, pad + i * cell, side, 1, kGrid);
        for (std::size_t j = 0; j <= n; ++j) cv.rect(x0 + j * cell, pad, 1, side, kGrid);
    }
    return cv;
}

} // namespace

int main(int argc, char** argv) {
    const std::filesystem::path out = argc > 1 ? argv[1] : ".";
    std::filesystem::create_directories(out);
    const std::size_t n = 12, cell = 22, k = 4;

    // Right-looking LU, step k: L(k+1:n, k) = A(k+1:n, k) / U(k, k), then the
    // trailing update A(i, j) -= L(i, k) U(k, j) for i, j > k. Every trailing
    // entry is written, and rounded, at every step.
    draw(1, n, cell,
         [&](std::size_t, std::size_t i, std::size_t j) {
             if (i < k || j < k) return kDone;
             if (i == k || j == k) return kRead;   // pivot row U(k, k:n), column L(k+1:n, k)
             return kWritten;                       // trailing submatrix
         })
        .save(out / "accuracy-right-looking.png");

    // Left-looking LU, column j = k: U(0:k, k) and L(k+1:n, k) are each one
    // reduction over the finished columns 0..k-1 of L. Only column k is written,
    // once per entry.
    draw(1, n, cell,
         [&](std::size_t, std::size_t, std::size_t j) {
             if (j == k) return kWritten;
             if (j < k) return kRead;          // the finished columns of L (and U above)
             return kUntouched;                // columns not reached yet
         })
        .save(out / "accuracy-left-looking.png");

    return 0;
}
