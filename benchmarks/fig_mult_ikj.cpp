// fig_mult_ikj -- figures for docs/performance/ikj-gemm-and-crs-dense.md
//
// Enlarged non-zero structure images of the filter's left operands, and the
// memory-access pattern of the i-j-k and i-k-j inner loops. Written with
// mtl::io::write_png_rgb: `spy` draws one pixel per element, which leaves an
// 81 x 81 matrix 81 pixels wide, so these draw each element as a square cell.
//
//   g++ -std=c++20 -O2 -I<mtl5>/include fig_mult_ikj.cpp && ./a.out <out-dir>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <vector>

#include <mtl/io/png.hpp>

namespace {

using rgb = std::array<std::uint8_t, 3>;

constexpr rgb kWhite{255, 255, 255};
constexpr rgb kGrid{232, 232, 236};
constexpr rgb kBlock{150, 150, 160};     // block boundaries
constexpr rgb kIdentity{40, 40, 48};     // the identity's diagonal
constexpr rgb kDense{31, 119, 180};      // a dense block's entries
constexpr rgb kCoupled{214, 96, 39};     // columns K H fills in
constexpr rgb kLight{205, 225, 240};     // memory not touched by the loop

constexpr std::size_t kState = 81, kImu = 15, kMeas = 15, kClone = 6;

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

/// Draw an m x n matrix whose cell (i, j) has colour color(i, j), `cell` pixels
/// per element, with a light grid and darker lines on the block boundaries.
canvas draw(std::size_t m, std::size_t n, std::size_t cell,
            const std::function<rgb(std::size_t, std::size_t)>& color,
            const std::vector<std::size_t>& row_blocks,
            const std::vector<std::size_t>& col_blocks) {
    const std::size_t pad = 2;
    canvas cv(n * cell + 1 + 2 * pad, m * cell + 1 + 2 * pad);
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j)
            cv.rect(pad + j * cell + 1, pad + i * cell + 1, cell - 1, cell - 1, color(i, j));
    for (std::size_t i = 0; i <= m; ++i)
        cv.rect(pad, pad + i * cell, n * cell + 1, 1, kGrid);
    for (std::size_t j = 0; j <= n; ++j)
        cv.rect(pad + j * cell, pad, 1, m * cell + 1, kGrid);
    for (std::size_t b : row_blocks)
        cv.rect(pad, pad + b * cell, n * cell + 1, 1, kBlock);
    for (std::size_t b : col_blocks)
        cv.rect(pad + b * cell, pad, 1, m * cell + 1, kBlock);
    return cv;
}

std::vector<std::size_t> state_blocks() {
    std::vector<std::size_t> b{0, kImu};
    for (std::size_t s = kImu + kClone; s <= kState; s += kClone) b.push_back(s);
    return b;
}

bool observed(std::size_t j, std::size_t clones) { return j < kImu + kClone * clones; }

} // namespace

int main(int argc, char** argv) {
    const std::filesystem::path out = argc > 1 ? argv[1] : ".";
    std::filesystem::create_directories(out);
    const std::size_t cell = 6;
    const auto blocks = state_blocks();
    const std::size_t clones = 3;

    // F = Phi (+) I
    draw(kState, kState, cell,
         [](std::size_t i, std::size_t j) {
             if (i < kImu && j < kImu) return kDense;
             return i == j ? kIdentity : kWhite;
         },
         blocks, blocks)
        .save(out / "ikj-structure-transition.png");

    // H: the IMU block and the observing clones
    draw(kMeas, kState, cell,
         [&](std::size_t, std::size_t j) { return observed(j, clones) ? kDense : kWhite; },
         {0, kMeas}, blocks)
        .save(out / "ikj-structure-jacobian.png");

    // I - K H: identity, plus K H's dense columns where H is non-zero
    draw(kState, kState, cell,
         [&](std::size_t i, std::size_t j) {
             if (observed(j, clones)) return kCoupled;
             return i == j ? kIdentity : kWhite;
         },
         blocks, blocks)
        .save(out / "ikj-structure-joseph.png");

    // Access pattern of one inner loop over a 12 x 12 row-major B, with the same
    // 144 elements laid out as memory below it (12 per line, in address order).
    const std::size_t n = 12, c2 = 36, col = 4, row = 4;
    auto access = [&](bool by_row, const std::filesystem::path& p) {
        auto hit = [&](std::size_t i, std::size_t j) { return by_row ? i == row : j == col; };
        canvas grid = draw(n, n, c2,
                           [&](std::size_t i, std::size_t j) { return hit(i, j) ? kDense : kLight; },
                           {}, {});
        // memory strip: 144 cells in one line, scaled to the grid's width
        const std::size_t sc = 3, gap = 16;
        const std::size_t strip_w = n * n * sc;
        canvas cv(std::max(grid.w, strip_w + 4), grid.h + gap + 3 * sc * 4 + 4);
        for (std::size_t y = 0; y < grid.h; ++y)
            for (std::size_t x = 0; x < grid.w; ++x) {
                const auto* q = &grid.px[(y * grid.w + x) * 3];
                cv.set(x, y, {q[0], q[1], q[2]});
            }
        const std::size_t y0 = grid.h + gap;
        for (std::size_t a = 0; a < n * n; ++a) {
            const std::size_t i = a / n, j = a % n;
            cv.rect(2 + a * sc, y0, sc, 3 * sc * 4, hit(i, j) ? kDense : kLight);
        }
        for (std::size_t i = 0; i <= n; ++i)   // row boundaries in memory
            cv.rect(2 + i * n * sc, y0, 1, 3 * sc * 4, kBlock);
        cv.save(p);
    };
    access(false, out / "ikj-access-ijk.png");
    access(true, out / "ikj-access-ikj.png");
    return 0;
}
