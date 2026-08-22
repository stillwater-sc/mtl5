// Quad-interleaved GEMM packing -- #451 phase 5.
//
// The ordinary pack stores one k value at a time because the pairwise
// micro-kernel does one rank-1 update per k. The quad multiply-accumulate
// consumes FOUR k values per instruction, and the four belonging to one
// (row, lane) have to sit in four ADJACENT bytes -- so this is a different
// layout, not a different traversal.
//
// The layout is the kernel's contract, and it is checked here ELEMENT BY
// ELEMENT rather than through a GEMM result, because a pack bug and a kernel bug
// produce the same symptom (a wrong C) and only one of them is visible from
// here. Three properties, and each has cost someone a day somewhere:
//
//   1. the index formula, for every element of every panel
//   2. zero padding of BOTH edges -- the ragged row/column edge, which the
//      pairwise pack also has, and the k TAIL, which is new: k is not a multiple
//      of four in general, and the kernel has no size check
//   3. panel independence -- a cooperative split across a thread team must
//      produce byte-identical output, which is what keeps the threaded quad GEMM
//      bit-identical to the serial one
#include <catch2/catch_test_macros.hpp>

#include <mtl/detail/gemm_quad_pack.hpp>

#include <algorithm>   // std::min, in the cooperative-split case
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

constexpr std::size_t MR = 4, NR = 6;

/// A(i,j) = a distinct nonzero value, so a misplaced element cannot coincide
/// with the zero padding or with another cell.
std::int8_t cell(std::size_t i, std::size_t j) {
    return static_cast<std::int8_t>(1 + (i * 37 + j * 11) % 100);
}

} // namespace

TEST_CASE("quad_depth rounds k up to whole 4-value groups", "[detail][gemm][quad][pack]") {
    using mtl::detail::quad_depth;
    CHECK(quad_depth(0) == 0);
    CHECK(quad_depth(1) == 4);
    CHECK(quad_depth(3) == 4);
    CHECK(quad_depth(4) == 4);
    CHECK(quad_depth(5) == 8);
    CHECK(quad_depth(256) == 256);
    CHECK(quad_depth(257) == 260);
}

TEST_CASE("packed quad sizes count padded panels at padded depth", "[detail][gemm][quad][pack]") {
    using mtl::detail::packed_A_quad_size;
    using mtl::detail::packed_B_quad_size;
    // 7 rows in MR=4 panels -> 2 panels of 4; k=5 -> depth 8.
    CHECK(packed_A_quad_size(7, 5, 4) == 2 * 4 * 8);
    CHECK(packed_B_quad_size(5, 7, 6) == 2 * 6 * 8);
    CHECK(packed_A_quad_size(0, 5, 4) == 0);
    CHECK(packed_B_quad_size(5, 0, 6) == 0);
}

TEST_CASE("pack_A_quad lays out Ap[panel][group][row][q] == A(row, 4g+q)",
          "[detail][gemm][quad][pack]") {
    // m and k both ragged: 7 rows into MR=4 panels leaves one row of padding,
    // k=6 into groups of 4 leaves two k values of padding.
    constexpr std::size_t m = 7, k = 6;
    const std::size_t KP = mtl::detail::quad_depth(k);

    std::vector<std::int8_t> A(m * k);
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < k; ++j) A[i * k + j] = cell(i, j);

    std::vector<std::int8_t> Ap(mtl::detail::packed_A_quad_size(m, k, MR), -1);
    mtl::detail::pack_A_quad<std::int8_t, MR>(
        A.data(), static_cast<std::ptrdiff_t>(k), 1, m, k, Ap.data());

    const std::size_t npanels = (m + MR - 1) / MR;
    bool all = true;
    for (std::size_t p = 0; p < npanels; ++p)
        for (std::size_t g = 0; g < KP / 4; ++g)
            for (std::size_t i = 0; i < MR; ++i)
                for (std::size_t q = 0; q < 4; ++q) {
                    const std::size_t row = p * MR + i, kk = 4 * g + q;
                    const std::int8_t want = (row < m && kk < k) ? cell(row, kk) : std::int8_t(0);
                    const std::int8_t got = Ap[p * MR * KP + g * MR * 4 + i * 4 + q];
                    INFO("panel " << p << " group " << g << " row " << i << " q " << q);
                    if (got != want) all = false;
                    CHECK(got == want);
                }
    CHECK(all);
}

TEST_CASE("pack_B_quad lays out Bp[panel][group][col][q] == B(4g+q, col)",
          "[detail][gemm][quad][pack]") {
    constexpr std::size_t k = 6, n = 7;
    const std::size_t KP = mtl::detail::quad_depth(k);

    std::vector<std::int8_t> B(k * n);
    for (std::size_t i = 0; i < k; ++i)
        for (std::size_t j = 0; j < n; ++j) B[i * n + j] = cell(i, j);

    std::vector<std::int8_t> Bp(mtl::detail::packed_B_quad_size(k, n, NR), -1);
    mtl::detail::pack_B_quad<std::int8_t, NR>(
        B.data(), static_cast<std::ptrdiff_t>(n), 1, k, n, Bp.data());

    const std::size_t npanels = (n + NR - 1) / NR;
    bool all = true;
    for (std::size_t p = 0; p < npanels; ++p)
        for (std::size_t g = 0; g < KP / 4; ++g)
            for (std::size_t j = 0; j < NR; ++j)
                for (std::size_t q = 0; q < 4; ++q) {
                    const std::size_t col = p * NR + j, kk = 4 * g + q;
                    const std::int8_t want = (col < n && kk < k) ? cell(kk, col) : std::int8_t(0);
                    const std::int8_t got = Bp[p * NR * KP + g * NR * 4 + j * 4 + q];
                    INFO("panel " << p << " group " << g << " col " << j << " q " << q);
                    if (got != want) all = false;
                    CHECK(got == want);
                }
    CHECK(all);
}

TEST_CASE("column-major sources pack identically via their strides",
          "[detail][gemm][quad][pack]") {
    // Generic (rs, cs) is what lets any orientation -- and a transposed view --
    // pack with no special-casing. Same logical matrix, two storage orders, one
    // packed result.
    constexpr std::size_t m = 5, k = 7;
    std::vector<std::int8_t> row(m * k), col(m * k);
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < k; ++j) {
            row[i * k + j] = cell(i, j);
            col[i + j * m] = cell(i, j);
        }

    const std::size_t sz = mtl::detail::packed_A_quad_size(m, k, MR);
    std::vector<std::int8_t> pr(sz, -1), pc(sz, -2);
    mtl::detail::pack_A_quad<std::int8_t, MR>(row.data(), static_cast<std::ptrdiff_t>(k), 1,
                                              m, k, pr.data());
    mtl::detail::pack_A_quad<std::int8_t, MR>(col.data(), 1, static_cast<std::ptrdiff_t>(m),
                                              m, k, pc.data());
    CHECK(pr == pc);
}

TEST_CASE("a cooperative panel split is byte-identical to packing whole",
          "[detail][gemm][quad][pack]") {
    // The threaded nest hands each team member a disjoint range of NR-column
    // panels. Packing is pure data movement, so the bytes -- and therefore the
    // GEMM result -- must not depend on the split. This is the property the
    // bit-identity claim rests on.
    constexpr std::size_t k = 13, n = 19;
    std::vector<std::int8_t> B(k * n);
    for (std::size_t i = 0; i < k; ++i)
        for (std::size_t j = 0; j < n; ++j) B[i * n + j] = cell(i, j);

    const std::size_t sz = mtl::detail::packed_B_quad_size(k, n, NR);
    const std::size_t npanels = (n + NR - 1) / NR;
    std::vector<std::int8_t> whole(sz, -1);
    mtl::detail::pack_B_quad<std::int8_t, NR>(B.data(), static_cast<std::ptrdiff_t>(n), 1,
                                              k, n, whole.data());

    for (std::size_t parts = 1; parts <= npanels + 2; ++parts) {
        std::vector<std::int8_t> split(sz, -1);
        const std::size_t per = (npanels + parts - 1) / parts;
        for (std::size_t t = 0; t < parts; ++t) {
            const std::size_t q0 = std::min(npanels, t * per);
            const std::size_t q1 = std::min(npanels, q0 + per);
            if (q0 < q1)
                mtl::detail::pack_B_quad_panels<std::int8_t, NR>(
                    B.data(), static_cast<std::ptrdiff_t>(n), 1, k, n, split.data(), q0, q1);
        }
        INFO("split into " << parts << " ranges");
        CHECK(split == whole);
    }
}

TEST_CASE("an out-of-range panel range is clamped, not read", "[detail][gemm][quad][pack]") {
    // The nest computes q1 from a ceiling division, so the last member's range
    // can run past the panel count. pack_B_quad_panels clamps; if it did not,
    // it would write past the buffer.
    constexpr std::size_t k = 5, n = 6;
    std::vector<std::int8_t> B(k * n, 3);
    std::vector<std::int8_t> Bp(mtl::detail::packed_B_quad_size(k, n, NR), -1);
    mtl::detail::pack_B_quad_panels<std::int8_t, NR>(
        B.data(), static_cast<std::ptrdiff_t>(n), 1, k, n, Bp.data(), 0, 99);
    // One panel of 6 columns at depth 8; every stored k value is 3, the tail 0.
    const std::size_t KP = mtl::detail::quad_depth(k);
    bool all = true;
    for (std::size_t g = 0; g < KP / 4; ++g)
        for (std::size_t j = 0; j < NR; ++j)
            for (std::size_t q = 0; q < 4; ++q) {
                const std::int8_t want = (4 * g + q < k) ? std::int8_t(3) : std::int8_t(0);
                if (Bp[g * NR * 4 + j * 4 + q] != want) all = false;
            }
    CHECK(all);
}
