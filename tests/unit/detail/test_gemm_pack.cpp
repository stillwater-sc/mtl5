// Tests for GEMM packing routines (#89): pack_A (MR-row col-major panels) and
// pack_B (NR-col row-major panels), with generic strides and edge padding.
//
// We verify the packed buffer layout DIRECTLY against the micro-kernel contract
//   Ap[panel*MR*k + p*MR + i] == A(panel*MR+i, p)   (0 when row is padded)
//   Bp[panel*NR*k + p*NR + j] == B(p, panel*NR+j)   (0 when col is padded)
// rather than through the micro-kernel, so the test is independent of #88.
// Integer-valued, distinct entries make every (i,j) mismatch (stride/transpose
// bug) detectable by exact comparison.
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include <mtl/detail/gemm_pack.hpp>

#include <cstddef>
#include <vector>

namespace {

// Distinct nonzero value for logical entry (i,j): never collides with the 0 pad.
template <typename T>
T entry(std::size_t i, std::size_t j) { return T(1 + i * 31 + j); }

// Pack an m x k logical A in BOTH row-major and col-major storage and check the
// packed buffer slot-by-slot (real entries and zero padding).
template <typename T, std::size_t MR>
void check_pack_A(std::size_t m, std::size_t k) {
    std::vector<T> rowmaj(m * k), colmaj(m * k);
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < k; ++j) {
            rowmaj[i * k + j] = entry<T>(i, j);  // rs=k, cs=1
            colmaj[j * m + i] = entry<T>(i, j);  // rs=1, cs=m
        }

    const std::size_t panels = (m + MR - 1) / MR;
    const std::size_t sz = mtl::detail::packed_A_size(m, k, MR);
    CHECK(sz == panels * MR * k);

    const T sentinel = T(-99999);
    for (int orient = 0; orient < 2; ++orient) {
        std::vector<T> Ap(sz, sentinel);
        if (orient == 0)
            mtl::detail::pack_A<T, MR>(rowmaj.data(), static_cast<std::ptrdiff_t>(k), 1, m, k, Ap.data());
        else
            mtl::detail::pack_A<T, MR>(colmaj.data(), 1, static_cast<std::ptrdiff_t>(m), m, k, Ap.data());

        for (std::size_t q = 0; q < panels; ++q)
            for (std::size_t p = 0; p < k; ++p)
                for (std::size_t i = 0; i < MR; ++i) {
                    const std::size_t row = q * MR + i;
                    const T expected = (row < m) ? entry<T>(row, p) : T(0);
                    CHECK(Ap[q * MR * k + p * MR + i] == expected);
                }
    }
}

// Pack a k x n logical B in BOTH row-major and col-major storage and check.
template <typename T, std::size_t NR>
void check_pack_B(std::size_t k, std::size_t n) {
    std::vector<T> rowmaj(k * n), colmaj(k * n);
    for (std::size_t p = 0; p < k; ++p)
        for (std::size_t j = 0; j < n; ++j) {
            rowmaj[p * n + j] = entry<T>(p, j);  // rs=n, cs=1
            colmaj[j * k + p] = entry<T>(p, j);  // rs=1, cs=k
        }

    const std::size_t panels = (n + NR - 1) / NR;
    const std::size_t sz = mtl::detail::packed_B_size(k, n, NR);
    CHECK(sz == panels * NR * k);

    const T sentinel = T(-99999);
    for (int orient = 0; orient < 2; ++orient) {
        std::vector<T> Bp(sz, sentinel);
        if (orient == 0)
            mtl::detail::pack_B<T, NR>(rowmaj.data(), static_cast<std::ptrdiff_t>(n), 1, k, n, Bp.data());
        else
            mtl::detail::pack_B<T, NR>(colmaj.data(), 1, static_cast<std::ptrdiff_t>(k), k, n, Bp.data());

        for (std::size_t q = 0; q < panels; ++q)
            for (std::size_t p = 0; p < k; ++p)
                for (std::size_t j = 0; j < NR; ++j) {
                    const std::size_t col = q * NR + j;
                    const T expected = (col < n) ? entry<T>(p, col) : T(0);
                    CHECK(Bp[q * NR * k + p * NR + j] == expected);
                }
    }
}

const std::size_t kDims[] = {1, 2, 3, 4, 5, 7, 8, 9, 16};

} // namespace

TEMPLATE_TEST_CASE("pack_A: MR-row col-major panels, both orientations + padding", "[detail][gemm][pack]", float, double) {
    for (std::size_t m : kDims)
        for (std::size_t k : kDims) {
            check_pack_A<TestType, 1>(m, k);
            check_pack_A<TestType, 3>(m, k);  // odd MR -> exercises padding for most m
            check_pack_A<TestType, 4>(m, k);  // AVX2 fp64 MR
        }
}

TEMPLATE_TEST_CASE("pack_B: NR-col row-major panels, both orientations + padding", "[detail][gemm][pack]", float, double) {
    for (std::size_t k : kDims)
        for (std::size_t n : kDims) {
            check_pack_B<TestType, 1>(k, n);
            check_pack_B<TestType, 3>(k, n);  // odd NR
            check_pack_B<TestType, 8>(k, n);  // AVX2 fp64 NR
        }
}

// Spot-check the exact contract values for a known 3x2 A with MR=2 (one full
// panel + one padded panel), so a layout regression is caught concretely.
TEST_CASE("pack_A: explicit 3x2, MR=2 layout", "[detail][gemm][pack]") {
    // A = [[10,11],[20,21],[30,31]] row-major, rs=2, cs=1
    const double A[] = {10, 11, 20, 21, 30, 31};
    std::vector<double> Ap(mtl::detail::packed_A_size(3, 2, 2));  // 2 panels * 2 * 2 = 8
    mtl::detail::pack_A<double, 2>(A, 2, 1, 3, 2, Ap.data());
    // panel0 (rows 0..1), col-major: p0->{10,20}, p1->{11,21}
    // panel1 (row 2 + pad),         : p0->{30, 0}, p1->{31, 0}
    const double expected[] = {10, 20, 11, 21, 30, 0, 31, 0};
    for (std::size_t t = 0; t < 8; ++t) CHECK(Ap[t] == expected[t]);
}
