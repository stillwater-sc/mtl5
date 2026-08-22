// Integer GEMM: the blocked nest over integer element types.
//
// Two shapes of it, and the difference matters more than it looks:
//
//   same-type   mult(dense2D<int32_t>, ...)      -> gemm_blocked<int32_t>
//   widening    mult<int32_t>(dense2D<int8_t>, ...) -> gemm_blocked<int32_t, int8_t>
//
// The widening form is the interesting one. It reuses the micro-kernel's
// widening load, so an int8 GEMM reads an EIGHTH of the bytes an fp64 GEMM does
// while accumulating just as exactly. That is where the measured integer
// speedup lives (see docs/performance): on the dot kernels the operand width
// was worth ~18x and the specialised instruction ~1.2x.
//
// What this is NOT is a VNNI kernel. `vpdpbusd` consumes four k-values per
// instruction and needs a quad-interleaved pack layout and a different
// micro-kernel; this path promotes narrow operands into int32 lanes and does an
// ordinary multiply-add. That kernel exists as of #451 phase 5 and is tested in
// test_gemm_quad.cpp -- reached through `mult_quad`, never through `mult`, so
// the two stay separately measurable. The tests here therefore still pin the
// WIDENING kernel. That `mult` has not been quietly rerouted to the other one
// is checked in test_gemm_quad.cpp, and has to be checked structurally: the two
// kernels give bit-identical answers, so no result can tell them apart.
//
// Every case is EXACT. Integer arithmetic wraps mod 2^32 (#451), and wrapping
// addition is associative, so the blocked nest -- which reorders the k loop
// across pc blocks, splits m across threads, and accumulates in a register tile
// -- must produce bit-identical results to a naive triple loop. A float GEMM
// could only claim that within a rounding tolerance.
#define MTL5_NATIVE_FAST_GEMM 1

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include <mtl/interface/dispatch_traits.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/parameter.hpp>
#include <mtl/operation/mult.hpp>
#include <mtl/tag/orientation.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

using rowmaj = mtl::mat::parameters<mtl::tag::row_major>;
using colmaj = mtl::mat::parameters<mtl::tag::col_major>;

/// Exact reference, computed in 64 bits and reduced to the accumulator width.
template <typename TC, typename MA, typename MB>
TC ref_cell(const MA& A, const MB& B, std::size_t i, std::size_t j, std::size_t K) {
    std::uint64_t acc = 0;
    for (std::size_t k = 0; k < K; ++k)
        acc += static_cast<std::uint64_t>(static_cast<std::uint32_t>(static_cast<TC>(A(i, k)))) *
               static_cast<std::uint32_t>(static_cast<TC>(B(k, j)));
    return static_cast<TC>(static_cast<std::uint32_t>(acc));
}

// Shapes straddling the register tile (mr x nr), the cache blocks, and the
// thread partition, plus ragged ones that leave partial panels everywhere.
const std::size_t kShapes[][3] = {
    {1, 1, 1}, {1, 8, 1}, {8, 1, 1}, {4, 4, 4}, {7, 5, 3}, {13, 11, 7},
    {16, 16, 16}, {17, 16, 16}, {16, 17, 16}, {16, 16, 17}, {33, 17, 9},
    {64, 64, 64}, {65, 63, 31}, {128, 96, 48}, {96, 128, 64},
};

} // namespace

TEST_CASE("integer matrices reach the blocked GEMM, not the triple loop",
          "[operation][gemm][integer]") {
    using i32m = mtl::mat::dense2D<std::int32_t, rowmaj>;
    using i8m  = mtl::mat::dense2D<std::int8_t,  rowmaj>;
    using dm   = mtl::mat::dense2D<double, rowmaj>;

    // Same-type int32 now satisfies the gate the blocked nest is behind. It
    // satisfies no BLAS predicate -- there is no external ?gemm for int32 -- so
    // the native path is the only accelerated one it can take.
    STATIC_REQUIRE(mtl::interface::SimdDenseMatrix<i32m>);
    STATIC_REQUIRE_FALSE(mtl::interface::BlasDenseMatrix<i32m>);

    // 8-bit operands are not lanes at all, so they are described by the layout
    // predicate only; the element type is constrained separately at the call.
    STATIC_REQUIRE_FALSE(mtl::interface::SimdDenseMatrix<i8m>);
    STATIC_REQUIRE(mtl::interface::ContiguousMatrixData<i8m>);

    // Float/double are untouched: still both.
    STATIC_REQUIRE(mtl::interface::SimdDenseMatrix<dm>);
    STATIC_REQUIRE(mtl::interface::BlasDenseMatrix<dm>);
}

TEMPLATE_TEST_CASE("same-type integer GEMM is exact at every shape",
                   "[operation][gemm][integer]", std::int32_t, std::uint32_t) {
    using T = TestType;
    for (auto& s : kShapes) {
        const std::size_t M = s[0], N = s[1], K = s[2];
        mtl::mat::dense2D<T, rowmaj> A(M, K), B(K, N), C(M, N);
        for (std::size_t i = 0; i < M; ++i)
            for (std::size_t k = 0; k < K; ++k)
                A(i, k) = static_cast<T>(static_cast<std::uint32_t>(0x9E3779B9u * (i * K + k + 1)));
        for (std::size_t k = 0; k < K; ++k)
            for (std::size_t j = 0; j < N; ++j)
                B(k, j) = static_cast<T>(static_cast<std::uint32_t>(0x85EBCA77u * (k * N + j + 3)));

        mtl::mult(A, B, C);

        bool all = true;
        for (std::size_t i = 0; i < M && all; ++i)
            for (std::size_t j = 0; j < N && all; ++j)
                all = (C(i, j) == (ref_cell<T>(A, B, i, j, K)));
        INFO("M=" << M << " N=" << N << " K=" << K);
        CHECK(all);
    }
}

TEMPLATE_TEST_CASE("widening integer GEMM is exact at every shape",
                   "[operation][gemm][integer]", std::int8_t, std::int16_t) {
    // 8- and 16-bit operands accumulated in int32 through the widening load.
    using TAB = TestType;
    for (auto& s : kShapes) {
        const std::size_t M = s[0], N = s[1], K = s[2];
        mtl::mat::dense2D<TAB, rowmaj> A(M, K), B(K, N);
        mtl::mat::dense2D<std::int32_t, rowmaj> C(M, N);
        for (std::size_t i = 0; i < M; ++i)
            for (std::size_t k = 0; k < K; ++k) A(i, k) = static_cast<TAB>((i * 7 + k * 3) % 251 - 125);
        for (std::size_t k = 0; k < K; ++k)
            for (std::size_t j = 0; j < N; ++j) B(k, j) = static_cast<TAB>((k * 5 + j * 2) % 241 - 120);

        mtl::mult<std::int32_t>(A, B, C);

        bool all = true;
        for (std::size_t i = 0; i < M && all; ++i)
            for (std::size_t j = 0; j < N && all; ++j)
                all = (C(i, j) == (ref_cell<std::int32_t>(A, B, i, j, K)));
        INFO("M=" << M << " N=" << N << " K=" << K << "  operand width=" << sizeof(TAB));
        CHECK(all);
    }
}

TEST_CASE("the blocked nest agrees with the triple loop bit for bit",
          "[operation][gemm][integer]") {
    // The claim a float GEMM cannot make. The nest reorders the k loop across
    // pc blocks, partitions m across threads and accumulates in a register tile;
    // on integers all of that is addition mod 2^32, which is associative, so the
    // answer must be IDENTICAL to the naive order -- not merely close.
    constexpr std::size_t M = 67, N = 53, K = 41;
    mtl::mat::dense2D<std::int32_t, rowmaj> A(M, K), B(K, N), C(M, N), Cref(M, N);
    for (std::size_t i = 0; i < M; ++i)
        for (std::size_t k = 0; k < K; ++k)
            A(i, k) = static_cast<std::int32_t>(static_cast<std::uint32_t>(0x9E3779B9u * (i * K + k + 1)));
    for (std::size_t k = 0; k < K; ++k)
        for (std::size_t j = 0; j < N; ++j)
            B(k, j) = static_cast<std::int32_t>(static_cast<std::uint32_t>(0x85EBCA77u * (k * N + j + 3)));

    mtl::mult(A, B, C);
    mtl::detail::mult_generic(A, B, Cref);      // the naive triple loop

    bool all = true;
    for (std::size_t i = 0; i < M && all; ++i)
        for (std::size_t j = 0; j < N && all; ++j) all = (C(i, j) == Cref(i, j));
    CHECK(all);
}

TEST_CASE("column-major C takes the transposed path and still agrees",
          "[operation][gemm][integer]") {
    // Col-major C computes C^T = B^T A^T into the same buffer by swapping each
    // operand's strides, which is a different traversal of the same arithmetic.
    constexpr std::size_t M = 33, N = 21, K = 17;
    mtl::mat::dense2D<std::int32_t, rowmaj> A(M, K), B(K, N);
    mtl::mat::dense2D<std::int32_t, colmaj> C(M, N);
    for (std::size_t i = 0; i < M; ++i)
        for (std::size_t k = 0; k < K; ++k) A(i, k) = static_cast<std::int32_t>((i * 13 + k * 5) % 97 - 48);
    for (std::size_t k = 0; k < K; ++k)
        for (std::size_t j = 0; j < N; ++j) B(k, j) = static_cast<std::int32_t>((k * 11 + j * 7) % 89 - 44);

    mtl::mult(A, B, C);

    bool all = true;
    for (std::size_t i = 0; i < M && all; ++i)
        for (std::size_t j = 0; j < N && all; ++j)
            all = (C(i, j) == (ref_cell<std::int32_t>(A, B, i, j, K)));
    CHECK(all);
}

TEST_CASE("integer GEMM wraps rather than overflowing", "[operation][gemm][integer]") {
    // Full-range operands over a long k: the products alone exceed int32 and the
    // sum exceeds it many times over. The answer is the true product-sum reduced
    // mod 2^32 -- defined, not undefined, and not saturated.
    constexpr std::size_t M = 9, N = 9, K = 257;
    mtl::mat::dense2D<std::int32_t, rowmaj> A(M, K), B(K, N), C(M, N);
    for (std::size_t i = 0; i < M; ++i)
        for (std::size_t k = 0; k < K; ++k)
            A(i, k) = static_cast<std::int32_t>(static_cast<std::uint32_t>(0xC2B2AE35u * (i * K + k + 1)));
    for (std::size_t k = 0; k < K; ++k)
        for (std::size_t j = 0; j < N; ++j)
            B(k, j) = static_cast<std::int32_t>(static_cast<std::uint32_t>(0x27D4EB2Fu * (k * N + j + 7)));

    mtl::mult(A, B, C);

    bool all = true, any_negative = false;
    for (std::size_t i = 0; i < M; ++i)
        for (std::size_t j = 0; j < N; ++j) {
            const auto want = ref_cell<std::int32_t>(A, B, i, j, K);
            all = all && (C(i, j) == want);
            any_negative = any_negative || (C(i, j) < 0);
        }
    CHECK(all);
    CHECK(any_negative);      // i.e. it really did wrap, so the check has teeth
}

TEST_CASE("a mixed-signedness operand pair is not silently widened",
          "[operation][gemm][integer]") {
    // batch::load_widen requires matching signedness, and the GEMM dispatch
    // repeats that requirement rather than guessing. A uint8 x uint8 pair into a
    // SIGNED int32 accumulator therefore does not take the widening kernel; it
    // falls to the accumulator-aware generic loop, which is correct.
    using u8m = mtl::mat::dense2D<std::uint8_t, rowmaj>;
    constexpr std::size_t M = 5, N = 4, K = 6;
    u8m A(M, K), B(K, N);
    mtl::mat::dense2D<std::int32_t, rowmaj> C(M, N);
    for (std::size_t i = 0; i < M; ++i)
        for (std::size_t k = 0; k < K; ++k) A(i, k) = static_cast<std::uint8_t>((i * 7 + k) % 200 + 5);
    for (std::size_t k = 0; k < K; ++k)
        for (std::size_t j = 0; j < N; ++j) B(k, j) = static_cast<std::uint8_t>((k * 3 + j) % 200 + 5);

    mtl::mult<std::int32_t>(A, B, C);          // must still compile, and be right

    bool all = true;
    for (std::size_t i = 0; i < M && all; ++i)
        for (std::size_t j = 0; j < N && all; ++j) {
            std::int32_t want = 0;
            for (std::size_t k = 0; k < K; ++k) want += std::int32_t(A(i, k)) * std::int32_t(B(k, j));
            all = (C(i, j) == want);
        }
    CHECK(all);
}
