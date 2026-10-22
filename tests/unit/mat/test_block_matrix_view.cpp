// block_matrix_view and the OCP MX schemes -- #629.
//
// The schemes are checked against the specification's tables, code by code,
// rather than against another implementation: E8M0 at its extremes and NaN,
// every FP4 E2M1 code in both nibble positions of the GGML block_mxfp4 layout,
// and the INT8 element's implicit 2^-6. The view is then checked as a Matrix:
// construction rejects malformed shapes, element access, row decode and
// convert agree, and mult / operator* match a dense2D reference. The
// generic headers (mult.hpp, convert.hpp, operators.hpp) are included on
// purpose, so the block overloads are chosen over them, not merely compiled.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <mtl/concepts/block_scheme.hpp>
#include <mtl/concepts/matrix.hpp>
#include <mtl/mat/block_matrix_view.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/mx_schemes.hpp>
#include <mtl/mat/operators.hpp>
#include <mtl/operation/convert.hpp>
#include <mtl/operation/mult.hpp>
#include <mtl/vec/dense_vector.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

using mtl::mat::block_matrix_view;

/// A scheme with no decode_element: element access must decode the block.
/// Four int8 values per block, scaled by the block's leading byte.
struct int8x4_scaled {
    static constexpr std::size_t block_size  = 4;
    static constexpr std::size_t block_bytes = 1 + 4;
    static void decode(const std::byte* src, float* dst) {
        const float s = float(static_cast<std::uint8_t>(src[0]));
        for (std::size_t j = 0; j < 4; ++j) {
            std::int8_t q;
            std::memcpy(&q, src + 1 + j, 1);
            dst[j] = s * float(q);
        }
    }
};

/// The same format, with a fused kernel that records that it ran.
struct int8x4_fused : int8x4_scaled {};
int fused_calls = 0;

/// The same format, with a fused kernel that needs scratch: it quantizes x
/// into the scratch first, as a real integer kernel would, and records what it
/// was given.
struct int8x4_scratch : int8x4_scaled {};
const std::byte* scratch_seen = nullptr;
std::size_t scratch_size_seen = 0;

struct not_a_scheme {
    static constexpr std::size_t block_size = 4;   // no block_bytes, no decode
};

std::vector<std::byte> random_mxint8(std::size_t rows, std::size_t cols, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> scale(120, 134), q(-128, 127);
    std::vector<std::byte> v(rows * (cols / 32) * mtl::mx::mxint8::block_bytes);
    for (std::size_t b = 0; b < v.size(); b += mtl::mx::mxint8::block_bytes) {
        v[b] = std::byte(scale(rng));
        for (std::size_t j = 1; j <= 32; ++j) v[b + j] = std::byte(std::uint8_t(q(rng)));
    }
    return v;
}

std::vector<std::byte> random_mxfp4(std::size_t rows, std::size_t cols, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> scale(120, 134), byte(0, 255);
    std::vector<std::byte> v(rows * (cols / 32) * mtl::mx::mxfp4::block_bytes);
    for (std::size_t b = 0; b < v.size(); b += mtl::mx::mxfp4::block_bytes) {
        v[b] = std::byte(scale(rng));
        for (std::size_t j = 1; j <= 16; ++j) v[b + j] = std::byte(byte(rng));
    }
    return v;
}

} // namespace

template <>
struct mtl::block_kernels<int8x4_fused> {
    static void gemv(const mtl::mat::block_matrix_view<int8x4_fused>& A, const float* x, float* y,
                     std::span<std::byte>) {
        ++fused_calls;
        std::vector<float> row(A.num_cols());
        for (std::size_t r = 0; r < A.num_rows(); ++r) {
            A.decode_row(r, row.data());
            float acc = 0;
            for (std::size_t c = 0; c < A.num_cols(); ++c) acc += row[c] * x[c];
            y[r] = acc;
        }
    }
};

template <>
struct mtl::block_kernels<int8x4_scratch> {
    static constexpr std::size_t scratch_alignment = 64;
    static std::size_t scratch_bytes(std::size_t cols) { return cols * sizeof(float); }
    static void gemv(const mtl::mat::block_matrix_view<int8x4_scratch>& A, const float* x, float* y,
                     std::span<std::byte> scratch) {
        scratch_seen = scratch.data();
        scratch_size_seen = scratch.size();
        REQUIRE(scratch.size() >= scratch_bytes(A.num_cols()));
        REQUIRE(reinterpret_cast<std::uintptr_t>(scratch.data()) % scratch_alignment == 0);
        auto* xs = reinterpret_cast<float*>(scratch.data());   // the kernel's own copy of x
        for (std::size_t c = 0; c < A.num_cols(); ++c) xs[c] = x[c];
        std::vector<float> row(A.num_cols());
        for (std::size_t r = 0; r < A.num_rows(); ++r) {
            A.decode_row(r, row.data());
            float acc = 0;
            for (std::size_t c = 0; c < A.num_cols(); ++c) acc += row[c] * xs[c];
            y[r] = acc;
        }
    }
};

// ---- the concept --------------------------------------------------------------

static_assert(mtl::BlockScheme<mtl::mx::mxint8>);
static_assert(mtl::BlockScheme<mtl::mx::mxfp4>);
static_assert(mtl::BlockScheme<int8x4_scaled>);
static_assert(!mtl::BlockScheme<not_a_scheme>);
static_assert(mtl::ElementDecodableScheme<mtl::mx::mxint8>);
static_assert(!mtl::ElementDecodableScheme<int8x4_scaled>);
static_assert(mtl::BlockGemvScheme<int8x4_fused>);
static_assert(!mtl::BlockGemvScheme<int8x4_scaled>);
static_assert(mtl::BlockGemvScheme<int8x4_scratch>);
static_assert(mtl::block_gemv_scratch_alignment<int8x4_scratch> == 64);
static_assert(mtl::block_gemv_scratch_alignment<int8x4_fused> == alignof(std::max_align_t));
static_assert(mtl::block_gemv_scratch_alignment<int8x4_scaled> == alignof(std::max_align_t));

static_assert(mtl::Matrix<block_matrix_view<mtl::mx::mxint8>>);
static_assert(mtl::Matrix<block_matrix_view<mtl::mx::mxfp4>>);
static_assert(mtl::DenseMatrix<block_matrix_view<mtl::mx::mxint8>>);
static_assert(!mtl::Vector<block_matrix_view<mtl::mx::mxint8>>);
// No data() of float type, so no BLAS/SIMD kernel can mistake it for storage.
static_assert(!mtl::interface::ContiguousMatrixData<block_matrix_view<mtl::mx::mxint8>>);

// ---- the MX encodings -----------------------------------------------------------

TEST_CASE("E8M0 is 2^(e-127) and 0xFF is NaN", "[block][mx]") {
    CHECK(mtl::mx::e8m0_to_float(127) == 1.0f);
    CHECK(mtl::mx::e8m0_to_float(128) == 2.0f);
    CHECK(mtl::mx::e8m0_to_float(126) == 0.5f);
    CHECK(mtl::mx::e8m0_to_float(0) == std::ldexp(1.0f, -127));   // a float subnormal
    CHECK(mtl::mx::e8m0_to_float(0) > 0.0f);
    CHECK(mtl::mx::e8m0_to_float(254) == std::ldexp(1.0f, 127));
    CHECK(std::isnan(mtl::mx::e8m0_to_float(255)));
}

TEST_CASE("every FP4 E2M1 code decodes per the specification's table", "[block][mx]") {
    const float expected[16] = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
                                -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
    for (std::uint8_t c = 0; c < 16; ++c) {
        INFO("code " << int(c));
        CHECK(mtl::mx::e2m1_to_float(c) == expected[c]);
        CHECK(std::signbit(mtl::mx::e2m1_to_float(c)) == (c >= 8));
    }
}

TEST_CASE("mxfp4 uses GGML's block_mxfp4 nibble order", "[block][mx]") {
    // Byte 1 + j holds element j in its low nibble and element j + 16 in its
    // high nibble. Fill every code into both positions, with distinct codes per
    // position, so a swapped order or a sequential-pair order would fail.
    std::byte blk[mtl::mx::mxfp4::block_bytes];
    blk[0] = std::byte(128);   // X = 2
    for (std::size_t j = 0; j < 16; ++j)
        blk[1 + j] = std::byte(std::uint8_t(j | ((15 - j) << 4)));
    float out[32];
    mtl::mx::mxfp4::decode(blk, out);
    for (std::size_t j = 0; j < 16; ++j) {
        INFO("j " << j);
        CHECK(out[j] == 2.0f * mtl::mx::e2m1_to_float(std::uint8_t(j)));
        CHECK(out[j + 16] == 2.0f * mtl::mx::e2m1_to_float(std::uint8_t(15 - j)));
        CHECK(mtl::mx::mxfp4::decode_element(blk, j) == out[j]);
        CHECK(mtl::mx::mxfp4::decode_element(blk, j + 16) == out[j + 16]);
    }
}

TEST_CASE("mxfp4 at the scale extremes", "[block][mx]") {
    std::byte blk[mtl::mx::mxfp4::block_bytes] = {};
    blk[1] = std::byte(0x31);   // element 0: code 1 (0.5), element 16: code 3 (1.5)
    blk[2] = std::byte(0x04);   // element 1: code 4 (2.0)
    blk[0] = std::byte(0);      // X = 2^-127
    CHECK(mtl::mx::mxfp4::decode_element(blk, 0) == std::ldexp(1.0f, -128));
    blk[0] = std::byte(254);    // X = 2^127
    CHECK(mtl::mx::mxfp4::decode_element(blk, 16) == std::ldexp(1.5f, 127));   // finite
    CHECK(std::isinf(mtl::mx::mxfp4::decode_element(blk, 1)));   // 2 * 2^127 > FLT_MAX
    blk[0] = std::byte(255);
    for (std::size_t j = 0; j < 32; ++j) CHECK(std::isnan(mtl::mx::mxfp4::decode_element(blk, j)));
}

TEST_CASE("mxint8 elements are q / 64 times the scale", "[block][mx]") {
    std::byte blk[mtl::mx::mxint8::block_bytes];
    const std::int8_t q[] = {0, 1, -1, 64, -64, 127, -128, 33};
    for (std::size_t j = 0; j < 32; ++j) blk[1 + j] = std::byte(std::uint8_t(q[j % 8]));
    for (int e : {127, 130, 0, 254}) {
        blk[0] = std::byte(e);
        float out[32];
        mtl::mx::mxint8::decode(blk, out);
        for (std::size_t j = 0; j < 32; ++j) {
            INFO("e " << e << ", q " << int(q[j % 8]));
            CHECK(out[j] == std::ldexp(float(q[j % 8]) / 64.0f, e - 127));
            // Only q = -128 (exactly -2) under the largest scale leaves float's
            // range: -2 * 2^127 = -2^128.
            CHECK(std::isfinite(out[j]) == !(e == 254 && q[j % 8] == -128));
        }
    }
    blk[0] = std::byte(255);
    for (std::size_t j = 0; j < 32; ++j) CHECK(std::isnan(mtl::mx::mxint8::decode_element(blk, j)));
}

// ---- the view -------------------------------------------------------------------

TEST_CASE("block_matrix_view rejects a malformed shape", "[block][view]") {
    std::vector<std::byte> bytes(2 * 2 * 33);
    CHECK_NOTHROW(block_matrix_view<mtl::mx::mxint8>(2, 64, bytes));
    CHECK_THROWS_AS(block_matrix_view<mtl::mx::mxint8>(2, 48, bytes), std::invalid_argument);
    CHECK_THROWS_AS(block_matrix_view<mtl::mx::mxint8>(3, 64, bytes), std::invalid_argument);
    CHECK_THROWS_AS(block_matrix_view<mtl::mx::mxint8>(2, 64, std::span(bytes).first(131)),
                    std::invalid_argument);
    CHECK_NOTHROW(block_matrix_view<mtl::mx::mxint8>(0, 64, std::span<const std::byte>{}));
}

TEST_CASE("block_matrix_view rejects a shape whose size wraps", "[block][view]") {
    using Catch::Matchers::ContainsSubstring;
    constexpr std::size_t max = std::numeric_limits<std::size_t>::max();

    // The review's case: rows * 33 wraps to a handful of bytes. Unchecked, a
    // span of exactly that many bytes validated, and row(0) read far past it.
    // The assertion is on the overflow message, so it is the new check that
    // fires, not the size comparison.
    const std::size_t rows = max / 33 + 1;
    const std::size_t wrapped = rows * 33;   // unsigned: wraps, by definition
    REQUIRE(wrapped < 64);
    std::vector<std::byte> small(wrapped);
    CHECK_THROWS_WITH((block_matrix_view<mtl::mx::mxint8>(rows, 32, small)),
                      ContainsSubstring("overflows"));

    // Bytes per row alone wraps: 33-byte blocks for 32 elements.
    std::vector<std::byte> one_block(33);
    CHECK_THROWS_WITH((block_matrix_view<mtl::mx::mxint8>(1, (max / 32) * 32, one_block)),
                      ContainsSubstring("overflows"));

    // The byte count fits but rows * cols does not: 17-byte blocks hold 32
    // elements, so N blocks need 17 N bytes but describe 32 N elements.
    std::vector<std::byte> fp4_block(17);
    CHECK_THROWS_WITH((block_matrix_view<mtl::mx::mxfp4>(max / 17 / 2, 64, fp4_block)),
                      ContainsSubstring("overflows"));
}

TEST_CASE("element access, row decode, block bytes and convert agree", "[block][view]") {
    const std::size_t rows = 5, cols = 96;
    const auto bytes = random_mxfp4(rows, cols, 1);
    const block_matrix_view<mtl::mx::mxfp4> A(rows, cols, bytes);
    CHECK(A.num_rows() == rows);
    CHECK(A.num_cols() == cols);
    CHECK(A.size() == rows * cols);
    CHECK(A.blocks_per_row() == 3);
    CHECK(A.row(2).data() == bytes.data() + 2 * 3 * 17);
    CHECK(A.row(2).size() == 3 * 17);
    CHECK(A.block(4, 1).data() == bytes.data() + (4 * 3 + 1) * 17);

    const auto D = mtl::convert<double>(A);   // the block overload: decodes each row once
    mtl::mat::dense2D<float> F(rows, cols);
    mtl::convert(A, F);
    std::vector<float> row(cols);
    for (std::size_t r = 0; r < rows; ++r) {
        A.decode_row(r, row.data());
        for (std::size_t c = 0; c < cols; ++c) {
            float blk[32];
            mtl::mx::mxfp4::decode(A.block(r, c / 32).data(), blk);
            CHECK(A(r, c) == row[c]);
            CHECK(blk[c % 32] == row[c]);
            CHECK(F(r, c) == row[c]);
            CHECK(D(r, c) == double(row[c]));
        }
    }
}

TEST_CASE("a scheme without decode_element still gives element access", "[block][view]") {
    std::vector<std::byte> bytes = {std::byte(3), std::byte(1), std::byte(2), std::byte(0xFF),
                                    std::byte(4)};   // scale 3, values 1, 2, -1, 4
    const block_matrix_view<int8x4_scaled> A(1, 4, bytes);
    CHECK(A(0, 0) == 3.0f);
    CHECK(A(0, 1) == 6.0f);
    CHECK(A(0, 2) == -3.0f);
    CHECK(A(0, 3) == 12.0f);
}

TEST_CASE("mult on a block matrix matches the decoded dense matrix", "[block][mult]") {
    const std::size_t rows = 37, cols = 128;
    const auto bytes = random_mxint8(rows, cols, 7);
    const block_matrix_view<mtl::mx::mxint8> A(rows, cols, bytes);
    const auto D = mtl::convert<double>(A);
    std::mt19937 rng(9);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    mtl::vec::dense_vector<float> x(cols);
    mtl::vec::dense_vector<double> xd(cols);
    for (std::size_t c = 0; c < cols; ++c) xd(c) = x(c) = d(rng);

    // Reference in double from the decoded values: each decoded element and
    // each x is a float, so their products are exact in double.
    std::vector<double> ref(rows), mag(rows);
    for (std::size_t r = 0; r < rows; ++r)
        for (std::size_t c = 0; c < cols; ++c) {
            ref[r] += D(r, c) * xd(c);
            mag[r] += std::fabs(D(r, c) * xd(c));
        }
    const double tol = cols * std::ldexp(1.0, -23);   // float accumulation over a row

    SECTION("float x: the SIMD dot") {
        mtl::vec::dense_vector<float> y(rows);
        mtl::mult(A, x, y);
        for (std::size_t r = 0; r < rows; ++r) CHECK(std::fabs(y(r) - ref[r]) <= tol * mag[r]);
        const auto z = A * x;   // operator* goes through mult
        for (std::size_t r = 0; r < rows; ++r) CHECK(z(r) == y(r));
    }
    SECTION("double x and y: the generic dot, in double") {
        mtl::vec::dense_vector<double> y(rows);
        mtl::mult(A, xd, y);
        for (std::size_t r = 0; r < rows; ++r)
            CHECK(std::fabs(y(r) - ref[r]) <= cols * std::ldexp(1.0, -52) * mag[r]);
    }
    SECTION("an explicit accumulator") {
        mtl::vec::dense_vector<float> y(rows);
        mtl::mult<double>(A, x, y);
        // Products and sum in double, rounded once to float.
        for (std::size_t r = 0; r < rows; ++r)
            CHECK(std::fabs(double(y(r)) - ref[r]) <=
                  std::ldexp(1.0, -24) * std::fabs(ref[r]) + cols * std::ldexp(1.0, -52) * mag[r]);
    }
}

TEST_CASE("mult calls a scheme's fused kernel only where it applies", "[block][mult]") {
    // Two rows of two blocks: scale 1 and scale 2.
    std::vector<std::byte> bytes;
    for (int r = 0; r < 2; ++r)
        for (int b = 0; b < 2; ++b) {
            bytes.push_back(std::byte(1 + b));
            for (int j = 0; j < 4; ++j) bytes.push_back(std::byte(std::uint8_t(r + j)));
        }
    const block_matrix_view<int8x4_fused> A(2, 8, bytes);
    const block_matrix_view<int8x4_scaled> B(2, 8, bytes);
    mtl::vec::dense_vector<float> x(8, 1.0f), y(2), yb(2);
    mtl::mult(B, x, yb);

    fused_calls = 0;
    mtl::mult(A, x, y);
    CHECK(fused_calls == 1);
    CHECK(y(0) == yb(0));
    CHECK(y(1) == yb(1));

    mtl::vec::dense_vector<double> xd(8, 1.0), yd(2);
    mtl::mult(A, xd, yd);   // double vectors: not the float kernel's signature
    mtl::mult<double>(A, x, y);   // an explicit accumulator: the caller asked for it
    CHECK(fused_calls == 1);
    CHECK(yd(0) == double(yb(0)));
}

TEST_CASE("a fused kernel's scratch comes from the caller, or from mult for one call",
          "[block][mult][scratch]") {
    std::vector<std::byte> bytes;
    for (int r = 0; r < 3; ++r)
        for (int b = 0; b < 4; ++b) {
            bytes.push_back(std::byte(1 + b));
            for (int j = 0; j < 4; ++j) bytes.push_back(std::byte(std::uint8_t(r * 7 + j - 5)));
        }
    const block_matrix_view<int8x4_scratch> A(3, 16, bytes);
    const block_matrix_view<int8x4_scaled> B(3, 16, bytes);
    mtl::vec::dense_vector<float> x(16), y(3), yref(3);
    for (std::size_t c = 0; c < 16; ++c) x(c) = 0.25f * float(c) - 1.0f;
    mtl::mult(B, x, yref);

    CHECK(mtl::block_gemv_scratch_bytes<int8x4_scratch>(16) == 64);
    CHECK(mtl::block_gemv_scratch_bytes<int8x4_fused>(16) == 0);    // a kernel that declares none
    CHECK(mtl::block_gemv_scratch_bytes<int8x4_scaled>(16) == 0);   // no kernel

    SECTION("caller-owned scratch is passed through as given") {
        alignas(64) std::array<std::byte, 128> buf{};
        scratch_seen = nullptr;
        mtl::mult(A, x, y, std::span<std::byte>(buf));
        CHECK(scratch_seen == buf.data());
        CHECK(scratch_size_seen == buf.size());
        for (std::size_t r = 0; r < 3; ++r) CHECK(y(r) == yref(r));
    }
    SECTION("without scratch, mult allocates an aligned block of the declared size") {
        scratch_seen = nullptr;
        mtl::mult(A, x, y);
        CHECK(scratch_seen != nullptr);
        CHECK(scratch_size_seen == 64);
        for (std::size_t r = 0; r < 3; ++r) CHECK(y(r) == yref(r));
        const auto z = A * x;
        for (std::size_t r = 0; r < 3; ++r) CHECK(z(r) == yref(r));
    }
    SECTION("too small or misaligned scratch is rejected before the kernel runs") {
        alignas(64) std::array<std::byte, 128> buf{};
        scratch_seen = nullptr;
        CHECK_THROWS_WITH(mtl::mult(A, x, y, std::span<std::byte>(buf.data(), 63)),
                          Catch::Matchers::ContainsSubstring("needs 64"));
        CHECK_THROWS_WITH(mtl::mult(A, x, y, std::span<std::byte>(buf.data() + 1, 64)),
                          Catch::Matchers::ContainsSubstring("not aligned to 64"));
        CHECK(scratch_seen == nullptr);
    }
    SECTION("off the fused path the scratch is unused, so an empty one is fine") {
        mtl::vec::dense_vector<double> xd(16), yd(3);
        for (std::size_t c = 0; c < 16; ++c) xd(c) = x(c);
        scratch_seen = nullptr;
        mtl::mult(A, xd, yd, std::span<std::byte>{});           // double vectors
        mtl::mult<double>(A, x, y, std::span<std::byte>{});     // explicit accumulator
        mtl::mult(B, x, y, std::span<std::byte>{});             // no kernel
        CHECK(scratch_seen == nullptr);
        for (std::size_t r = 0; r < 3; ++r) CHECK(yd(r) == double(yref(r)));
    }
}
