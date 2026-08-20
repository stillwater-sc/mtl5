#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <mtl/interface/dispatch_traits.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/dot.hpp>
#include <mtl/operation/mult.hpp>
#include <mtl/mat/dense2D.hpp>
#include <complex>
#include <cstddef>
#include <cstdint>

using namespace mtl;
using Catch::Matchers::WithinAbs;

TEST_CASE("dot product of double vectors", "[operation][dot]") {
    dense_vector<double> a = {1.0, 2.0, 3.0};
    dense_vector<double> b = {4.0, 5.0, 6.0};
    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    REQUIRE(dot(a, b) == 32.0);
}

TEST_CASE("dot product of int vectors", "[operation][dot]") {
    dense_vector<int> a = {1, 2, 3};
    dense_vector<int> b = {2, 3, 4};
    // 1*2 + 2*3 + 3*4 = 2 + 6 + 12 = 20
    REQUIRE(dot(a, b) == 20);
}

TEST_CASE("dot product of orthogonal vectors is zero", "[operation][dot]") {
    dense_vector<double> a = {1.0, 0.0, 0.0};
    dense_vector<double> b = {0.0, 1.0, 0.0};
    REQUIRE(dot(a, b) == 0.0);
}

TEST_CASE("dot product with uniform vectors", "[operation][dot]") {
    dense_vector<double> a(3, 1.0);
    dense_vector<double> b(3, 2.0);
    REQUIRE(dot(a, b) == 6.0);
}

TEST_CASE("dot product of complex vectors is Hermitian", "[operation][dot]") {
    using cd = std::complex<double>;
    dense_vector<cd> a = {cd(1.0, 1.0), cd(2.0, -1.0)};
    dense_vector<cd> b = {cd(3.0, 0.0), cd(0.0, 1.0)};
    // conj(a[0])*b[0] + conj(a[1])*b[1]
    // = (1-i)*(3+0i) + (2+i)*(0+i)
    // = (3-3i) + (0+2i+0-1) = (3-3i) + (-1+2i) = (2-i)
    auto result = dot(a, b);
    REQUIRE_THAT(result.real(), WithinAbs(2.0, 1e-10));
    REQUIRE_THAT(result.imag(), WithinAbs(-1.0, 1e-10));
}

TEST_CASE("dot_real skips conjugation", "[operation][dot]") {
    using cd = std::complex<double>;
    dense_vector<cd> a = {cd(1.0, 1.0), cd(2.0, -1.0)};
    dense_vector<cd> b = {cd(3.0, 0.0), cd(0.0, 1.0)};
    // a[0]*b[0] + a[1]*b[1]
    // = (1+i)*(3+0i) + (2-i)*(0+i)
    // = (3+3i) + (0+2i-0+1) = (3+3i) + (1+2i) = (4+5i)
    auto result = dot_real(a, b);
    REQUIRE_THAT(result.real(), WithinAbs(4.0, 1e-10));
    REQUIRE_THAT(result.imag(), WithinAbs(5.0, 1e-10));
}

TEST_CASE("dot product of single-element vectors", "[operation][dot]") {
    dense_vector<double> a = {5.0};
    dense_vector<double> b = {3.0};
    REQUIRE(dot(a, b) == 15.0);
}

// Integer dot after #451 phase 0: same values, different route. dense_vector<int>
// used to fall to the generic accumulate loop because the SIMD gate asked for a
// BLAS scalar type; it now goes through simd::reduce_dot. The results below are
// small enough to be order-independent either way -- the point of stating them
// is that widening the gate did not change any answer.
TEST_CASE("integer dot takes the SIMD path and keeps its values", "[operation][dot][integer]") {
    STATIC_REQUIRE(mtl::interface::SimdDenseVector<dense_vector<std::int32_t>>);
    STATIC_REQUIRE_FALSE(mtl::interface::BlasDenseVector<dense_vector<std::int32_t>>);

    dense_vector<std::int32_t> a = {1, 2, 3, 4, 5};
    dense_vector<std::int32_t> b = {2, 3, 4, 5, 6};
    // 2 + 6 + 12 + 20 + 30 = 70
    CHECK(dot(a, b) == 70);
    CHECK(dot_real(a, b) == 70);
}

// Long enough to run the four-accumulator SIMD body, the single-batch loop and
// the scalar tail, with operands large enough that the sum overflows int32 many
// times over. The contract is that the answer is the exact sum reduced mod 2^32,
// and that it does not depend on where the kernel happened to split the work.
TEST_CASE("integer dot overflows into a defined value, not an order-dependent one",
          "[operation][dot][integer]") {
    constexpr std::size_t n = 1031;                       // prime, so tails are ragged
    dense_vector<std::int32_t> a(n), b(n);
    std::uint64_t acc = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const auto ai = static_cast<std::uint32_t>(0x9E3779B9u * (i + 1));
        const auto bi = static_cast<std::uint32_t>(0x85EBCA77u * (i + 3));
        a(i) = static_cast<std::int32_t>(ai);
        b(i) = static_cast<std::int32_t>(bi);
        acc += static_cast<std::uint64_t>(ai) * bi;       // uint64 wrap keeps low 32 bits
    }
    const auto expected = static_cast<std::int32_t>(static_cast<std::uint32_t>(acc));
    CHECK(dot(a, b) == expected);
    CHECK(dot_real(a, b) == expected);
}

// The GENERIC dense mult path, which is what an integer matrix takes in the
// DEFAULT build: MTL5_NATIVE_FAST_GEMM is off, so mult() falls to
// detail::mult_generic. Its inner loop used to be `acc += A(r,c) * x(c)`, which
// on int32 is signed-overflow UB and disagrees with the mod-2^32 answer the SIMD
// kernel is documented to give. Both paths must land on the same value.
TEST_CASE("generic integer mat*vec wraps rather than overflowing",
          "[operation][mult][integer][generic]") {
    constexpr std::size_t n = 37;
    mtl::mat::dense2D<std::int32_t> A(n, n);
    dense_vector<std::int32_t> x(n), y(n);
    for (std::size_t i = 0; i < n; ++i) {
        x(i) = static_cast<std::int32_t>(static_cast<std::uint32_t>(0x85EBCA77u * (i + 3)));
        for (std::size_t j = 0; j < n; ++j)
            A(i, j) = static_cast<std::int32_t>(
                static_cast<std::uint32_t>(0x9E3779B9u * (i * n + j + 1)));
    }
    mtl::mult(A, x, y);                       // generic path (no MTL5_NATIVE_FAST_GEMM here)

    for (std::size_t i = 0; i < n; ++i) {
        std::uint64_t acc = 0;
        for (std::size_t j = 0; j < n; ++j)
            acc += static_cast<std::uint64_t>(static_cast<std::uint32_t>(A(i, j)))
                 * static_cast<std::uint32_t>(x(j));
        INFO("row " << i);
        REQUIRE(y(i) == static_cast<std::int32_t>(static_cast<std::uint32_t>(acc)));
    }
}
