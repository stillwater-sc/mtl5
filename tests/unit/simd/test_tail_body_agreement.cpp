// The scalar tail must round as the vector body does (#512).
//
// Every mtl::simd kernel that accumulates has two loops: a vector body over
// whole lane groups and a scalar tail over the last `n mod W` elements. Those
// used to disagree -- the body called `fma()`, the tail a plain multiply-add --
// so an element's rounding depended on its INDEX. For a reduction that perturbs
// one number; for `axpy`, where each element is an independent output, it means
// the first W*floor(n/W) entries of the result vector were computed differently
// from the last few.
//
// WHY THIS IS NOT "ASSERT std::fma IN THE TAIL". Under Highway, `fma` is
// `hn::MulAdd`, which is a hardware FMA only when the target has one: AVX2 gives
// `_mm256_fmadd_ps`, SSE4 gives `mul * x + add`. An unconditionally fused tail
// against a decomposed body is the same defect mirrored. Measured before the fix:
// on a 128-bit Highway target, every element of an axpy came out unfused -- body
// included -- while the scalar-fallback build fused all of them.
//
// So the property under test is AGREEMENT, not fusion. These tests compare
// element i < W against element j >= W of the same call, and never against an
// oracle of their own, because the correct answer is whatever the body computes
// and that is a property of the build.
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <mtl/simd/batch.hpp>
#include <mtl/simd/algorithm.hpp>

using namespace mtl;

namespace {

/// A value whose fused and unfused multiply-add differ in float, so that a
/// body/tail disagreement is visible rather than absorbed by rounding.
constexpr float kThird = 1.0f / 3.0f;

} // namespace

TEST_CASE("simd::axpy rounds the tail as it rounds the body", "[simd][fma][tail]") {
    constexpr std::size_t W = simd::batch<float>::size;
    if (W < 2) {
        // With a lane count of 1 the body covers everything and the tail never
        // runs -- there is nothing to disagree. Say so rather than passing mutely.
        WARN("batch<float>::size == 1: no scalar tail exists on this build");
    }

    // Every element has identical inputs, so ANY difference between two elements
    // of the result is a difference in how they were computed.
    for (std::size_t extra = 1; extra <= 3; ++extra) {
        const std::size_t n = W + extra;              // W in the body, `extra` in the tail
        std::vector<float> x(n, kThird), y(n, kThird);
        simd::axpy<float>(kThird, x.data(), y.data(), n);

        for (std::size_t i = 1; i < n; ++i)
            REQUIRE(y[i] == y[0]);                    // body and tail agree
    }
}

TEST_CASE("simd::axpy on double rounds the tail as it rounds the body",
          "[simd][fma][tail]") {
    constexpr std::size_t W = simd::batch<double>::size;
    for (std::size_t extra = 1; extra <= 3; ++extra) {
        const std::size_t n = W + extra;
        const double t = 1.0 / 3.0;
        std::vector<double> x(n, t), y(n, t);
        simd::axpy<double>(t, x.data(), y.data(), n);
        for (std::size_t i = 1; i < n; ++i)
            REQUIRE(y[i] == y[0]);
    }
}

TEST_CASE("scalar_fma mirrors the batch fma, lane for lane", "[simd][fma][tail]") {
    // THE INVARIANT THE FIX RESTS ON, tested directly rather than through a
    // kernel. `scalar_fma` exists so a scalar tail computes what one lane of the
    // vector body computes; if the two ever diverge, every kernel's tail is wrong
    // again and no downstream test would say which.
    //
    // Broadcasting the same value to every lane makes the batch result uniform,
    // so reduce_max recovers the lane value exactly.
    {
        const float a = kThird, b = kThird, c = kThird;
        const auto v = fma(simd::batch<float>(a), simd::batch<float>(b),
                           simd::batch<float>(c));
        REQUIRE(simd::scalar_fma(a, b, c) == reduce_max(v));
    }
    {
        const double a = 1.0 / 3.0, b = 1.0 / 7.0, c = 1.0 / 11.0;
        const auto v = fma(simd::batch<double>(a), simd::batch<double>(b),
                           simd::batch<double>(c));
        REQUIRE(simd::scalar_fma(a, b, c) == reduce_max(v));
    }
    {
        // A pair whose fused and unfused results differ, so the check has teeth
        // on a target where MulAdd is native: 1/3 * 1/3 + 1/3 is the case the
        // #512 investigation used.
        const float a = kThird, b = kThird, c = kThird;
        const float fused = std::fma(a, b, c);
        // `volatile` is the contraction barrier, and it is load-bearing. Written
        // as `c + static_cast<float>(a * b)` the compiler is free to contract the
        // whole expression into an FMA -- a same-type cast is not a barrier, and
        // GCC defaults to -ffp-contract=fast -- which would make `unfused` equal
        // `fused` and silently void the premise below. Storing the product forces
        // the intermediate rounding. Raised in review of #513.
        volatile float prod = a * b;
        const float unfused = c + prod;
        REQUIRE(fused != unfused);                       // premise of the check
        const float got = simd::scalar_fma(a, b, c);
        REQUIRE((got == fused || got == unfused));       // one or the other...
        const auto v = fma(simd::batch<float>(a), simd::batch<float>(b),
                           simd::batch<float>(c));
        REQUIRE(got == reduce_max(v));                   // ... and the SAME one
    }
}

// NOT TESTED HERE: that a reduction's public result isolates its tail. The first
// version of this file asserted `reduce_dot(n) == reduce_dot(nb) + folded tail`
// and it failed -- correctly. `reduce_dot` carries four unrolled accumulators and
// combines them as `(a0+a1)+(a2+a3)`, so changing n changes the summation
// GROUPING, and floating-point addition is not associative. That invariant was
// testing associativity, not tail fusion. The lane-wise check above is the
// property the fix actually provides; the axpy cases are its visible consequence.

TEST_CASE("integer kernels keep the exact wrapping answer", "[simd][fma][tail][integer]") {
    // scalar_fma must NOT reach for std::fma on integer lanes: the contract there
    // is exact mod 2^N (#451/#460), which the wrapping multiply-add already gives
    // and a floating-point fma would not. Full-range values so a naive change
    // would be caught by the value, not only by the type.
    using i32 = std::int32_t;
    constexpr i32 imax = 2147483647;
    REQUIRE(simd::scalar_fma<i32>(imax, 2, 0) == static_cast<i32>(
                static_cast<std::uint32_t>(imax) * 2u));
    REQUIRE(simd::scalar_fma<i32>(imax, 1, 1) == static_cast<i32>(
                static_cast<std::uint32_t>(imax) + 1u));

    const std::size_t W = simd::batch<i32>::size;
    const std::size_t n = W + 3;
    std::vector<i32> a(n, imax), b(n, 3);
    const i32 got = simd::reduce_dot<i32>(a.data(), b.data(), n);

    std::uint32_t want = 0;
    for (std::size_t i = 0; i < n; ++i)
        want += static_cast<std::uint32_t>(imax) * 3u;
    REQUIRE(got == static_cast<i32>(want));
}
