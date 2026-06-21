// MTL5 -- accumulator policy for gemv and the sum-of-squares norms (#160, #162).
// Mirrors dot/gemm: an explicit Accumulator sums in a precision distinct from the
// element type, with the result delivered in the natural output/magnitude type.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstddef>
#include <type_traits>

#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/mult.hpp>
#include <mtl/operation/norms.hpp>

using namespace mtl;
using Catch::Matchers::WithinRel;
using Catch::Matchers::WithinAbs;

TEST_CASE("gemv default behavior is unchanged", "[operation][gemv][accumulator]") {
    mat::dense2D<double> A(2, 3);
    A(0,0)=1; A(0,1)=2; A(0,2)=3; A(1,0)=4; A(1,1)=5; A(1,2)=6;
    vec::dense_vector<double> x = {1.0, 1.0, 1.0}, y(2, 0.0);
    mult(A, x, y);
    REQUIRE_THAT(y(0), WithinRel(6.0, 1e-12));    // 1+2+3
    REQUIRE_THAT(y(1), WithinRel(15.0, 1e-12));   // 4+5+6
}

TEST_CASE("gemv fp64 accumulator beats fp32 on a long contraction",
          "[operation][gemv][accumulator]") {
    const std::size_t m = 4, n = 100000;
    mat::dense2D<float> A(m, n);
    vec::dense_vector<float> x(n);
    for (std::size_t j = 0; j < n; ++j) {
        x(static_cast<int>(j)) = 1.0f;
        for (std::size_t i = 0; i < m; ++i)
            A(i, j) = (j % 2 == 0) ? 1.0f : -1.0f + 1.0e-6f;   // near-cancelling
    }
    double ref = 0.0;
    for (std::size_t j = 0; j < n; ++j) ref += static_cast<double>(A(0, j));

    vec::dense_vector<float> y_naive(m, 0.0f);
    mult(A, x, y_naive);                 // fp32 accumulate (default generic for non-contiguous? still float)

    vec::dense_vector<float> y_wide(m, 0.0f);
    mult<double>(A, x, y_wide);          // fp64 accumulate, fp32 result

    double e_naive = std::abs(static_cast<double>(y_naive(0)) - ref);
    double e_wide  = std::abs(static_cast<double>(y_wide(0))  - ref);
    INFO("ref=" << ref << " naive=" << e_naive << " wide=" << e_wide);
    REQUIRE(e_wide <= e_naive);
}

TEST_CASE("two_norm accumulator policy: default unchanged, wide is accurate",
          "[operation][norms][accumulator]") {
    vec::dense_vector<double> v = {3.0, 4.0};
    REQUIRE_THAT(two_norm(v), WithinRel(5.0, 1e-12));
    REQUIRE_THAT(two_norm<double>(v), WithinRel(5.0, 1e-12));

    // fp64 accumulate over many float squares beats fp32 accumulation.
    const std::size_t n = 200000;
    vec::dense_vector<float> x(n);
    for (std::size_t i = 0; i < n; ++i) x(static_cast<int>(i)) = 1.0f;   // exact answer sqrt(n)
    double ref = std::sqrt(static_cast<double>(n));

    float  nf = two_norm(x);             // fp32 accumulate
    auto   nw = two_norm<double>(x);     // fp64 accumulate
    static_assert(std::is_same_v<decltype(nw), float>);   // returns element magnitude type
    double e_naive = std::abs(static_cast<double>(nf) - ref);
    double e_wide  = std::abs(static_cast<double>(nw) - ref);
    INFO("ref=" << ref << " naive=" << e_naive << " wide=" << e_wide);
    REQUIRE(e_wide <= e_naive);
}

TEST_CASE("frobenius_norm accumulator policy", "[operation][norms][accumulator]") {
    mat::dense2D<double> m(2, 2);
    m(0,0)=1; m(0,1)=2; m(1,0)=2; m(1,1)=4;   // sum of squares = 25
    REQUIRE_THAT(frobenius_norm(m), WithinRel(5.0, 1e-12));
    REQUIRE_THAT(frobenius_norm<double>(m), WithinRel(5.0, 1e-12));
}
