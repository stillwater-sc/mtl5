// MTL5 -- mixed-precision accumulator policy for GEMM (issue #161).
// mult<Accumulator>(A, B, C) sums each C element in Accumulator precision and
// rounds out to C's element type on store -- the result type is inferred from C
// (the Element -> Accumulate -> Result model, headline kernel of epic #157).
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>

#include <mtl/mat/dense2D.hpp>
#include <mtl/operation/mult.hpp>

using namespace mtl;
using Catch::Matchers::WithinRel;

namespace {
// Deterministic fill with mixed magnitudes / signs so the inner products have
// cancellation that low-precision accumulation loses.
template <typename T>
mat::dense2D<T> filled(std::size_t m, std::size_t n, int seed) {
    mat::dense2D<T> A(m, n);
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A(i, j) = static_cast<T>(std::sin(static_cast<double>((i * 13 + j * 7 + seed) % 97)));
    return A;
}
} // namespace

TEST_CASE("gemm default behavior is unchanged", "[operation][gemm][accumulator]") {
    mat::dense2D<double> A(2, 3), B(3, 2), C(2, 2);
    A(0, 0)=1; A(0, 1)=2; A(0, 2)=3; A(1, 0)=4; A(1, 1)=5; A(1, 2)=6;
    B(0, 0)=7; B(0, 1)=8; B(1, 0)=9; B(1, 1)=10; B(2, 0)=11; B(2, 1)=12;
    mult(A, B, C);
    REQUIRE_THAT(C(0, 0), WithinRel(58.0, 1e-12));   // 7+18+33
    REQUIRE_THAT(C(1, 1), WithinRel(154.0, 1e-12));  // 32+50+72
}

TEST_CASE("gemm accumulator precision is independent of element/result type",
          "[operation][gemm][accumulator]") {
    // fp32 operands, fp32 result tensor, but fp64 accumulation.
    const std::size_t m = 8, k = 200, n = 6;
    auto Af = filled<float>(m, k, 1);
    auto Bf = filled<float>(k, n, 2);
    auto Ad = filled<double>(m, k, 1);   // same values, double
    auto Bd = filled<double>(k, n, 2);

    mat::dense2D<double> Cref(m, n);
    mult(Ad, Bd, Cref);                  // double reference

    mat::dense2D<float> Cnaive(m, n);
    mult(Af, Bf, Cnaive);                // fp32 accumulate (default)

    mat::dense2D<float> Cwide(m, n);
    mult<double>(Af, Bf, Cwide);         // fp64 accumulate, fp32 result (from C)

    double err_naive = 0.0, err_wide = 0.0;
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            double ref = Cref(i, j);
            err_naive = std::max(err_naive, std::abs(static_cast<double>(Cnaive(i, j)) - ref));
            err_wide  = std::max(err_wide,  std::abs(static_cast<double>(Cwide(i, j))  - ref));
        }
    INFO("naive(f32-acc) err = " << err_naive << ", wide(f64-acc) err = " << err_wide);
    REQUIRE(err_wide <= err_naive);   // fp64 accumulate is at least as accurate
}
