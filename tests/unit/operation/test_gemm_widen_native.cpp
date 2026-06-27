// MTL5 -- native-fast coverage for the SIMD widening GEMM (issue #176).
//
// The default CI preset leaves MTL5_NATIVE_FAST_GEMM off, so the widening case in
// test_gemm_accumulator.cpp only exercises the generic scalar kernel in CI. This
// TU forces the macro on so mult<double>(A_float, B_float, C_double) routes through
// the blocked two-type GEMM (detail::gemm_blocked<double, float>) -- the #176 code
// path -- in every CI build. Without Highway the SIMD batch is the scalar fallback
// (width 1), which still exercises the two-type micro-kernel, the widening load
// fallback, edge tiles, and the dispatch; a Highway build runs the true SIMD path.
#ifndef MTL5_NATIVE_FAST_GEMM
#define MTL5_NATIVE_FAST_GEMM 1
#endif

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <initializer_list>

#include <mtl/mat/dense2D.hpp>
#include <mtl/operation/mult.hpp>

using namespace mtl;
using Catch::Matchers::WithinAbs;

namespace {
mat::dense2D<double> to_double(const mat::dense2D<float>& X) {
    mat::dense2D<double> Y(X.num_rows(), X.num_cols());
    for (std::size_t i = 0; i < X.num_rows(); ++i)
        for (std::size_t j = 0; j < X.num_cols(); ++j)
            Y(i, j) = static_cast<double>(X(i, j));
    return Y;
}
template <typename T>
mat::dense2D<T> filled(std::size_t m, std::size_t n, int seed) {
    mat::dense2D<T> A(m, n);
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A(i, j) = static_cast<T>(std::sin(static_cast<double>((i * 13 + j * 7 + seed) % 97)));
    return A;
}
} // namespace

// Same correctness contract as the generic-path widening test, but compiled with
// MTL5_NATIVE_FAST_GEMM so the result comes from the blocked widening kernel:
// float operands accumulated in fp64 must match a double GEMM on the EXACT widened
// operands to within summation-order rounding.
TEST_CASE("gemm widening via native-fast blocked kernel (#176)",
          "[operation][gemm][accumulator][widen][native]") {
    for (auto d : std::initializer_list<std::array<std::size_t, 3>>{
             {9, 200, 7}, {16, 64, 16}, {1, 130, 5}, {13, 1, 11}, {33, 257, 31}}) {
        const std::size_t m = d[0], k = d[1], n = d[2];
        auto Af = filled<float>(m, k, 3);
        auto Bf = filled<float>(k, n, 4);

        mat::dense2D<double> Cexact(m, n);
        mult(to_double(Af), to_double(Bf), Cexact);   // double GEMM on exact widened operands

        mat::dense2D<double> Cwide(m, n);
        mult<double>(Af, Bf, Cwide);                  // blocked widening path under test

        for (std::size_t i = 0; i < m; ++i)
            for (std::size_t j = 0; j < n; ++j)
                REQUIRE_THAT(Cwide(i, j), WithinAbs(Cexact(i, j), 1e-9));
    }
}
