// Tests for the optimized native GEMV (#87) and its mult(A,x,y) dispatch wiring.
// Integer-valued data => exact products/sums in float/double regardless of
// SIMD/FMA order, so we compare bit-exactly to a naive reference. One assertion
// per configuration (bool "all matched"), not one per element.
//
// Define the gate BEFORE including mult.hpp so the native path is exercised
// through mtl::mult for this translation unit.
#define MTL5_NATIVE_FAST_GEMM 1

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include <mtl/detail/gemv.hpp>
#include <mtl/operation/mult.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/parameter.hpp>
#include <mtl/tag/orientation.hpp>
#include <mtl/vec/dense_vector.hpp>

#include <cstddef>
#include <vector>

namespace {

template <typename T> T av(std::size_t i, std::size_t j) { return T((i * 3 + j) % 7) - 3; }
template <typename T> T xv(std::size_t j) { return T((j * 2) % 5) - 2; }

// Build A in the requested orientation, run the matching kernel, and return
// whether y == naive A*x everywhere.
template <typename T>
bool gemv_matches(std::size_t m, std::size_t n, bool rowmajor) {
    std::vector<T> A(m * n), x(n), y(m, T(-777));
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A[rowmajor ? i * n + j : j * m + i] = av<T>(i, j);
    for (std::size_t j = 0; j < n; ++j) x[j] = xv<T>(j);

    if (rowmajor) mtl::detail::gemv_rowmajor<T>(m, n, A.data(), n, x.data(), y.data());
    else          mtl::detail::gemv_colmajor<T>(m, n, A.data(), m, x.data(), y.data());

    for (std::size_t i = 0; i < m; ++i) {
        T s = 0;
        for (std::size_t j = 0; j < n; ++j) s += av<T>(i, j) * xv<T>(j);
        if (y[i] != s) return false;
    }
    return true;
}

// Sizes straddling the SIMD width, the MR=4 row block, and the UB*W y-strip;
// 0 exercises the empty-shape / zero-iteration paths (incl. m==n==0).
const std::size_t kDims[] = {0, 1, 2, 3, 4, 5, 7, 8, 9, 13, 16, 17, 31, 33, 64, 100};

} // namespace

TEMPLATE_TEST_CASE("gemv: row-major and col-major vs naive, all sizes", "[operation][gemv]", float, double) {
    for (std::size_t m : kDims)
        for (std::size_t n : kDims) {
            INFO("m=" << m << " n=" << n);
            CHECK(gemv_matches<TestType>(m, n, true));   // row-major (dot-of-rows)
            CHECK(gemv_matches<TestType>(m, n, false));  // col-major (axpy-of-columns)
        }
}

// mult(A,x,y) dispatch: with MTL5_NATIVE_FAST_GEMM defined, dense2D float/double
// x dense_vector routes through the native GEMV for both A orientations.
namespace {
using rowmaj = mtl::mat::parameters<mtl::tag::row_major>;
using colmaj = mtl::mat::parameters<mtl::tag::col_major>;

template <typename MatA>
bool mult_matches(std::size_t m, std::size_t n) {
    using T = typename MatA::value_type;
    MatA A(m, n);
    mtl::vec::dense_vector<T> x(n), y(m);
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j) A(i, j) = av<T>(i, j);
    for (std::size_t j = 0; j < n; ++j) x(j) = xv<T>(j);

    mtl::mult(A, x, y);

    for (std::size_t i = 0; i < m; ++i) {
        T s = 0;
        for (std::size_t j = 0; j < n; ++j) s += av<T>(i, j) * xv<T>(j);
        if (y(i) != s) return false;
    }
    return true;
}
} // namespace

TEMPLATE_TEST_CASE("mult() native GEMV dispatch matches naive (row- and col-major A)", "[operation][gemv][dispatch]", float, double) {
    const std::size_t cases[][2] = {{0, 0}, {0, 5}, {7, 0}, {1, 1}, {7, 5}, {16, 16}, {33, 20}, {100, 64}};
    for (auto& c : cases) {
        INFO("m=" << c[0] << " n=" << c[1]);
        CHECK(mult_matches<mtl::mat::dense2D<TestType, rowmaj>>(c[0], c[1]));
        CHECK(mult_matches<mtl::mat::dense2D<TestType, colmaj>>(c[0], c[1]));
    }
}
