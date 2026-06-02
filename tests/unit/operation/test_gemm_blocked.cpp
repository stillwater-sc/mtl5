// Tests for the blocked GEMM macro-kernel (#90) and its mult() dispatch wiring.
// Integer-valued data => exact products/sums in float/double regardless of
// blocking/FMA order, so we compare bit-exactly to a naive triple-loop ref.
//
// One assertion per configuration (a bool "all elements matched"), not one per
// element -- Catch2 CHECK is expensive, and the size sweep would otherwise fire
// millions of them. Multi-block sizes are derived from default_blocking so the
// cache-blocking nest is exercised (crossing mc/kc) regardless of SIMD width.
//
// Define the gate BEFORE including mult.hpp so the native-fast path compiles in
// and is exercised through mtl::mult for this translation unit.
#define MTL5_NATIVE_FAST_GEMM 1

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include <mtl/detail/gemm_blocked.hpp>
#include <mtl/operation/mult.hpp>
#include <mtl/simd/blocking.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/parameter.hpp>
#include <mtl/tag/orientation.hpp>

#include <cstddef>
#include <vector>

namespace {

template <typename T>
T av(std::size_t i, std::size_t j) { return T((i * 5 + j) % 7) - 3; }   // A entries
template <typename T>
T bv(std::size_t p, std::size_t j) { return T((p * 3 + 2 * j) % 5) - 2; } // B entries

// gemm_blocked produces a row-major C. Feed A and B in either orientation (the
// packing normalizes via strides) and return whether C == naive A*B everywhere.
template <typename T>
bool blocked_matches(std::size_t m, std::size_t n, std::size_t k, bool a_rowmaj, bool b_rowmaj) {
    std::vector<T> A(m * k), B(k * n), C(m * n, T(-777));
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t p = 0; p < k; ++p)
            A[a_rowmaj ? i * k + p : p * m + i] = av<T>(i, p);
    for (std::size_t p = 0; p < k; ++p)
        for (std::size_t j = 0; j < n; ++j)
            B[b_rowmaj ? p * n + j : j * k + p] = bv<T>(p, j);

    const std::ptrdiff_t a_rs = a_rowmaj ? (std::ptrdiff_t)k : 1, a_cs = a_rowmaj ? 1 : (std::ptrdiff_t)m;
    const std::ptrdiff_t b_rs = b_rowmaj ? (std::ptrdiff_t)n : 1, b_cs = b_rowmaj ? 1 : (std::ptrdiff_t)k;
    mtl::detail::gemm_blocked<T>(m, n, k, T(1), A.data(), a_rs, a_cs, B.data(), b_rs, b_cs,
                                 T(0), C.data(), n);

    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            T s = 0;
            for (std::size_t p = 0; p < k; ++p) s += av<T>(i, p) * bv<T>(p, j);
            if (C[i * n + j] != s) return false;
        }
    return true;
}

// Sizes that straddle the mr/nr register tile and small kc edges.
const std::size_t kEdge[] = {1, 2, 3, 4, 5, 7, 8, 9, 13, 16, 17};

} // namespace

TEMPLATE_TEST_CASE("gemm_blocked: C=A*B, all orientations, edge sizes", "[operation][gemm][blocked]", float, double) {
    for (std::size_t m : kEdge)
        for (std::size_t n : kEdge)
            for (std::size_t k : kEdge) {
                INFO("m=" << m << " n=" << n << " k=" << k);
                CHECK(blocked_matches<TestType>(m, n, k, true,  true));
                CHECK(blocked_matches<TestType>(m, n, k, false, true));   // A col-major
                CHECK(blocked_matches<TestType>(m, n, k, true,  false));  // B col-major
                CHECK(blocked_matches<TestType>(m, n, k, false, false));  // both col-major
            }
}

TEMPLATE_TEST_CASE("gemm_blocked: crosses mc/kc cache blocks", "[operation][gemm][blocked]", float, double) {
    // Derive sizes from the actual blocking so the ic/pc loops iterate >1 time
    // regardless of SIMD width: m past one mc block, k past one kc block.
    constexpr auto bp = mtl::simd::default_blocking<TestType>;
    const std::size_t m = bp.mc + bp.mr + 1;   // > mc  -> multiple ic blocks + edge
    const std::size_t k = bp.kc + 3;           // > kc  -> multiple pc blocks + edge
    const std::size_t n = bp.nr * 3 + 1;       // a few nr panels + edge
    INFO("m=" << m << " n=" << n << " k=" << k << "  (mc=" << bp.mc << " kc=" << bp.kc << ")");
    CHECK(blocked_matches<TestType>(m, n, k, true,  true));
    CHECK(blocked_matches<TestType>(m, n, k, false, false));   // both col-major
}

TEMPLATE_TEST_CASE("gemm_blocked: alpha/beta", "[operation][gemm][blocked]", float, double) {
    const std::size_t m = 9, n = 7, k = 5;
    std::vector<TestType> A(m * k), B(k * n);
    for (std::size_t i = 0; i < m; ++i) for (std::size_t p = 0; p < k; ++p) A[i * k + p] = av<TestType>(i, p);
    for (std::size_t p = 0; p < k; ++p) for (std::size_t j = 0; j < n; ++j) B[p * n + j] = bv<TestType>(p, j);

    const TestType alpha = TestType(2), beta = TestType(3);
    std::vector<TestType> C(m * n), ref(m * n);
    for (std::size_t t = 0; t < m * n; ++t) C[t] = TestType(t % 4) - 1;
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            TestType prod = 0;
            for (std::size_t p = 0; p < k; ++p) prod += av<TestType>(i, p) * bv<TestType>(p, j);
            ref[i * n + j] = beta * C[i * n + j] + alpha * prod;
        }

    mtl::detail::gemm_blocked<TestType>(m, n, k, alpha, A.data(), (std::ptrdiff_t)k, 1,
                                        B.data(), (std::ptrdiff_t)n, 1, beta, C.data(), n);
    bool ok = true;
    for (std::size_t t = 0; t < m * n; ++t) ok = ok && (C[t] == ref[t]);
    CHECK(ok);
}

// mult() dispatch: with MTL5_NATIVE_FAST_GEMM defined, dense2D float/double
// routes through gemm_blocked. Verify both a row-major and a col-major C (the
// latter exercises the C^T = B^T A^T branch).
namespace {
using rowmaj = mtl::mat::parameters<mtl::tag::row_major>;
using colmaj = mtl::mat::parameters<mtl::tag::col_major>;

template <typename Mat>
void fill(Mat& M, bool isA) {
    for (std::size_t i = 0; i < M.num_rows(); ++i)
        for (std::size_t j = 0; j < M.num_cols(); ++j)
            M(i, j) = isA ? av<typename Mat::value_type>(i, j) : bv<typename Mat::value_type>(i, j);
}

template <typename MC, typename MA, typename MB>
bool mult_matches(const MA& A, const MB& B, std::size_t m, std::size_t n, std::size_t k) {
    MC C(m, n);
    mtl::mult(A, B, C);
    using T = typename MC::value_type;
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            T s = 0;
            for (std::size_t p = 0; p < k; ++p) s += A(i, p) * B(p, j);
            if (C(i, j) != s) return false;
        }
    return true;
}
} // namespace

TEMPLATE_TEST_CASE("mult() native-fast dispatch matches naive (row- and col-major C)", "[operation][gemm][blocked][dispatch]", float, double) {
    const std::size_t m = 13, n = 10, k = 6;
    mtl::mat::dense2D<TestType, rowmaj> A(m, k), B(k, n);
    fill(A, true); fill(B, false);

    CHECK(mult_matches<mtl::mat::dense2D<TestType, rowmaj>>(A, B, m, n, k));  // row-major C
    CHECK(mult_matches<mtl::mat::dense2D<TestType, colmaj>>(A, B, m, n, k));  // col-major C
}
