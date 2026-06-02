// Numerical correctness of the native fast GEMM/GEMV path (#91) on *random
// floating-point* data -- the integer-exact tests (test_gemm_blocked.cpp,
// test_gemv.cpp) prove the structure; this proves the arithmetic is right to
// floating-point rounding across orientations, alpha/beta, rectangular and
// multi-block sizes.
//
// Reference: the same T-rounded inputs accumulated in long double (the trusted
// high-precision generic triple product). native-fast accumulates in T, so it
// may differ by up to ~k * eps(T); we allow a generous multiple of that and
// take the worst element per configuration (one assertion each). A real bug is
// O(1) off and trips this immediately.
//
// MTL5_NATIVE_FAST_GEMM is defined before mult.hpp so mtl::mult routes through
// gemm_blocked / gemv for this translation unit.
#define MTL5_NATIVE_FAST_GEMM 1

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include <mtl/detail/gemm_blocked.hpp>
#include <mtl/operation/mult.hpp>
#include <mtl/simd/blocking.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/parameter.hpp>
#include <mtl/tag/orientation.hpp>
#include <mtl/vec/dense_vector.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

namespace {

using rowmaj = mtl::mat::parameters<mtl::tag::row_major>;
using colmaj = mtl::mat::parameters<mtl::tag::col_major>;

// Relative tolerance for a length-k contraction accumulated in T.
template <typename T>
long double tol_for(std::size_t k) {
    return 64.0L * static_cast<long double>(k) *
           static_cast<long double>(std::numeric_limits<T>::epsilon());
}

// C = A*B via mtl::mult (native-fast) vs a long-double reference over the same
// T-rounded inputs. MatA/MatB/MatC fix the orientations. Returns true if every
// element is within tolerance.
template <typename MatA, typename MatB, typename MatC>
bool gemm_ok(std::size_t m, std::size_t n, std::size_t k, std::uint64_t seed) {
    using T = typename MatC::value_type;
    MatA A(m, k); MatB B(k, n); MatC C(m, n);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t p = 0; p < k; ++p) A(i, p) = static_cast<T>(dist(rng));
    for (std::size_t p = 0; p < k; ++p)
        for (std::size_t j = 0; j < n; ++j) B(p, j) = static_cast<T>(dist(rng));

    mtl::mult(A, B, C);

    const long double tol = tol_for<T>(k);
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            long double ref = 0.0L;
            for (std::size_t p = 0; p < k; ++p)
                ref += static_cast<long double>(A(i, p)) * static_cast<long double>(B(p, j));
            const long double err = std::fabs(static_cast<long double>(C(i, j)) - ref);
            if (err > tol * (std::fabs(ref) + 1.0L)) return false;
        }
    return true;
}

// y = A*x via mtl::mult (native-fast) vs long-double reference.
template <typename MatA>
bool gemv_ok(std::size_t m, std::size_t n, std::uint64_t seed) {
    using T = typename MatA::value_type;
    MatA A(m, n);
    mtl::vec::dense_vector<T> x(n), y(m);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j) A(i, j) = static_cast<T>(dist(rng));
    for (std::size_t j = 0; j < n; ++j) x(j) = static_cast<T>(dist(rng));

    mtl::mult(A, x, y);

    const long double tol = tol_for<T>(n);
    for (std::size_t i = 0; i < m; ++i) {
        long double ref = 0.0L;
        for (std::size_t j = 0; j < n; ++j)
            ref += static_cast<long double>(A(i, j)) * static_cast<long double>(x(j));
        if (std::fabs(static_cast<long double>(y(i)) - ref) > tol * (std::fabs(ref) + 1.0L))
            return false;
    }
    return true;
}

// Square + rectangular sizes straddling mr/nr; tiny through a few hundred.
const std::size_t kSizes[] = {1, 2, 5, 7, 13, 16, 17, 31, 64, 100, 129};

} // namespace

TEMPLATE_TEST_CASE("fast GEMM: random data vs long-double ref (row-major)", "[operation][gemm][numeric]", float, double) {
    using R = mtl::mat::dense2D<TestType, rowmaj>;
    std::uint64_t seed = 1;
    for (std::size_t m : kSizes)
        for (std::size_t n : kSizes) {
            // rectangular k mixes too: k = n-ish and an unrelated value
            for (std::size_t k : {std::size_t(1), std::size_t(7), std::size_t(33)}) {
                INFO("m=" << m << " n=" << n << " k=" << k);
                CHECK(gemm_ok<R, R, R>(m, n, k, seed++));
            }
        }
}

TEMPLATE_TEST_CASE("fast GEMM: orientation combos (random)", "[operation][gemm][numeric]", float, double) {
    using RM = mtl::mat::dense2D<TestType, rowmaj>;
    using CM = mtl::mat::dense2D<TestType, colmaj>;
    const std::size_t m = 37, n = 29, k = 23;   // all odd, rectangular, non-mr/nr
    std::uint64_t s = 100;
    CHECK(gemm_ok<RM, RM, RM>(m, n, k, s++));
    CHECK(gemm_ok<CM, RM, RM>(m, n, k, s++));
    CHECK(gemm_ok<RM, CM, RM>(m, n, k, s++));
    CHECK(gemm_ok<CM, CM, RM>(m, n, k, s++));
    CHECK(gemm_ok<RM, RM, CM>(m, n, k, s++));   // col-major C (C^T = B^T A^T branch)
    CHECK(gemm_ok<CM, CM, CM>(m, n, k, s++));
}

TEMPLATE_TEST_CASE("fast GEMM: crosses mc/kc blocks (random)", "[operation][gemm][numeric]", float, double) {
    using R = mtl::mat::dense2D<TestType, rowmaj>;
    constexpr auto bp = mtl::simd::default_blocking<TestType>;
    const std::size_t m = bp.mc + bp.mr + 3;   // > mc
    const std::size_t k = bp.kc + 5;           // > kc
    const std::size_t n = bp.nr * 2 + 1;
    INFO("m=" << m << " n=" << n << " k=" << k << " (mc=" << bp.mc << " kc=" << bp.kc << ")");
    CHECK(gemm_ok<R, R, R>(m, n, k, 7));
}

TEMPLATE_TEST_CASE("fast GEMM: alpha/beta numeric (gemm_blocked)", "[operation][gemm][numeric]", float, double) {
    const std::size_t m = 31, n = 19, k = 23;
    std::vector<TestType> A(m * k), B(k * n), C(m * n), C0(m * n);
    std::mt19937_64 rng(55);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (auto& a : A) a = static_cast<TestType>(dist(rng));
    for (auto& b : B) b = static_cast<TestType>(dist(rng));
    for (std::size_t t = 0; t < m * n; ++t) { C[t] = static_cast<TestType>(dist(rng)); C0[t] = C[t]; }

    const TestType alpha = TestType(1.5), beta = TestType(-0.75);
    mtl::detail::gemm_blocked<TestType>(m, n, k, alpha, A.data(), (std::ptrdiff_t)k, 1,
                                        B.data(), (std::ptrdiff_t)n, 1, beta, C.data(), n);

    const long double tol = tol_for<TestType>(k);
    bool ok = true;
    for (std::size_t i = 0; i < m && ok; ++i)
        for (std::size_t j = 0; j < n && ok; ++j) {
            long double prod = 0.0L;
            for (std::size_t p = 0; p < k; ++p)
                prod += static_cast<long double>(A[i * k + p]) * static_cast<long double>(B[p * n + j]);
            const long double ref = static_cast<long double>(beta) * static_cast<long double>(C0[i * n + j])
                                  + static_cast<long double>(alpha) * prod;
            if (std::fabs(static_cast<long double>(C[i * n + j]) - ref) > tol * (std::fabs(ref) + 1.0L))
                ok = false;
        }
    CHECK(ok);
}

TEMPLATE_TEST_CASE("fast GEMV: random data vs long-double ref (both orientations)", "[operation][gemv][numeric]", float, double) {
    using RM = mtl::mat::dense2D<TestType, rowmaj>;
    using CM = mtl::mat::dense2D<TestType, colmaj>;
    std::uint64_t seed = 500;
    for (std::size_t m : kSizes)
        for (std::size_t n : kSizes) {
            INFO("m=" << m << " n=" << n);
            CHECK(gemv_ok<RM>(m, n, seed++));
            CHECK(gemv_ok<CM>(m, n, seed++));
        }
}
