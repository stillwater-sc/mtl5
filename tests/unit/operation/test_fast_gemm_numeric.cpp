// Numerical correctness of the native fast GEMM/GEMV path (#91) on *random
// floating-point* data -- the integer-exact tests (test_gemm_blocked.cpp,
// test_gemv.cpp) prove the structure; this proves the arithmetic is right to
// floating-point rounding across orientations, alpha/beta, rectangular and
// multi-block sizes.
//
// Reference: the same T-rounded inputs accumulated in DOUBLE-DOUBLE via FMA --
// an exact two-product plus a Knuth two-sum, giving ~106 bits. native-fast
// accumulates in T, so it may differ by up to ~k * eps(T); we allow a multiple
// of that and take the worst element per configuration (one assertion each). A
// real bug is O(1) off and trips this immediately.
//
// The reference used to be `long double`, which is 80-bit on x86-64, 64-bit --
// i.e. plain double -- on Apple ARM64 and MSVC, and 128-bit on ARM64 Linux.
// That is a ~60-bit spread across targets this repo already builds on, for a
// value described as "the trusted high-precision" reference. Double-double
// needs only IEEE double and is therefore identical everywhere (#386).
//
// Measured before the change: the observed error sits at ~0.002 of the
// tolerance, so no assertion was ever at risk from the reference width -- the
// portability problem was real but the exposure was not. What the exact
// reference DOES buy is room to tighten: see tol_for below.
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
//
// This is an EMPIRICAL tolerance calibrated to the data this file generates,
// not a worst-case bound. Saying so matters, because the obvious citation does
// not actually support a bound here:
//
//   Higham (Accuracy and Stability of Numerical Algorithms, 3.1) gives
//       |fl(x.y) - x.y|  <=  gamma_k * sum_i |x_i||y_i|,
//       gamma_k = k*u/(1 - k*u) ~ k*u,   u = unit roundoff = eps/2
//
//   Note two things. The bound is against sum|x_i||y_i|, NOT against |x.y|,
//   and this test compares relative to (|ref| + 1). The ratio between them is
//   the dot product's condition number. For the zero-mean uniform [-1, 1]
//   entries this file generates that is ~sqrt(k) -- E|x*y| = 1/4 gives
//   sum|x_i y_i| ~ k/4, while the sum itself is ~sqrt(k)/3 -- so the worst-case
//   RELATIVE error is ~k^1.5 * u. No modest multiple of k*eps dominates that as
//   k grows.
//
// What justifies the number is measurement, not the bound: uniform [-1, 1] data
// accumulates far below the worst case because the rounding errors partially
// cancel, giving ~sqrt(k)*u in practice. Every configuration in this file
// passes at a factor of 0.125, so 4 leaves roughly 32x over the worst measured
// case.
//
// 4 * k * eps is therefore 16x TIGHTER THAN THE PREVIOUS 64 -- that is the only
// comparison being claimed -- while keeping deliberate headroom for targets
// that cannot be measured here (ARM64, MSVC, and the FP-contract lane, where
// contraction changes the rounding). It is not, and should not be read as, a
// proven bound.
template <typename T>
double tol_for(std::size_t k) {
    return 4.0 * static_cast<double>(k) *
           static_cast<double>(std::numeric_limits<T>::epsilon());
}

// Exact-ish accumulation in double-double: Knuth two-sum plus an FMA-based
// exact two-product. Needs only IEEE double, so it is bit-identical on every
// target -- unlike long double.
struct dd_acc {
    double hi{}, lo{};
    void add(double x) {                       // two-sum
        const double s  = hi + x;
        const double bb = s - hi;
        lo += (hi - (s - bb)) + (x - bb);
        hi = s;
    }
    void add_product(double a, double b) {     // exact a*b, both halves
        const double p = a * b;
        add(p);
        add(std::fma(a, b, -p));
    }
    double value() const { return hi + lo; }
};

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

    const double tol = tol_for<T>(k);
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            dd_acc acc;
            for (std::size_t p = 0; p < k; ++p)
                acc.add_product(static_cast<double>(A(i, p)), static_cast<double>(B(p, j)));
            const double ref = acc.value();
            const double err = std::fabs(static_cast<double>(C(i, j)) - ref);
            if (err > tol * (std::fabs(ref) + 1.0)) return false;
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

    const double tol = tol_for<T>(n);
    for (std::size_t i = 0; i < m; ++i) {
        dd_acc acc;
        for (std::size_t j = 0; j < n; ++j)
            acc.add_product(static_cast<double>(A(i, j)), static_cast<double>(x(j)));
        const double ref = acc.value();
        if (std::fabs(static_cast<double>(y(i)) - ref) > tol * (std::fabs(ref) + 1.0))
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

    const double tol = tol_for<TestType>(k);
    bool ok = true;
    for (std::size_t i = 0; i < m && ok; ++i)
        for (std::size_t j = 0; j < n && ok; ++j) {
            dd_acc acc;
            for (std::size_t p = 0; p < k; ++p)
                acc.add_product(static_cast<double>(A[i * k + p]), static_cast<double>(B[p * n + j]));
            // beta*C0 + alpha*(A*B), the scalings applied to the exact product.
            const double ref = static_cast<double>(beta) * static_cast<double>(C0[i * n + j])
                             + static_cast<double>(alpha) * acc.value();
            if (std::fabs(static_cast<double>(C[i * n + j]) - ref) > tol * (std::fabs(ref) + 1.0))
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
