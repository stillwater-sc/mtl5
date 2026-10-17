// The configuration-1/2/3 accumulation matrix, MTL5 slice (#586).
//
// The same routine runs under each accumulation configuration of
// accumulator_traits.hpp, and its error is measured EXACTLY against the true
// sum of products:
//
//   1. plain  (Acc = T)                  product rounds, then the sum rounds
//   2. FMA    (fma_accumulator<T>)       one rounding per term
//   3. exact  (exact_test::exact_acc<T>) one rounding at the end
//
// Configuration 3 is played here by a test-only Kulisch accumulator
// (exact_accumulator.hpp), which keeps MTL5's CI Universal-free. The peer repos
// run the same rows with `sw::universal::quire<float>` / `quire<double>`; each
// of their tests names the MTL5 test case it mirrors by the tag in brackets,
// e.g. [586:dot], so the matrix reads across repositories.
//
// What is asserted, and why it is the honest statement:
//
//   * Configuration 3 IS the correctly rounded result, bit for bit. Through
//     MTL5's own dot / mult / norm plumbing, threaded merges included.
//   * err(3) <= err(2) and err(3) <= err(1) on EVERY instance: the correctly
//     rounded value is the T nearest the exact sum, so no T can be closer.
//   * Configurations 1 and 2 each stay within the classical error bound of a
//     recursively summed dot product, gamma_n * sum |x_i y_i| (Higham, Accuracy
//     and Stability of Numerical Algorithms, 2nd ed., section 3.1), with
//     gamma_n = n u / (1 - n u). That is what makes the error grow with the
//     condition number, while configuration 3's does not.
//   * err(2) <= err(1) is a per-TERM guarantee (accumulator_traits.hpp), and it
//     is asserted where the product rounding is the whole error. It is NOT a
//     per-sum ordering, and on ill-conditioned dot products it does not hold
//     even on average: there the error is the cancellation in the running sum,
//     which FMA does not touch. Measured on the battery below, 40 instances of
//     n = 100, mean relative error plain vs fma:
//
//         float   cond 1e4   5.1e-3 vs 3.4e-3    cond 1e6   0.136 vs 0.101
//         double  cond 1e8   3.4e-8 vs 3.2e-8    cond 1e12  1.06e-4 vs 1.15e-4
//
//     so the ordering of 1 and 2 flips with the instance; only 3 dominates.
//
// Built with -ffp-contract=off (tests/unit/CMakeLists.txt). Configuration 1 is
// "the product rounds": a compiler that contracts acc += a*b into an FMA turns
// configuration 1 into configuration 2, and the comparison measures nothing.
//
// Sets MTL5_NUM_THREADS before the pool's first use, so the reductions split.
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>

#include <mtl/detail/thread_pool.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/math/accumulator_traits.hpp>
#include <mtl/operation/dot.hpp>
#include <mtl/operation/mult.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/vec/dense_vector.hpp>

#include "exact_accumulator.hpp"

namespace {

const int g_set_threads = [] {
#if defined(_WIN32)
    _putenv_s("MTL5_NUM_THREADS", "4");
#else
    setenv("MTL5_NUM_THREADS", "4", /*overwrite=*/1);
#endif
    return 0;
}();

using exact_test::exact_acc;
using exact_test::err;
template <typename T> using fma_acc = mtl::math::fma_accumulator<T>;
template <typename T> using vec = mtl::vec::dense_vector<T>;

static_assert(mtl::math::adl_accumulator<exact_acc<float>, float>);
static_assert(mtl::math::adl_accumulator<exact_acc<double>, double>);
static_assert(mtl::math::mergeable_accumulator<exact_acc<float>, float>);

template <typename T>
bool same_bits(T a, T b) { return std::memcmp(&a, &b, sizeof(T)) == 0; }

/// A portable random source: the raw mt19937_64 stream is specified by the
/// standard; the distributions are not, and differ between standard libraries.
struct rng {
    std::mt19937_64 g;
    explicit rng(std::uint64_t seed) : g(seed) {}
    double uniform() { return static_cast<double>(g() >> 11) * 0x1p-53; }   // [0, 1)
    double symmetric() { return 2.0 * uniform() - 1.0; }                     // [-1, 1)
    int integer(int lo, int hi) {                                            // [lo, hi]
        return lo + static_cast<int>(g() % static_cast<std::uint64_t>(hi - lo + 1));
    }
    template <typename V>
    void shuffle(V& x, V& y) {
        for (std::size_t i = x.size(); i > 1; --i) {
            const std::size_t j = static_cast<std::size_t>(g() % i);
            std::swap(x(i - 1), x(j));
            std::swap(y(i - 1), y(j));
        }
    }
};

template <typename T, typename V>
exact_acc<T> exact_dot(const V& x, const V& y, std::size_t n) {
    exact_acc<T> a = exact_test::zero<T>();
    for (std::size_t i = 0; i < n; ++i) accumulator_add_product(a, x(i), y(i));
    return a;
}

template <typename T>
T round_out(const exact_acc<T>& a) { return accumulator_value(a, std::type_identity<T>{}); }

/// An ill-conditioned dot product of condition number about `cond`: Ogita,
/// Rump and Oishi, "Accurate sum and dot product", SIAM J. Sci. Comput. 26
/// (2005), Algorithm 6.1 (GenDot). The second half cancels the first, term by
/// term, against the exact running sum -- which the exact accumulator supplies.
template <typename T>
void gen_dot(std::size_t n, double cond, rng& r, vec<T>& x, vec<T>& y) {
    x.change_dim(n);
    y.change_dim(n);
    const std::size_t n2 = n / 2;
    const int b = static_cast<int>(std::log2(cond));
    const int bh = b / 2;
    for (std::size_t i = 0; i < n2; ++i) {
        const int e = i == 0 ? bh : (i + 1 == n2 ? 0 : r.integer(0, bh));
        x(i) = static_cast<T>(std::ldexp(r.symmetric(), e));
        y(i) = static_cast<T>(std::ldexp(r.symmetric(), e));
    }
    for (std::size_t i = n2; i < n; ++i) {
        const int e = static_cast<int>(std::lround(
            bh - static_cast<double>(bh) * static_cast<double>(i - n2) / static_cast<double>(n - n2)));
        x(i) = static_cast<T>(std::ldexp(r.symmetric(), e));
        const T partial = round_out(exact_dot<T>(x, y, i));
        y(i) = static_cast<T>((static_cast<T>(std::ldexp(r.symmetric(), e)) - partial) / x(i));
    }
    r.shuffle(x, y);
}

bool pool_is_threaded() {
    if (mtl::detail::thread_pool::instance().size() < 2) {
        WARN("pool < 2 workers: the threaded split is not exercised");
        return false;
    }
    return true;
}

} // namespace

// ---------------------------------------------------------------------------
// The reference itself.
// ---------------------------------------------------------------------------

TEST_CASE("exact accumulator: exact sums and one correctly rounded round-out",
          "[586:exact][accumulator]") {
    using exact_test::zero;
    SECTION("a single product rounds to the IEEE product") {
        rng r(1);
        for (int i = 0; i < 10000; ++i) {
            const float a = static_cast<float>(std::ldexp(r.symmetric(), r.integer(-60, 60)));
            const float b = static_cast<float>(std::ldexp(r.symmetric(), r.integer(-60, 60)));
            auto acc = zero<float>();
            accumulator_add_product(acc, a, b);
            REQUIRE(same_bits(round_out(acc), a * b));
            const double ad = r.symmetric() * 1e150, bd = r.symmetric() * 1e-140;
            auto accd = zero<double>();
            accumulator_add_product(accd, ad, bd);
            REQUIRE(same_bits(round_out(accd), ad * bd));
        }
    }
    SECTION("the product's rounding error is held exactly (TwoProduct)") {
        rng r(2);
        for (int i = 0; i < 10000; ++i) {
            const double a = r.symmetric(), b = r.symmetric();
            const double p = a * b;
            auto acc = zero<double>();
            accumulator_add_product(acc, a, b);
            accumulator_add_product(acc, -p, 1.0);
            REQUIRE(round_out(acc) == std::fma(a, b, -p));
        }
    }
    SECTION("ties round to even; a sticky bit breaks the tie") {
        auto acc = zero<float>();
        accumulator_add_product(acc, 1.0f, 1.0f);
        accumulator_add_product(acc, 0x1p-24f, 1.0f);              // exactly half an ulp
        REQUIRE(round_out(acc) == 1.0f);
        accumulator_add_product(acc, 0x1p-60f, 0x1p-60f);          // a sticky bit far below
        REQUIRE(round_out(acc) == 1.0f + 0x1p-23f);
        auto odd = zero<float>();
        accumulator_add_product(odd, 1.0f + 0x1p-23f, 1.0f);
        accumulator_add_product(odd, 0x1p-24f, 1.0f);              // tie, odd: rounds up
        REQUIRE(round_out(odd) == 1.0f + 0x1p-22f);
    }
    SECTION("cancellation to zero, negative sums, subnormals and overflow") {
        auto acc = zero<float>();
        accumulator_add_product(acc, 3.0f, 7.0f);
        accumulator_add_product(acc, -3.0f, 7.0f);
        REQUIRE(round_out(acc) == 0.0f);
        accumulator_add_product(acc, -0x1p-149f, 0x1p-10f);        // below every float
        REQUIRE(round_out(acc) == -0.0f);
        REQUIRE(accumulator_value(acc, std::type_identity<double>{}) == -0x1p-159);
        auto sub = zero<float>();
        accumulator_add_product(sub, 0x1p-75f, 0x1p-75f);           // 2^-150: half the least
        REQUIRE(round_out(sub) == 0.0f);                            // tie to even: zero
        accumulator_add_product(sub, 0x1p-75f, 0x1p-76f);           // + 2^-151: past the tie
        REQUIRE(round_out(sub) == 0x1p-149f);
        auto big = zero<float>();
        accumulator_add_product(big, 0x1p100f, 0x1p100f);
        REQUIRE(round_out(big) == std::numeric_limits<float>::infinity());
        REQUIRE(accumulator_value(big, std::type_identity<double>{}) == 0x1p200);
        auto bigd = zero<double>();
        accumulator_add_product(bigd, std::numeric_limits<double>::max(), 2.0);
        accumulator_add_product(bigd, -std::numeric_limits<double>::max(), 1.5);
        REQUIRE(round_out(bigd) == std::numeric_limits<double>::max() / 2);
    }
    SECTION("merge equals one sequential sum; order does not matter") {
        rng r(3);
        vec<double> x, y;
        gen_dot<double>(200, 1e20, r, x, y);
        const auto all = exact_dot<double>(x, y, 200);
        auto lo = zero<double>(), hi = zero<double>();
        for (std::size_t i = 0; i < 200; ++i)
            accumulator_add_product(i < 77 ? lo : hi, x(i), y(i));
        accumulator_merge(lo, hi);
        REQUIRE(lo.w == all.w);
        r.shuffle(x, y);
        REQUIRE(exact_dot<double>(x, y, 200).w == all.w);
    }
}

// ---------------------------------------------------------------------------
// Inner products.
// ---------------------------------------------------------------------------

TEST_CASE("[586:dot] FMA removes the product rounding, exactly",
          "[586:dot][accumulator]") {
    // dot([-p, a], [1, b]) with p = fl(a*b): the exact value is the rounding
    // error of a*b. Configuration 1 rounds a*b back to p and returns 0;
    // configuration 2 forms a*b - p unrounded; configuration 3 is exact.
    rng r(4);
    int nonzero = 0;
    for (int i = 0; i < 2000; ++i) {
        const float a = static_cast<float>(r.symmetric()), b = static_cast<float>(r.symmetric());
        const float p = a * b;
        vec<float> x(2), y(2);
        x(0) = -p; y(0) = 1.0f;
        x(1) = a;  y(1) = b;
        const auto exact = exact_dot<float>(x, y, 2);
        const float c1 = mtl::dot<float, float>(x, y);
        const float c2 = mtl::dot<fma_acc<float>, float>(x, y);
        const float c3 = mtl::dot<exact_acc<float>, float>(x, y);
        REQUIRE(c1 == 0.0f);
        REQUIRE(err(exact, c2) == 0.0);
        REQUIRE(err(exact, c3) == 0.0);
        if (c2 != 0.0f) ++nonzero;
    }
    REQUIRE(nonzero > 1000);   // most products of random floats are inexact
}

namespace {

/// gamma_n * sum |x_i y_i|: the error bound of a recursively summed dot product
/// of n terms in T, whether each product rounds (configuration 1) or not (2).
/// gamma_{n+1}, to absorb the rounding of the exact sum to double.
template <typename T, typename V>
double dot_error_bound(const V& x, const V& y, std::size_t n) {
    auto a = exact_test::zero<T>();
    for (std::size_t i = 0; i < n; ++i) accumulator_add_product(a, std::fabs(x(i)), std::fabs(y(i)));
    const double u = std::numeric_limits<T>::epsilon() / 2;
    const double nu = static_cast<double>(n + 1) * u;
    return nu / (1.0 - nu) * accumulator_value(a, std::type_identity<double>{});
}

/// The ill-conditioned dot battery for element type T, at one condition number.
template <typename T>
void dot_battery(double cond, std::size_t n, int instances) {
    rng r(static_cast<std::uint64_t>(cond));
    vec<T> x, y;
    for (int s = 0; s < instances; ++s) {
        gen_dot<T>(n, cond, r, x, y);
        const auto exact = exact_dot<T>(x, y, n);
        const T c1 = mtl::dot<T, T>(x, y);
        const T c2 = mtl::dot<fma_acc<T>, T>(x, y);
        const T c3 = mtl::dot<exact_acc<T>, T>(x, y);
        INFO("cond=" << cond << " instance=" << s);
        REQUIRE(same_bits(c3, round_out(exact)));        // config 3 is correctly rounded
        const double e1 = err(exact, c1), e2 = err(exact, c2), e3 = err(exact, c3);
        REQUIRE(e3 <= e2);
        REQUIRE(e3 <= e1);
        const double bound = dot_error_bound<T>(x, y, n);
        REQUIRE(e1 <= bound);
        REQUIRE(e2 <= bound);
    }
}

} // namespace

TEST_CASE("[586:dot] ill-conditioned dot: err(3) <= err(1), err(2), both within the bound",
          "[586:dot][accumulator]") {
    SECTION("float") {
        for (double cond : {1e2, 1e4, 1e6, 1e8, 1e12})
            dot_battery<float>(cond, 100, 40);
    }
    SECTION("double") {
        for (double cond : {1e4, 1e8, 1e12, 1e16, 1e24})
            dot_battery<double>(cond, 100, 40);
    }
}

TEST_CASE("[586:gemm] gemm: configuration 3 correctly rounded, 1 and 2 within the bound",
          "[586:gemm][accumulator]") {
    // Row i of A and column i of B are one ill-conditioned dot product, so the
    // diagonal of C is the dot battery through mult<Acc>; the off-diagonal
    // entries are ordinary random dots. exact_acc<float> is 80 bytes, so mult
    // keeps the dot-product kernel for it (#587's acc_panel_gemm_v).
    constexpr std::size_t M = 24, K = 64;
    rng r(5);
    mtl::mat::dense2D<float> A(M, K), B(K, M);
    vec<float> x, y;
    for (std::size_t i = 0; i < M; ++i) {
        gen_dot<float>(K, 1e6, r, x, y);
        for (std::size_t k = 0; k < K; ++k) { A(i, k) = x(k); B(k, i) = y(k); }
    }
    mtl::mat::dense2D<float> C1(M, M), C2(M, M), C3(M, M);
    mtl::mult<float>(A, B, C1);
    mtl::mult<fma_acc<float>>(A, B, C2);
    mtl::mult<exact_acc<float>>(A, B, C3);

    for (std::size_t i = 0; i < M; ++i)
        for (std::size_t j = 0; j < M; ++j) {
            auto exact = exact_test::zero<float>();
            for (std::size_t k = 0; k < K; ++k) accumulator_add_product(exact, A(i, k), B(k, j));
            INFO("C(" << i << ", " << j << ")");
            REQUIRE(same_bits(C3(i, j), round_out(exact)));
            const double e1 = err(exact, C1(i, j)), e2 = err(exact, C2(i, j));
            REQUIRE(err(exact, C3(i, j)) <= e2);
            REQUIRE(err(exact, C3(i, j)) <= e1);
            vec<float> row(K), col(K);
            for (std::size_t k = 0; k < K; ++k) { row(k) = A(i, k); col(k) = B(k, j); }
            const double bound = dot_error_bound<float>(row, col, K);
            REQUIRE(e1 <= bound);
            REQUIRE(e2 <= bound);
        }
}

// ---------------------------------------------------------------------------
// Norms: a wide dynamic range.
// ---------------------------------------------------------------------------

TEST_CASE("[586:norms] two_norm and frobenius_norm: no overflow or underflow loss",
          "[586:norms][accumulator]") {
    // In float, the square of 1e30 overflows and the square of 1e-30 underflows
    // to zero. Configurations 1 and 2 at float form or round that square in
    // float; the exact accumulator does not, and rounds the sum of squares once
    // -- to double here, through Result -- before the square root.
    auto expect = [](const auto& v) {
        auto a = exact_test::zero<float>();
        for (std::size_t i = 0; i < v.size(); ++i) accumulator_add_product(a, v(i), v(i));
        return std::sqrt(accumulator_value(a, std::type_identity<double>{}));
    };
    SECTION("large: plain and fma overflow, exact does not") {
        vec<float> v(1000);
        for (std::size_t i = 0; i < v.size(); ++i)
            v(i) = (i % 3 ? 1e30f : -3e29f) * static_cast<float>(1 + i % 7);
        REQUIRE(std::isinf(mtl::two_norm<float, double>(v)));
        REQUIRE(std::isinf(mtl::two_norm<fma_acc<float>, double>(v)));
        const double n3 = mtl::two_norm<exact_acc<float>, double>(v);
        REQUIRE(std::isfinite(n3));
        REQUIRE(n3 == expect(v));
    }
    SECTION("small: plain and fma underflow to zero, exact does not") {
        vec<float> v(1000);
        for (std::size_t i = 0; i < v.size(); ++i)
            v(i) = 1e-30f * static_cast<float>(1 + i % 5);
        REQUIRE(mtl::two_norm<float, double>(v) == 0.0);
        REQUIRE(mtl::two_norm<fma_acc<float>, double>(v) == 0.0);
        const double n3 = mtl::two_norm<exact_acc<float>, double>(v);
        REQUIRE(n3 > 0.0);
        REQUIRE(n3 == expect(v));
    }
    SECTION("mixed range, matrix: frobenius_norm") {
        mtl::mat::dense2D<float> m(40, 30);
        vec<float> flat(40 * 30);
        for (std::size_t r = 0; r < 40; ++r)
            for (std::size_t c = 0; c < 30; ++c) {
                const float s = (r + c) % 4 == 0 ? 1e25f : ((r + c) % 4 == 1 ? 1.0f : 1e-28f);
                m(r, c) = s * static_cast<float>(1 + (r * 30 + c) % 11);
                flat(r * 30 + c) = m(r, c);
            }
        REQUIRE(std::isinf(mtl::frobenius_norm<float, double>(m)));
        REQUIRE(std::isinf(mtl::frobenius_norm<fma_acc<float>, double>(m)));
        REQUIRE(mtl::frobenius_norm<exact_acc<float>, double>(m) == expect(flat));
        // A wide plain accumulator (configuration 1 at double) also survives
        // this range; it is the element-precision comparison that isolates the
        // accumulation.
        // 1200 squares summed in double: within gamma_1201, plus the sqrt.
        const double wide = mtl::frobenius_norm<double, double>(m);
        REQUIRE(std::fabs(wide - expect(flat)) <= 1e-12 * expect(flat));
    }
}

// ---------------------------------------------------------------------------
// Determinism.
// ---------------------------------------------------------------------------

TEST_CASE("[586:determinism] Accumulator = void is configuration 1 at the same type",
          "[586:determinism][accumulator]") {
#ifdef MTL5_HAS_BLAS
    SKIP("with BLAS, Accumulator = void dispatches to the vendor kernel, whose order is its own");
#else
    // The default GEMM and GEMV sum each entry in ascending k, as the
    // accumulator kernels do, so the bits agree. (dot and two_norm with
    // Accumulator = void take a SIMD reduction that splits the sum across
    // lanes; that is a different order by design, and not claimed here.)
    rng r(6);
    constexpr std::size_t M = 37, N = 29, K = 53;
    mtl::mat::dense2D<double> A(M, K), B(K, N), Cv(M, N), Ca(M, N);
    for (std::size_t i = 0; i < M; ++i)
        for (std::size_t k = 0; k < K; ++k) A(i, k) = r.symmetric();
    for (std::size_t k = 0; k < K; ++k)
        for (std::size_t j = 0; j < N; ++j) B(k, j) = r.symmetric();
    mtl::mult(A, B, Cv);
    mtl::mult<double>(A, B, Ca);
    for (std::size_t i = 0; i < M; ++i)
        for (std::size_t j = 0; j < N; ++j) {
            INFO("C(" << i << ", " << j << ")");
            REQUIRE(same_bits(Cv(i, j), Ca(i, j)));
        }
    vec<double> x(K), yv(M), ya(M);
    for (std::size_t k = 0; k < K; ++k) x(k) = r.symmetric();
    mtl::mult(A, x, yv);
    mtl::mult<double>(A, x, ya);
    for (std::size_t i = 0; i < M; ++i) REQUIRE(same_bits(yv(i), ya(i)));
#endif
}

TEST_CASE("[586:determinism] configuration 3 is independent of summation order",
          "[586:determinism][accumulator]") {
    // Shuffle one ill-conditioned dot product ten ways. Configuration 3 gives
    // the same bits every time; configurations 1 and 2 do not.
    rng r(7);
    vec<float> x, y;
    gen_dot<float>(2000, 1e8, r, x, y);
    const float c3 = mtl::dot<exact_acc<float>, float>(x, y);
    const float c1 = mtl::dot<float, float>(x, y);
    const float c2 = mtl::dot<fma_acc<float>, float>(x, y);
    bool c1_moved = false, c2_moved = false;
    for (int s = 0; s < 10; ++s) {
        r.shuffle(x, y);
        REQUIRE(same_bits(mtl::dot<exact_acc<float>, float>(x, y), c3));
        c1_moved = c1_moved || !same_bits(mtl::dot<float, float>(x, y), c1);
        c2_moved = c2_moved || !same_bits(mtl::dot<fma_acc<float>, float>(x, y), c2);
    }
    CHECK(c1_moved);
    CHECK(c2_moved);
}

TEST_CASE("[586:determinism] configuration 3 is independent of the thread count",
          "[586:determinism][accumulator][mt]") {
    // Threaded, dot partitions the terms into chunks and merges the partials.
    // For configuration 3 the merge is exact, so the threaded result is the
    // serial one at any MTL5_NUM_THREADS. Configurations 1 and 2 merge with a
    // rounding add, and so regroup the sum.
    const unsigned n_threads = mtl::detail::thread_pool::instance().size();
    REQUIRE(n_threads >= 1);
    REQUIRE(n_threads <= 4);

    // Long enough for configurations 1 and 2 to split too: 65536 terms a chunk.
    constexpr std::size_t n = 300000;
    rng r(8);
    vec<float> x(n), y(n);
    for (std::size_t i = 0; i < n; ++i) {
        x(i) = static_cast<float>(std::ldexp(r.symmetric(), r.integer(-20, 20)));
        y(i) = static_cast<float>(r.symmetric());
    }
    const auto exact = exact_dot<float>(x, y, n);
    REQUIRE(same_bits(mtl::dot<exact_acc<float>, float>(x, y), round_out(exact)));

    float s1 = 0.0f;
    fma_acc<float> s2;
    for (std::size_t i = 0; i < n; ++i) {
        s1 += x(i) * y(i);
        s2.sum = std::fma(x(i), y(i), s2.sum);
    }
    const float t1 = mtl::dot<float, float>(x, y);
    const float t2 = mtl::dot<fma_acc<float>, float>(x, y);
    if (pool_is_threaded()) {
        CHECK(!same_bits(t1, s1));
        CHECK(!same_bits(t2, s2.sum));
    } else {
        REQUIRE(same_bits(t1, s1));
        REQUIRE(same_bits(t2, s2.sum));
    }
}
