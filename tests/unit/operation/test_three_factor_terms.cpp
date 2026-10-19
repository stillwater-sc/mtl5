// MTL5 -- Exact three-factor terms in the LDL^T accumulator paths (#602).
//
// The LDL^T-type factorizations sum terms L(i,k) d_k L(j,k), and the
// accumulator takes two factors, so one product is formed in the element type
// first. math::two_product splits that product into hi + lo exactly, and the
// kernels feed lo in as a second product when math::splits_product_terms_v
// says the accumulator keeps it. Checked here:
//
//   gate      -- which (Acc, Value) pairs split;
//   split     -- hi + lo is the product exactly, and an overflow stays inf;
//   agreement -- with an exact accumulator every term reaches it whole, so the
//                four implementations (dense ldlt, ILDL on a dense pattern,
//                sparse and supernodal LDL^T in natural order) round the SAME
//                exact sums and give the same bits. Before #602 they did not:
//                ILDL pre-rounded L(i,k)^2 where the others pre-round L d;
//   accuracy  -- [586:direct] / [586:pc]: over an ensemble of perturbed float
//                Hilbert matrices the split exact accumulator beats the same
//                accumulator fed pre-rounded terms, and both beat plain / fma.
//
// Measured (geometric mean of ||A x - 1||_inf over 300 draws per n, float,
// ILDL; dense LDL^T agrees -- plain / fma / exact pre-rounded / exact split):
//
//    n =  6:  5.4e-7 / 4.2e-7 / 3.2e-7 / 2.8e-7
//    n =  8:  7.1e-7 / 5.8e-7 / 4.1e-7 / 3.5e-7
//    n = 12:  1.4e-6 / 1.1e-6 / 7.0e-7 / 5.4e-7
//    n = 20:  2.9e-6 / 2.3e-6 / 1.4e-6 / 9.1e-7
//
// So exact accumulation already orders below plain and fma with pre-rounded
// terms; the split adds 1.1-1.6x on top, growing with n. The compensated
// accumulator (Dot2, #611) takes the split as well, and lands on the exact
// split's mean to two digits: 5.3e-7 at n = 12 and 8.6e-7 at n = 20 over 100
// draws, against 5.3e-7 and 8.6e-7 for exact. Bunch-Kaufman is not
// affected: it feeds the stored Schur column W, not a product formed for the
// accumulator.
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstring>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>

#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/math/accumulator_traits.hpp>
#include <mtl/operation/ldlt.hpp>
#include <mtl/itl/pc/ildl.hpp>
#include <mtl/sparse/factorization/sparse_ldlt.hpp>
#include <mtl/sparse/factorization/supernodal_ldlt.hpp>

#include "exact_accumulator.hpp"

using namespace mtl;

namespace unsplit {

/// The exact accumulator behind a direct accumulator_traits specialization
/// instead of the ADL hook, which splits_product_terms_v does not detect: the
/// kernels feed it the pre-rounded terms, as before #602.
struct exact_pre { exact_test::exact_acc<float> e; };

} // namespace unsplit

template <>
struct mtl::math::accumulator_traits<unsplit::exact_pre, float> {
    using A = unsplit::exact_pre;
    static void clear(A& a) { accumulator_clear(a.e); }
    static void assign(A& a, const float& v) { accumulator_assign(a.e, v); }
    template <typename R = float>
    static R value(const A& a) { return accumulator_value(a.e, std::type_identity<R>{}); }
    static void add_product(A& a, const float& m, const float& v) { accumulator_add_product(a.e, m, v); }
};

namespace {

using F     = float;
using exact = exact_test::exact_acc<F>;
using fma_acc = math::fma_accumulator<F>;

static_assert(math::splits_product_terms_v<double, float>);
static_assert(math::splits_product_terms_v<exact, float>);
static_assert(math::splits_product_terms_v<exact_test::exact_acc<double>, double>);
static_assert(!math::splits_product_terms_v<float, float>);
static_assert(!math::splits_product_terms_v<double, double>);
static_assert(!math::splits_product_terms_v<fma_acc, float>);
static_assert(math::splits_product_terms_v<math::compensated_accumulator<float>, float>);
static_assert(math::splits_product_terms_v<math::compensated_accumulator<double>, float>);
static_assert(!math::splits_product_terms_v<math::compensated_accumulator<float>, double>);
static_assert(!math::splits_product_terms_v<math::fma_accumulator<double>, float>);
static_assert(!math::splits_product_terms_v<unsplit::exact_pre, float>);
static_assert(!math::splits_product_terms_v<std::complex<double>, std::complex<float>>);
static_assert(math::splits_product_terms_v<long double, double> ==
              (std::numeric_limits<long double>::digits > std::numeric_limits<double>::digits));

/// A perturbed Hilbert matrix: symmetric positive definite, ill-conditioned,
/// dense, so every reduction is long. eps = 0 is Hilbert itself.
std::vector<F> perturbed_hilbert(std::size_t n, std::mt19937& gen, double eps) {
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::vector<F> a(n * n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j <= i; ++j)
            a[i * n + j] = a[j * n + i] = F((1.0 / double(i + j + 1)) * (1.0 + eps * u(gen)));
    return a;
}

mat::dense2D<F> dense(const std::vector<F>& a, std::size_t n) {
    mat::dense2D<F> A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) A(i, j) = a[i * n + j];
    return A;
}

mat::compressed2D<F> to_sparse(const std::vector<F>& a, std::size_t n) {
    mat::compressed2D<F> A(n, n);
    mat::inserter<mat::compressed2D<F>> ins(A);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) ins[i][j] << a[i * n + j];
    return A;
}

/// ||A x - b||_inf, b = 1, in double; NaN propagates.
double residual(const std::vector<F>& a, std::size_t n, const vec::dense_vector<F>& x) {
    double worst = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double r = -1.0;
        for (std::size_t j = 0; j < n; ++j) r += double(a[i * n + j]) * double(x(j));
        if (std::isnan(r)) return r;
        worst = std::max(worst, std::fabs(r));
    }
    return worst;
}

template <typename Acc>
vec::dense_vector<F> dense_ldlt_solve(const std::vector<F>& a, std::size_t n) {
    auto A = dense(a, n);
    vec::dense_vector<F> x(n), b(n, F(1));
    REQUIRE(ldlt_factor<Acc>(A) == 0);
    ldlt_solve<Acc>(A, x, b);
    return x;
}

template <typename Acc>
vec::dense_vector<F> ildl_solve(const std::vector<F>& a, std::size_t n) {
    itl::pc::ildl<F, mat::parameters<>, Acc> M(to_sparse(a, n));
    vec::dense_vector<F> x(n), b(n, F(1));
    M.solve(x, b);
    return x;
}

bool same_bits(const vec::dense_vector<F>& x, const vec::dense_vector<F>& y) {
    if (x.size() != y.size()) return false;
    for (std::size_t i = 0; i < x.size(); ++i)
        if (std::memcmp(&x(i), &y(i), sizeof(F)) != 0) return false;
    return true;
}

} // namespace

TEST_CASE("two_product splits a product exactly (#602)", "[accumulator][602]") {
    std::mt19937 gen(1);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    for (int t = 0; t < 1000; ++t) {
        // One draw per statement, exponent first: a call's arguments are evaluated in
        // an unspecified order, and GCC's (the last argument first) is the one the
        // recorded numbers were measured with.
        const int ea = int(u(gen) * 20);
        const F a = F(std::ldexp(u(gen), ea));
        const int eb = int(u(gen) * 20);
        const F b = F(std::ldexp(u(gen), eb));
        const auto s = math::two_product(a, b);
        CHECK(s.hi == a * b);
        CHECK(double(s.hi) + double(s.lo) == double(a) * double(b));   // exact in double
    }
    const auto big = math::two_product(std::numeric_limits<F>::max(), F(2));
    CHECK(std::isinf(big.hi));
    CHECK(big.lo == F(0));
}

TEST_CASE("with an exact accumulator the four LDL^T implementations agree bit for bit (#602)",
          "[accumulator][602][ldlt]") {
    std::mt19937 gen(7);
    int pre_rounded_disagree = 0;
    for (std::size_t n : {5u, 8u, 12u}) {
        for (double eps : {0.0, 0.05}) {
            const auto a = perturbed_hilbert(n, gen, eps);
            INFO("n = " << n << ", eps = " << eps);
            const auto xd = dense_ldlt_solve<exact>(a, n);
            const auto xi = ildl_solve<exact>(a, n);
            vec::dense_vector<F> xs(n), xn(n), b(n, F(1));
            sparse::factorization::sparse_ldlt_solve<exact>(to_sparse(a, n), xs, b);
            sparse::factorization::supernodal_ldlt_solve<exact>(to_sparse(a, n), xn, b);
            CHECK(same_bits(xd, xi));
            CHECK(same_bits(xd, xs));
            CHECK(same_bits(xd, xn));
            pre_rounded_disagree += !same_bits(dense_ldlt_solve<unsplit::exact_pre>(a, n),
                                               ildl_solve<unsplit::exact_pre>(a, n));
        }
    }
    // The same exact accumulator fed pre-rounded terms: ILDL rounds L(i,k)^2
    // where dense LDL^T rounds L(j,k) D(k), so the bits part.
    CHECK(pre_rounded_disagree > 0);
}

TEST_CASE("[586:direct] [586:pc] the split exact accumulator beats pre-rounded terms",
          "[586:direct][586:pc][accumulator][602][ldlt]") {
    for (std::size_t n : {12u, 20u}) {
        std::mt19937 gen(static_cast<std::uint32_t>(n));
        double lp = 0, lf = 0, le = 0, ls = 0, ldw = 0, lc = 0;
        int count = 0;
        for (int t = 0; t < 100; ++t) {
            const auto a = perturbed_hilbert(n, gen, 0.05);
            const double r[6] = {
                residual(a, n, ildl_solve<F>(a, n)),
                residual(a, n, ildl_solve<fma_acc>(a, n)),
                residual(a, n, ildl_solve<unsplit::exact_pre>(a, n)),
                residual(a, n, ildl_solve<exact>(a, n)),
                residual(a, n, dense_ldlt_solve<exact>(a, n)),
                residual(a, n, ildl_solve<math::compensated_accumulator<F>>(a, n)),
            };
            bool finite = true;
            for (double v : r) finite = finite && std::isfinite(v) && v > 0.0;
            if (!finite) continue;
            ++count;
            lp += std::log(r[0]); lf += std::log(r[1]); le += std::log(r[2]);
            ls += std::log(r[3]); ldw += std::log(r[4]); lc += std::log(r[5]);
        }
        REQUIRE(count > 50);
        const double plain = std::exp(lp / count), fm = std::exp(lf / count);
        const double pre = std::exp(le / count), split = std::exp(ls / count);
        const double comp = std::exp(lc / count);
        INFO("n = " << n << " (" << count << " draws): plain " << plain << ", fma " << fm
             << ", exact pre-rounded " << pre << ", exact split " << split << ", compensated " << comp);
        CHECK(split < pre);
        CHECK(pre < plain);
        CHECK(pre < fm);
        CHECK(std::exp(ldw / count) == split);   // dense LDL^T: the same bits, so the same mean
        // The compensated accumulator (#611) takes the split too, and its twice-
        // working precision is enough for reductions this short: within 5% of
        // the exact accumulator.
        CHECK(comp < pre);
        CHECK(comp <= 1.05 * split);
        CHECK(comp >= split / 1.05);
    }
}

TEST_CASE("a wider built-in accumulator takes the split too (#602)", "[accumulator][602][ldlt]") {
    // double over float: hi * c and lo * c are both exact in double, so the
    // accumulated sum differs from the pre-rounded one, and the solve still holds.
    std::mt19937 gen(3);
    const std::size_t n = 10;
    const auto a = perturbed_hilbert(n, gen, 0.05);
    CHECK(residual(a, n, dense_ldlt_solve<double>(a, n)) < 1e-4);
    CHECK(residual(a, n, ildl_solve<double>(a, n)) < 1e-4);
    vec::dense_vector<F> xs(n), xn(n), b(n, F(1));
    sparse::factorization::sparse_ldlt_solve<double>(to_sparse(a, n), xs, b);
    sparse::factorization::supernodal_ldlt_solve<double>(to_sparse(a, n), xn, b);
    CHECK(residual(a, n, xs) < 1e-4);
    CHECK(residual(a, n, xn) < 1e-4);
}
