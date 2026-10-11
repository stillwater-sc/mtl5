// MTL5 -- the Householder accumulator reaches its callers (#536).
//
// householder() and apply_householder_left/right() have taken an Accumulator
// since #537, but nothing forwarded one: qr_factor, lq_factor, the Hessenberg
// reduction and the eigensolvers all called them with the default. Each caller
// now takes a leading `Accumulator = void` and forwards it.
//
// A forwarding bug is silent -- the default still gives a right answer -- so the
// tests count the accumulator's add_product calls and compare them with the
// exact number each routine's sums contain. A caller that drops the accumulator
// anywhere comes up short. Because a non-void accumulator also skips the LAPACK
// dispatch, the counts hold in MTL5_HAS_LAPACK builds too: a LAPACK call would
// make no add_product calls at all.
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <random>
#include <type_traits>
#include <vector>

#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/math/accumulator_traits.hpp>
#include <mtl/operation/qr.hpp>
#include <mtl/operation/lq.hpp>
#include <mtl/operation/hessenberg.hpp>
#include <mtl/operation/eigenvalue_symmetric.hpp>
#include <mtl/operation/eigenvalue.hpp>

using namespace mtl;

namespace {

struct counting_acc {
    double v = 0.0;
    static inline std::size_t products = 0;
};

/// The complex counterpart. A complex T's reflector uses one accumulator type
/// for two value types: the real sum of squares |x_i|^2 and the complex
/// v^H*A(:,j), so this one holds a complex sum and rounds out either.
struct counting_cacc {
    std::complex<double> v{};
    static inline std::size_t products = 0;
};

} // namespace

namespace mtl::math {
template <typename Value>
struct accumulator_traits<::counting_acc, Value> {
    static void clear(::counting_acc& a) { a.v = 0.0; }
    static void assign(::counting_acc& a, const Value& x) { a.v = static_cast<double>(x); }
    template <typename Result = Value>
    static Result value(const ::counting_acc& a) { return static_cast<Result>(a.v); }
    static void add_product(::counting_acc& a, const Value& m, const Value& x) {
        a.v += static_cast<double>(m) * static_cast<double>(x);
        ++::counting_acc::products;
    }
};
template <typename Value>
struct accumulator_traits<::counting_cacc, Value> {
    static void clear(::counting_cacc& a) { a.v = 0.0; }
    static void assign(::counting_cacc& a, const Value& x) { a.v = std::complex<double>(x); }
    template <typename Result = Value>
    static Result value(const ::counting_cacc& a) {
        if constexpr (std::is_same_v<Result, std::complex<double>>) return a.v;
        else return static_cast<Result>(a.v.real());
    }
    static void add_product(::counting_cacc& a, const Value& m, const Value& x) {
        a.v += std::complex<double>(m) * std::complex<double>(x);
        ++::counting_cacc::products;
    }
};
} // namespace mtl::math

namespace {

mat::dense2D<double> random_matrix(std::size_t m, std::size_t n, std::uint64_t seed,
                                   bool symmetric = false) {
    std::mt19937_64 g(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    mat::dense2D<double> A(m, n);
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A(i, j) = u(g);
    if (symmetric)
        for (std::size_t i = 0; i < m; ++i)
            for (std::size_t j = 0; j < i; ++j)
                A(j, i) = A(i, j);
    return A;
}

// Exact add_product counts, for a dense random input where no reflector is the
// identity. householder(x) sums the squares of x(1:), i.e. len - 1 terms; the
// left update forms one len-term dot per column, the right update one per row.

/// qr_factor on a tall m x n (m > n, so every column has len >= 2).
std::size_t qr_factor_products(std::size_t m, std::size_t n) {
    std::size_t c = 0;
    for (std::size_t j = 0; j < n; ++j) {
        const std::size_t len = m - j;
        c += (len - 1) + (n - j) * len;
    }
    return c;
}

/// qr_extract_Q: every reflector applied to all m columns of Q.
std::size_t qr_extract_Q_products(std::size_t m, std::size_t n) {
    std::size_t c = 0;
    for (std::size_t j = 0; j < n; ++j) c += m * (m - j);
    return c;
}

/// qr_solve: one v^H*y per reflector, then sum_{j>i} R(i,j)*x(j) per row.
std::size_t qr_solve_products(std::size_t m, std::size_t n) {
    std::size_t c = 0;
    for (std::size_t j = 0; j < n; ++j) c += m - j;
    return c + n * (n - 1) / 2;
}

/// lq_factor on a wide m x n (n > m).
std::size_t lq_factor_products(std::size_t m, std::size_t n) {
    std::size_t c = 0;
    for (std::size_t i = 0; i < m; ++i) {
        const std::size_t len = n - i;
        c += (len - 1) + (m - i) * len;
    }
    return c;
}

/// lq_extract_Q: every reflector applied to all n columns of Q.
std::size_t lq_extract_Q_products(std::size_t m, std::size_t n) {
    std::size_t c = 0;
    for (std::size_t i = 0; i < m; ++i) c += n * (n - i);
    return c;
}

/// hessenberg_factor on n x n: n-2 reflectors, each applied on the left to
/// columns j.. and on the right to all n rows.
std::size_t hessenberg_products(std::size_t n) {
    std::size_t c = 0;
    for (std::size_t j = 0; j + 2 < n; ++j) {
        const std::size_t len = n - j - 1;
        c += (len - 1) + (n - j) * len + n * len;
    }
    return c;
}

/// eigen_symmetric additionally right-multiplies Q by every reflector.
std::size_t eigen_symmetric_products(std::size_t n) {
    std::size_t c = hessenberg_products(n);
    for (std::size_t j = 0; j + 2 < n; ++j) c += n * (n - j - 1);
    return c;
}

} // namespace

TEST_CASE("qr_factor, qr_extract_Q and qr_solve forward the accumulator",
          "[operation][qr][accumulator][536]") {
    const std::size_t m = 9, n = 5;
    auto A = random_matrix(m, n, 5361);
    vec::dense_vector<double> tau;

    counting_acc::products = 0;
    REQUIRE(qr_factor<counting_acc>(A, tau) == 0);
    REQUIRE(counting_acc::products == qr_factor_products(m, n));

    counting_acc::products = 0;
    auto Q = qr_extract_Q<counting_acc>(A, tau);
    REQUIRE(counting_acc::products == qr_extract_Q_products(m, n));

    vec::dense_vector<double> b(m, 1.0), x(n);
    counting_acc::products = 0;
    qr_solve<counting_acc>(A, tau, x, b);
    REQUIRE(counting_acc::products == qr_solve_products(m, n));

    // The default path makes no accumulator calls at all.
    auto B = random_matrix(m, n, 5361);
    counting_acc::products = 0;
    qr_factor(B, tau);
    qr_solve(B, tau, x, b);
    REQUIRE(counting_acc::products == 0);
}

TEST_CASE("lq_factor and lq_extract_Q forward the accumulator",
          "[operation][lq][accumulator][536]") {
    const std::size_t m = 5, n = 9;
    auto A = random_matrix(m, n, 5362);
    vec::dense_vector<double> tau;

    counting_acc::products = 0;
    REQUIRE(lq_factor<counting_acc>(A, tau) == 0);
    REQUIRE(counting_acc::products == lq_factor_products(m, n));

    counting_acc::products = 0;
    auto Q = lq_extract_Q<counting_acc>(A, tau);
    REQUIRE(counting_acc::products == lq_extract_Q_products(m, n));
}

TEST_CASE("hessenberg_factor, hessenberg and tridiagonalize forward the accumulator",
          "[operation][hessenberg][accumulator][536]") {
    const std::size_t n = 8;
    vec::dense_vector<double> tau;

    auto A = random_matrix(n, n, 5363);
    counting_acc::products = 0;
    hessenberg_factor<counting_acc>(A, tau);
    REQUIRE(counting_acc::products == hessenberg_products(n));

    counting_acc::products = 0;
    auto H = hessenberg<counting_acc>(random_matrix(n, n, 5363));
    REQUIRE(counting_acc::products == hessenberg_products(n));

    auto S = random_matrix(n, n, 5364, /*symmetric=*/true);
    counting_acc::products = 0;
    tridiagonalize<counting_acc>(S, tau);
    REQUIRE(counting_acc::products == hessenberg_products(n));
}

TEST_CASE("the eigensolvers forward the accumulator and skip LAPACK",
          "[operation][eigenvalue][accumulator][536]") {
    const std::size_t n = 8;
    auto S = random_matrix(n, n, 5365, /*symmetric=*/true);
    auto A = random_matrix(n, n, 5366);

    counting_acc::products = 0;
    auto e1 = eigenvalue_symmetric_generic<counting_acc>(S);
    REQUIRE(counting_acc::products == hessenberg_products(n));

    // Would be 0 if this dispatched to syev in a LAPACK build.
    counting_acc::products = 0;
    auto e2 = eigenvalue_symmetric<counting_acc>(S);
    REQUIRE(counting_acc::products == hessenberg_products(n));

    counting_acc::products = 0;
    auto r = eigen_symmetric<counting_acc>(S);
    REQUIRE(counting_acc::products == eigen_symmetric_products(n));

    counting_acc::products = 0;
    auto e3 = eigenvalue<counting_acc>(A);
    REQUIRE(counting_acc::products == hessenberg_products(n));

    counting_acc::products = 0;
    auto r2 = eigen<counting_acc>(A);
    REQUIRE(counting_acc::products == hessenberg_products(n));

    // The accumulated results are still eigenvalues: same spectrum as the
    // default path, to rounding.
    auto d1 = eigenvalue_symmetric_generic(S);
    for (std::size_t i = 0; i < n; ++i)
        REQUIRE(std::abs(e1(i) - d1(i)) < 1e-12);
}

TEST_CASE("eigenvalue_symmetric forwards the accumulator for a complex Hermitian matrix",
          "[operation][eigenvalue][accumulator][complex][536]") {
    // The Hermitian path has its own LAPACK branch (heev), separate from the
    // real one (syev). Both must be skipped for a non-void accumulator. Would
    // be 0 if this dispatched to heev in a LAPACK build.
    const std::size_t n = 8;
    std::mt19937_64 g(5368);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    mat::dense2D<std::complex<double>> H(n, n);
    for (std::size_t i = 0; i < n; ++i) {
        H(i, i) = u(g);
        for (std::size_t j = 0; j < i; ++j) {
            H(i, j) = std::complex<double>(u(g), u(g));
            H(j, i) = std::conj(H(i, j));
        }
    }
    counting_cacc::products = 0;
    auto e = eigenvalue_symmetric<counting_cacc>(H);
    REQUIRE(counting_cacc::products == hessenberg_products(n));

    auto d = eigenvalue_symmetric_generic(H);
    for (std::size_t i = 0; i < n; ++i)
        REQUIRE(std::abs(e(i) - d(i)) < 1e-12);
}

TEST_CASE("a double accumulator lowers the backward error of a float QR",
          "[operation][qr][accumulator][536]") {
    // float elements; every reflector sum accumulated in float, then in double.
    // Both are explicit accumulators, so both run the generic loop even in a
    // LAPACK build (the default would dispatch float to sgeqrf). Measured on
    // random 256 x 256 matrices (gcc 15, -O2): ||QR - A||_F/||A||_F is
    // 6.8e-7 to 7.0e-7 accumulating in float and 3.7e-7 in double, a factor of
    // 1.9 in each of 3 seeds. The bound below asks for 1.3.
    const std::size_t n = 256;
    std::mt19937_64 g(5367);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    mat::dense2D<float> A0(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A0(i, j) = static_cast<float>(u(g));

    auto backward_error = [&](auto tag) {
        using Acc = typename decltype(tag)::type;
        auto A = A0;
        vec::dense_vector<float> tau;
        qr_factor<Acc>(A, tau);
        auto Q = qr_extract_Q<Acc>(A, tau);
        auto R = qr_extract_R(A);
        double num = 0.0, den = 0.0;
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j) {
                double s = 0.0;
                for (std::size_t k = 0; k <= j; ++k)
                    s += static_cast<double>(Q(i, k)) * static_cast<double>(R(k, j));
                const double d = s - static_cast<double>(A0(i, j));
                num += d * d;
                den += static_cast<double>(A0(i, j)) * static_cast<double>(A0(i, j));
            }
        return std::sqrt(num / den);
    };
    const double narrow = backward_error(std::type_identity<float>{});
    const double wide   = backward_error(std::type_identity<double>{});
    INFO("||QR - A||/||A||: float accumulation " << narrow << ", double accumulator " << wide);
    REQUIRE(wide * 1.3 < narrow);
}
