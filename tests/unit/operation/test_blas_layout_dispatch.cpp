// trsm, trmm, symm, symv, syrk and syr2k call BLAS for row-major operands
// (#724), by passing the column-major transpose: a row-major m x n buffer IS
// the column-major n x m matrix. Before #724 each gated its BLAS call on every
// operand being column-major, so the default (row-major) dense2D never reached
// BLAS at all.
//
// Two kinds of test:
//   - the *_dispatches_blas_v traits say which calls reach BLAS. In a BLAS
//     build every orientation qualifies (symm needs B and C to share one,
//     syr2k A and B), and nothing does under a non-default Accumulator. In a
//     build without BLAS nothing qualifies. A regression to the old
//     column-major-only gate fails these at compile time.
//   - every orientation combination, in f64 and f32, on non-square shapes (so a
//     swapped dimension or leading dimension is wrong, not merely transposed),
//     against a long-double reference. The triangle that must not be read is
//     filled with NaN, so a flipped uplo poisons the result.
//
// Without BLAS the same cases exercise the generic paths, which must agree.
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <limits>
#include <tuple>

#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/parameter.hpp>
#include <mtl/operation/symm.hpp>
#include <mtl/operation/symv.hpp>
#include <mtl/operation/syr2k.hpp>
#include <mtl/operation/syrk.hpp>
#include <mtl/operation/trmm.hpp>
#include <mtl/operation/trsm.hpp>
#include <mtl/tag/orientation.hpp>
#include <mtl/vec/dense_vector.hpp>

using namespace mtl;

namespace {

template <typename T> using rm = mat::dense2D<T, mat::parameters<tag::row_major>>;
template <typename T> using cm = mat::dense2D<T, mat::parameters<tag::col_major>>;

// -- the dispatch traits -------------------------------------------------------

#ifdef MTL5_HAS_BLAS
constexpr bool blas = true;
#else
constexpr bool blas = false;
#endif

static_assert(detail::trsm_dispatches_blas_v<void, rm<double>, rm<double>> == blas);
static_assert(detail::trsm_dispatches_blas_v<void, cm<double>, rm<double>> == blas);
static_assert(detail::trsm_dispatches_blas_v<void, rm<float>, cm<float>> == blas);
static_assert(detail::trsm_dispatches_blas_v<void, cm<double>, cm<double>> == blas);
static_assert(!detail::trsm_dispatches_blas_v<double, rm<double>, rm<double>>);
static_assert(!detail::trsm_dispatches_blas_v<void, rm<float>, rm<double>>);

static_assert(detail::trmm_dispatches_blas_v<void, rm<double>, rm<double>> == blas);
static_assert(detail::trmm_dispatches_blas_v<void, rm<double>, cm<double>> == blas);
static_assert(!detail::trmm_dispatches_blas_v<double, rm<double>, rm<double>>);

static_assert(detail::symm_dispatches_blas_v<void, rm<double>, rm<double>, rm<double>> == blas);
static_assert(detail::symm_dispatches_blas_v<void, cm<double>, rm<double>, rm<double>> == blas);
static_assert(detail::symm_dispatches_blas_v<void, rm<float>, cm<float>, cm<float>> == blas);
static_assert(!detail::symm_dispatches_blas_v<void, rm<double>, rm<double>, cm<double>>);  // B, C differ
static_assert(!detail::symm_dispatches_blas_v<double, rm<double>, rm<double>, rm<double>>);

static_assert(detail::symv_dispatches_blas_v<void, rm<double>, vec::dense_vector<double>,
                                             vec::dense_vector<double>> == blas);
static_assert(!detail::symv_dispatches_blas_v<double, rm<double>, vec::dense_vector<double>,
                                              vec::dense_vector<double>>);

static_assert(detail::syrk_dispatches_blas_v<void, rm<double>, rm<double>> == blas);
static_assert(detail::syrk_dispatches_blas_v<void, rm<double>, cm<double>> == blas);
static_assert(detail::syrk_dispatches_blas_v<void, cm<float>, rm<float>> == blas);
static_assert(!detail::syrk_dispatches_blas_v<double, rm<double>, rm<double>>);

static_assert(detail::syr2k_dispatches_blas_v<void, rm<double>, rm<double>, rm<double>> == blas);
static_assert(detail::syr2k_dispatches_blas_v<void, cm<double>, cm<double>, rm<double>> == blas);
static_assert(!detail::syr2k_dispatches_blas_v<void, rm<double>, cm<double>, rm<double>>);  // A, B differ
static_assert(!detail::syr2k_dispatches_blas_v<double, rm<double>, rm<double>, rm<double>>);

// -- inputs and references -----------------------------------------------------

/// Deterministic values in [-1, 1], distinct per (seed, i, j).
long double val(std::size_t seed, std::size_t i, std::size_t j) {
    std::size_t h = seed * 0x9E3779B97F4A7C15ull + i * 0xBF58476D1CE4E5B9ull + j * 0x94D049BB133111EBull;
    h ^= h >> 31;
    h *= 0xD6E8FEB86659FD93ull;
    h ^= h >> 32;
    return static_cast<long double>(h % 2001) / 1000.0L - 1.0L;
}

template <typename M>
M filled(std::size_t r, std::size_t c, std::size_t seed) {
    M A(r, c);
    for (std::size_t i = 0; i < r; ++i)
        for (std::size_t j = 0; j < c; ++j) A(i, j) = static_cast<typename M::value_type>(val(seed, i, j));
    return A;
}

/// Triangular, well conditioned: diagonal in [2, 3], NaN in the other triangle.
template <typename M>
M triangular(std::size_t m, bool upper, std::size_t seed) {
    using T = typename M::value_type;
    M A(m, m);
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < m; ++j) {
            const bool stored = upper ? j >= i : j <= i;
            A(i, j) = !stored ? std::numeric_limits<T>::quiet_NaN()
                    : i == j  ? static_cast<T>(2.5L + val(seed, i, j) / 2)
                              : static_cast<T>(val(seed, i, j) / static_cast<long double>(m));
        }
    return A;
}

/// Symmetric data in the lower triangle, NaN in the strict upper one.
template <typename M>
M lower_symmetric(std::size_t m, std::size_t seed) {
    using T = typename M::value_type;
    M A(m, m);
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < m; ++j)
            A(i, j) = j <= i ? static_cast<T>(val(seed, i, j)) : std::numeric_limits<T>::quiet_NaN();
    return A;
}

template <typename T> long double tol() { return 64 * static_cast<long double>(std::numeric_limits<T>::epsilon()); }

template <typename M>
long double at(const M& A, std::size_t i, std::size_t j) { return static_cast<long double>(A(i, j)); }

/// A's lower triangle read as a symmetric matrix.
template <typename M>
long double sym(const M& A, std::size_t i, std::size_t j) { return j <= i ? at(A, i, j) : at(A, j, i); }

template <typename T, typename M, typename Ref>
void require_close(const M& C, std::size_t r, std::size_t c, Ref ref) {
    for (std::size_t i = 0; i < r; ++i)
        for (std::size_t j = 0; j < c; ++j) {
            const long double want = ref(i, j);
            INFO("(" << i << ", " << j << ") got " << static_cast<double>(at(C, i, j))
                     << ", want " << static_cast<double>(want));
            REQUIRE(std::fabs(at(C, i, j) - want) <= tol<T>() * (1 + std::fabs(want)));
        }
}

// Every orientation combination of the operands, in f64 and f32.
template <typename T, typename OA, typename OB, typename OC = OA>
struct combo { using value = T; using A = mat::dense2D<T, mat::parameters<OA>>;
               using B = mat::dense2D<T, mat::parameters<OB>>; using C = mat::dense2D<T, mat::parameters<OC>>; };
using R = tag::row_major;
using K = tag::col_major;

} // namespace

// Catch's TEMPLATE_TEST_CASE splits its type list on commas, so the combos are
// named: d_RK = f64 with A row-major and B column-major, and so on.
namespace {
using d_RR = combo<double, R, R>;
using d_RRR = combo<double, R, R, R>;
using d_RRK = combo<double, R, R, K>;
using d_RK = combo<double, R, K>;
using d_RKR = combo<double, R, K, R>;
using d_RKK = combo<double, R, K, K>;
using d_KR = combo<double, K, R>;
using d_KRR = combo<double, K, R, R>;
using d_KRK = combo<double, K, R, K>;
using d_KK = combo<double, K, K>;
using d_KKR = combo<double, K, K, R>;
using d_KKK = combo<double, K, K, K>;
using f_RR = combo<float, R, R>;
using f_RRR = combo<float, R, R, R>;
using f_RRK = combo<float, R, R, K>;
using f_RK = combo<float, R, K>;
using f_RKR = combo<float, R, K, R>;
using f_RKK = combo<float, R, K, K>;
using f_KR = combo<float, K, R>;
using f_KRR = combo<float, K, R, R>;
using f_KRK = combo<float, K, R, K>;
using f_KK = combo<float, K, K>;
using f_KKR = combo<float, K, K, R>;
using f_KKK = combo<float, K, K, K>;
} // namespace

TEMPLATE_TEST_CASE("trsm and trmm: every orientation of A and B", "[operation][blas][724]",
                   d_RR, d_RK, d_KR, d_KK, f_RR, f_RK, f_KR, f_KK) {
    using T = typename TestType::value;
    using MA = typename TestType::A;
    using MB = typename TestType::B;
    const std::size_t m = 7, n = 4;   // B is m x n, m != n
    const T alpha = T(0.75);
    for (const bool upper : {false, true})
        for (const bool unit : {false, true}) {
            CAPTURE(upper, unit);
            const MA A = triangular<MA>(m, upper, 1);
            const MB B0 = filled<MB>(m, n, 2);
            const auto a = [&](std::size_t i, std::size_t k) -> long double {
                if (i == k) return unit ? 1.0L : at(A, i, i);
                return (upper ? k > i : k < i) ? at(A, i, k) : 0.0L;
            };

            MB B = B0;
            trmm(alpha, A, B, upper, unit);
            require_close<T>(B, m, n, [&](std::size_t i, std::size_t j) {
                long double s = 0;
                for (std::size_t k = 0; k < m; ++k) s += a(i, k) * at(B0, k, j);
                return static_cast<long double>(alpha) * s;
            });

            // trsm: A X = alpha B0, checked by its residual A X - alpha B0.
            MB X = B0;
            trsm(alpha, A, X, upper, unit);
            require_close<T>(X, m, n, [&](std::size_t i, std::size_t j) {
                long double r = static_cast<long double>(alpha) * at(B0, i, j);
                for (std::size_t k = 0; k < m; ++k) if (k != i) r -= a(i, k) * at(X, k, j);
                return r / a(i, i);
            });
        }
}

TEMPLATE_TEST_CASE("symm: every orientation of A, B and C", "[operation][blas][724]",
                   d_RRR, d_RRK, d_RKR, d_RKK, d_KRR, d_KRK, d_KKR, d_KKK,
                   f_RRR, f_RRK, f_RKR, f_RKK, f_KRR, f_KRK, f_KKR, f_KKK) {
    using T = typename TestType::value;
    const std::size_t m = 6, n = 5;
    const auto A = lower_symmetric<typename TestType::A>(m, 3);
    const auto B = filled<typename TestType::B>(m, n, 4);
    const auto C0 = filled<typename TestType::C>(m, n, 5);
    auto C = C0;
    symm(T(1.5), A, B, T(-0.5), C);
    require_close<T>(C, m, n, [&](std::size_t i, std::size_t j) {
        long double s = 0;
        for (std::size_t k = 0; k < m; ++k) s += sym(A, i, k) * at(B, k, j);
        return 1.5L * s - 0.5L * at(C0, i, j);
    });
}

TEMPLATE_TEST_CASE("syrk and syr2k: every orientation", "[operation][blas][724]",
                   d_RRR, d_RRK, d_RKR, d_RKK, d_KRR, d_KRK, d_KKR, d_KKK,
                   f_RRR, f_RRK, f_RKR, f_RKK, f_KRR, f_KRK, f_KKR, f_KKK) {
    using T = typename TestType::value;
    using MC = typename TestType::C;
    const std::size_t m = 6, k = 9;   // A, B are m x k
    const auto A = filled<typename TestType::A>(m, k, 6);
    const auto B = filled<typename TestType::B>(m, k, 7);
    // beta != 0: C must be symmetric on input (only one triangle is read).
    MC C0(m, m);
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < m; ++j) C0(i, j) = static_cast<T>(val(8, std::max(i, j), std::min(i, j)));

    MC C = C0;
    syrk(T(0.5), A, T(2), C);
    require_close<T>(C, m, m, [&](std::size_t i, std::size_t j) {
        long double s = 0;
        for (std::size_t l = 0; l < k; ++l) s += at(A, i, l) * at(A, j, l);
        return 0.5L * s + 2.0L * at(C0, i, j);
    });

    C = C0;
    syr2k(T(0.5), A, B, T(2), C);
    require_close<T>(C, m, m, [&](std::size_t i, std::size_t j) {
        long double s = 0;
        for (std::size_t l = 0; l < k; ++l) s += at(A, i, l) * at(B, j, l) + at(B, i, l) * at(A, j, l);
        return 0.5L * s + 2.0L * at(C0, i, j);
    });
}

TEMPLATE_TEST_CASE("symv: either orientation of A", "[operation][blas][724]",
                   d_RR, d_KK, f_RR, f_KK) {
    using T = typename TestType::value;
    const std::size_t n = 9;
    const auto A = lower_symmetric<typename TestType::A>(n, 9);
    vec::dense_vector<T> x(n), y0(n);
    for (std::size_t i = 0; i < n; ++i) { x(i) = static_cast<T>(val(10, i, 0)); y0(i) = static_cast<T>(val(11, i, 0)); }
    auto y = y0;
    symv(T(1.25), A, x, T(0.5), y);
    for (std::size_t i = 0; i < n; ++i) {
        long double s = 0;
        for (std::size_t j = 0; j < n; ++j) s += sym(A, i, j) * static_cast<long double>(x(j));
        const long double want = 1.25L * s + 0.5L * static_cast<long double>(y0(i));
        INFO("y(" << i << ")");
        REQUIRE(std::fabs(static_cast<long double>(y(i)) - want) <= tol<T>() * (1 + std::fabs(want)));
    }
}
