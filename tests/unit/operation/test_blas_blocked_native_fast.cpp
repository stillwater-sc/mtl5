// trsm, trmm and symm on the native-fast build go through the blocked GEMM
// (#724, part 3). trsm and trmm walk A's diagonal in blocks of
// detail::triangular_block rows: the row loop does the diagonal block and
// gemm_blocked everything else. symm expands A to the full symmetric matrix
// and makes one GEMM call.
//
// The rounding is the blocked GEMM's, so these are checked against a
// long-double reference at a gamma_k-style tolerance, not bit for bit. Shapes
// straddle the block size (one block, one block plus one row, several blocks
// with a remainder) and B is never square, so a block-offset or leading-
// dimension slip lands on wrong rows rather than on a transposed copy.
//
// MTL5_NATIVE_FAST_GEMM is defined before the headers, as the other
// native-fast tests do, so this translation unit takes those paths.
#define MTL5_NATIVE_FAST_GEMM 1

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>

#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/parameter.hpp>
#include <mtl/operation/symm.hpp>
#include <mtl/operation/trmm.hpp>
#include <mtl/operation/trsm.hpp>
#include <mtl/tag/orientation.hpp>

using namespace mtl;

namespace {

template <typename T> using rm = mat::dense2D<T, mat::parameters<tag::row_major>>;
template <typename T> using cm = mat::dense2D<T, mat::parameters<tag::col_major>>;

static_assert(detail::trsm_uses_blocked_v<rm<double>, rm<double>>);
static_assert(detail::trsm_uses_blocked_v<cm<float>, rm<float>>);
static_assert(!detail::trsm_uses_blocked_v<rm<double>, cm<double>>);          // col-major B: not blocked
static_assert(!detail::trsm_uses_blocked_v<rm<long double>, rm<long double>>);
static_assert(detail::trmm_uses_blocked_v<rm<double>, rm<double>>);
static_assert(!detail::trmm_uses_blocked_v<rm<float>, rm<double>>);
static_assert(detail::symm_uses_blocked_v<rm<double>, cm<double>, rm<double>>);
static_assert(!detail::symm_uses_blocked_v<rm<double>, rm<double>, cm<double>>); // col-major C: not blocked

double val(std::size_t seed, std::size_t i, std::size_t j) {
    std::size_t h = seed * 0x9E3779B97F4A7C15ull + i * 0xBF58476D1CE4E5B9ull + j * 0x94D049BB133111EBull;
    h ^= h >> 29; h *= 0xD6E8FEB86659FD93ull; h ^= h >> 32;
    return static_cast<double>(h % 1000003) / 500001.5 - 1.0;
}

/// Triangular, well conditioned (diagonal in [2, 3], off-diagonal O(1/m)), NaN
/// in the triangle that must not be read.
template <typename M>
M triangular(std::size_t m, bool upper, std::size_t seed) {
    using T = typename M::value_type;
    M A(m, m);
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < m; ++j) {
            const bool stored = upper ? j >= i : j <= i;
            A(i, j) = !stored ? std::numeric_limits<T>::quiet_NaN()
                    : i == j  ? T(2.5 + val(seed, i, j) / 2)
                              : T(val(seed, i, j) / double(m));
        }
    return A;
}

template <typename M>
M filled(std::size_t r, std::size_t c, std::size_t seed) {
    M A(r, c);
    for (std::size_t i = 0; i < r; ++i)
        for (std::size_t j = 0; j < c; ++j) A(i, j) = typename M::value_type(val(seed, i, j));
    return A;
}

template <typename M>
long double at(const M& A, std::size_t i, std::size_t j) { return static_cast<long double>(A(i, j)); }

/// |got - want| <= 8 (k + 2) eps (|terms| + 1), the gamma_k shape.
template <typename T>
void check(long double got, long double want, long double absterms, std::size_t k) {
    const long double tol = 8.0L * static_cast<long double>(k + 2) *
                            static_cast<long double>(std::numeric_limits<T>::epsilon()) * (absterms + 1);
    REQUIRE(std::isfinite(static_cast<double>(got)));
    REQUIRE(std::fabs(got - want) <= tol);
}

} // namespace

TEMPLATE_TEST_CASE("blocked trsm and trmm match the reference across block boundaries", "[operation][724]",
                   (std::pair<rm<double>, rm<double>>), (std::pair<cm<double>, rm<double>>),
                   (std::pair<rm<float>, rm<float>>), (std::pair<cm<float>, rm<float>>)) {
    using MA = typename TestType::first_type;
    using MB = typename TestType::second_type;
    using T = typename MB::value_type;
    constexpr std::size_t nb = detail::triangular_block;
    for (const std::size_t m : {nb, nb + 1, 2 * nb + 22})
        for (const bool upper : {false, true})
            for (const bool unit : {false, true}) {
                const std::size_t n = 37;
                CAPTURE(m, upper, unit);
                const MA A = triangular<MA>(m, upper, 1);
                const MB B0 = filled<MB>(m, n, 2);
                const auto a = [&](std::size_t i, std::size_t k) -> long double {
                    if (i == k) return unit ? 1.0L : at(A, i, i);
                    return (upper ? k > i : k < i) ? at(A, i, k) : 0.0L;
                };
                const T alpha = T(-0.75);

                MB X = B0;
                trmm(alpha, A, X, upper, unit);
                for (std::size_t i = 0; i < m; ++i)
                    for (std::size_t j = 0; j < n; ++j) {
                        long double s = 0, abs = 0;
                        for (std::size_t k = 0; k < m; ++k) {
                            const long double t = a(i, k) * at(B0, k, j);
                            s += t; abs += std::fabs(t);
                        }
                        INFO("trmm (" << i << ", " << j << ")");
                        check<T>(at(X, i, j), static_cast<long double>(alpha) * s,
                                 std::fabs(static_cast<long double>(alpha)) * abs, m);
                    }

                // trsm: the residual A X - alpha B0, per |A| |X| + |alpha B0|.
                X = B0;
                trsm(alpha, A, X, upper, unit);
                for (std::size_t i = 0; i < m; ++i)
                    for (std::size_t j = 0; j < n; ++j) {
                        long double r = 0, abs = 0;
                        for (std::size_t k = 0; k < m; ++k) {
                            const long double t = a(i, k) * at(X, k, j);
                            r += t; abs += std::fabs(t);
                        }
                        const long double b = static_cast<long double>(alpha) * at(B0, i, j);
                        INFO("trsm (" << i << ", " << j << ")");
                        check<T>(r, b, abs + std::fabs(b), m);
                    }
            }
}

TEMPLATE_TEST_CASE("blocked symm matches the reference", "[operation][724]",
                   (std::pair<rm<double>, rm<double>>), (std::pair<rm<double>, cm<double>>),
                   (std::pair<cm<float>, rm<float>>), (std::pair<rm<float>, cm<float>>)) {
    using MA = typename TestType::first_type;
    using MB = typename TestType::second_type;
    using T = typename MA::value_type;
    using MC = rm<T>;
    const std::size_t m = 150, n = 41;
    MA A(m, m);   // data in the lower triangle, NaN above: only the lower may be read
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < m; ++j)
            A(i, j) = j <= i ? T(val(3, i, j)) : std::numeric_limits<T>::quiet_NaN();
    const MB B = filled<MB>(m, n, 4);
    const auto sym = [&](std::size_t i, std::size_t k) { return k <= i ? at(A, i, k) : at(A, k, i); };

    for (const T beta : {T(0), T(0.5)}) {
        CAPTURE(static_cast<double>(beta));
        MC C = filled<MC>(m, n, 5);
        const MC C0 = C;
        if (beta == T(0))   // beta == 0 must not read C
            for (std::size_t i = 0; i < m; ++i)
                for (std::size_t j = 0; j < n; ++j) C(i, j) = std::numeric_limits<T>::quiet_NaN();
        symm(T(1.5), A, B, beta, C);
        for (std::size_t i = 0; i < m; ++i)
            for (std::size_t j = 0; j < n; ++j) {
                long double s = 0, abs = 0;
                for (std::size_t k = 0; k < m; ++k) {
                    const long double t = sym(i, k) * at(B, k, j);
                    s += t; abs += std::fabs(t);
                }
                const long double c = beta == T(0) ? 0.0L : static_cast<long double>(beta) * at(C0, i, j);
                INFO("symm (" << i << ", " << j << ")");
                check<T>(at(C, i, j), 1.5L * s + c, 1.5L * abs + std::fabs(c), m);
            }
    }

    // alpha == 0 skips the GEMM: C = beta C, A and B unread.
    MC C = filled<MC>(m, n, 6);
    const MC C0 = C;
    symm(T(0), A, B, T(2), C);
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j) REQUIRE(C(i, j) == T(2) * C0(i, j));
}
