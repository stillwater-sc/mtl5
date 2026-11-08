// trmv and trsv on the fast paths (#747). In a BLAS build, dense float/double
// of EITHER orientation goes to ?trmv / ?trsv; a row-major A is passed as the
// transposed buffer with the other triangle. On the native-fast build without
// BLAS, n above detail::triangular_block is blocked: each block's off-diagonal
// panel through the SIMD gemv, its diagonal block through the row loop.
//
// The rounding is BLAS's or the SIMD gemv's, so results are checked against a
// long-double reference at a gamma_n-style tolerance, not bit for bit. The
// triangle that must not be read holds NaN, so reading it -- a wrong uplo
// mapping, a panel on the wrong side of the diagonal -- fails loudly. Shapes
// straddle the block size: exactly one block, one block plus one row, several
// blocks with a remainder.
//
// MTL5_NATIVE_FAST_GEMM is defined before the headers, as the other
// native-fast tests do, so in a build without BLAS this translation unit takes
// the blocked path.
#define MTL5_NATIVE_FAST_GEMM 1

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>

#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/parameter.hpp>
#include <mtl/operation/trmv.hpp>
#include <mtl/operation/trsv.hpp>
#include <mtl/tag/orientation.hpp>
#include <mtl/vec/dense_vector.hpp>

using namespace mtl;

namespace {

template <typename T> using rm = mat::dense2D<T, mat::parameters<tag::row_major>>;
template <typename T> using cm = mat::dense2D<T, mat::parameters<tag::col_major>>;
template <typename T> using vec_t = vec::dense_vector<T>;

// Every float/double dense case takes a fast path in a BLAS or a native-fast build.
template <typename M>
constexpr bool fast_v = detail::trmv_dispatches_blas_v<void, M, vec_t<typename M::value_type>> ||
                        detail::trmv_uses_blocked_v<void, M, vec_t<typename M::value_type>>;
template <typename M>
constexpr bool fast_sv = detail::trsv_dispatches_blas_v<void, M, vec_t<typename M::value_type>> ||
                         detail::trsv_uses_blocked_v<void, M, vec_t<typename M::value_type>>;
static_assert(fast_v<rm<double>> && fast_v<cm<double>> && fast_v<rm<float>> && fast_v<cm<float>>);
static_assert(fast_sv<rm<double>> && fast_sv<cm<double>> && fast_sv<rm<float>> && fast_sv<cm<float>>);
static_assert(!detail::trmv_uses_blocked_v<void, rm<long double>, vec_t<long double>>);
static_assert(!detail::trsv_uses_blocked_v<void, rm<long double>, vec_t<long double>>);
static_assert(!detail::trmv_uses_blocked_v<double, rm<double>, vec_t<double>>);   // explicit accumulator
static_assert(!detail::trsv_uses_blocked_v<void, rm<float>, vec_t<double>>);      // mixed types

double val(std::size_t seed, std::size_t i, std::size_t j) {
    std::size_t h = seed * 0x9E3779B97F4A7C15ull + i * 0xBF58476D1CE4E5B9ull + j * 0x94D049BB133111EBull;
    h ^= h >> 29; h *= 0xD6E8FEB86659FD93ull; h ^= h >> 32;
    return static_cast<double>(h % 1000003) / 500001.5 - 1.0;
}

/// Triangular and well conditioned (diagonal in [2, 3], off-diagonal O(1/n)),
/// NaN in the triangle that must not be read.
template <typename M>
M triangular(std::size_t n, bool upper, std::size_t seed) {
    using T = typename M::value_type;
    M A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            const bool stored = upper ? j >= i : j <= i;
            A(i, j) = !stored ? std::numeric_limits<T>::quiet_NaN()
                    : i == j  ? T(2.5 + val(seed, i, j) / 2)
                              : T(val(seed, i, j) / double(n));
        }
    return A;
}

template <typename T>
void check(long double got, long double want, long double absterms, std::size_t n) {
    const long double tol = 8.0L * static_cast<long double>(n + 2) *
                            static_cast<long double>(std::numeric_limits<T>::epsilon()) * (absterms + 1);
    REQUIRE(std::isfinite(static_cast<double>(got)));
    REQUIRE(std::fabs(got - want) <= tol);
}

} // namespace

TEMPLATE_TEST_CASE("trmv and trsv fast paths match the reference across block boundaries", "[operation][747]",
                   rm<double>, cm<double>, rm<float>, cm<float>) {
    using M = TestType;
    using T = typename M::value_type;
    constexpr std::size_t nb = detail::triangular_block;
    for (const std::size_t n : {nb, nb + 1, 3 * nb + 22})
        for (const bool upper : {false, true})
            for (const bool unit : {false, true}) {
                CAPTURE(n, upper, unit);
                const M A = triangular<M>(n, upper, 1);
                vec_t<T> x0(n);
                for (std::size_t i = 0; i < n; ++i) x0(i) = T(val(2, i, 0));
                const auto a = [&](std::size_t i, std::size_t k) -> long double {
                    if (i == k) return unit ? 1.0L : static_cast<long double>(A(i, i));
                    return (upper ? k > i : k < i) ? static_cast<long double>(A(i, k)) : 0.0L;
                };

                vec_t<T> x = x0;
                trmv(A, x, upper, unit);
                for (std::size_t i = 0; i < n; ++i) {
                    long double s = 0, abs = 0;
                    for (std::size_t k = 0; k < n; ++k) {
                        const long double t = a(i, k) * static_cast<long double>(x0(k));
                        s += t; abs += std::fabs(t);
                    }
                    INFO("trmv x(" << i << ")");
                    check<T>(static_cast<long double>(x(i)), s, abs, n);
                }

                // trsv, in place and with a separate b: the residual A x - b.
                for (const bool in_place : {true, false}) {
                    vec_t<T> y = x0;
                    if (in_place) trsv(A, y, upper, unit);
                    else          trsv(A, y, x0, upper, unit);
                    for (std::size_t i = 0; i < n; ++i) {
                        long double r = 0, abs = 0;
                        for (std::size_t k = 0; k < n; ++k) {
                            const long double t = a(i, k) * static_cast<long double>(y(k));
                            r += t; abs += std::fabs(t);
                        }
                        const long double b = static_cast<long double>(x0(i));
                        INFO("trsv x(" << i << ") in_place=" << in_place);
                        check<T>(r, b, abs + std::fabs(b), n);
                    }
                }
            }
}

TEST_CASE("trmv and trsv fast paths: small and empty", "[operation][747]") {
    // n <= triangular_block keeps the existing path; n == 0 is a no-op.
    for (const std::size_t n : {0u, 1u, 7u}) {
        CAPTURE(n);
        const rm<double> A = triangular<rm<double>>(n, false, 3);
        vec_t<double> x(n), y(n);
        for (std::size_t i = 0; i < n; ++i) x(i) = y(i) = val(4, i, 0);
        trmv(A, x, false);
        trsv(A, x, false);   // undoes the product
        for (std::size_t i = 0; i < n; ++i) REQUIRE(std::fabs(x(i) - y(i)) <= 1e-13 * (1 + std::fabs(y(i))));
    }
}
