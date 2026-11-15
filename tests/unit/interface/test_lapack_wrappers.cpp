// The thin LAPACK and BLAS wrappers that no MTL5 operation calls yet (#849):
// lapack::getrs, potrs, orgqr and trtrs, and blas::copy, asum and iamax.
// Nothing in the library reaches them, so without this file a wrong argument
// order or a wrong extern declaration would go unnoticed until a first caller
// arrived.
//
// Each wrapper is called in float and double on a non-square or multi-column
// problem (so a swapped dimension or leading dimension gives a wrong answer,
// not merely a transposed one), with a leading dimension larger than the row
// count where the routine takes one. The results are checked against a
// long-double computation from the inputs: a residual for the solves, the
// orthogonality and the reconstruction for orgqr, and the exact values for
// the BLAS level-1 routines.
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#ifdef MTL5_HAS_BLAS
#include <mtl/interface/blas.hpp>
#endif
#ifdef MTL5_HAS_LAPACK
#include <mtl/interface/lapack.hpp>
#endif

#ifdef MTL5_HAS_LAPACK

namespace {

/// Deterministic values in [-1, 1], distinct per (seed, i, j).
long double val(std::size_t seed, std::size_t i, std::size_t j) {
    std::uint64_t h = seed * 0x9E3779B97F4A7C15ull + i * 0xBF58476D1CE4E5B9ull + j * 0x94D049BB133111EBull;
    h ^= h >> 31;
    h *= 0xD6E8FEB86659FD93ull;
    h ^= h >> 32;
    return static_cast<long double>(h % 2001) / 1000.0L - 1.0L;
}

template <typename T> long double tol() { return 256 * static_cast<long double>(std::numeric_limits<T>::epsilon()); }

/// A column-major r x c matrix with leading dimension ld >= r; the padding
/// rows hold NaN, so a routine that reads them poisons its result.
template <typename T>
std::vector<T> colmajor(std::size_t r, std::size_t c, std::size_t ld, std::size_t seed) {
    std::vector<T> a(ld * c, std::numeric_limits<T>::quiet_NaN());
    for (std::size_t j = 0; j < c; ++j)
        for (std::size_t i = 0; i < r; ++i) a[j * ld + i] = static_cast<T>(val(seed, i, j));
    return a;
}

/// Adds `shift` to the diagonal of a column-major n x n matrix.
template <typename T>
void shift_diagonal(std::vector<T>& a, std::size_t n, std::size_t ld, long double shift) {
    for (std::size_t i = 0; i < n; ++i)
        a[i * ld + i] = static_cast<T>(static_cast<long double>(a[i * ld + i]) + shift);
}

/// |op(A) X - B| <= tol (1 + |B|) for each entry, op(A) = A or A^T, n x n.
template <typename T>
void require_solves(const std::vector<T>& A, std::size_t lda, bool trans, std::size_t n,
                    const std::vector<T>& X, const std::vector<T>& B, std::size_t ldb, std::size_t nrhs) {
    for (std::size_t c = 0; c < nrhs; ++c)
        for (std::size_t i = 0; i < n; ++i) {
            long double s = 0;
            for (std::size_t k = 0; k < n; ++k) {
                const T aik = trans ? A[i * lda + k] : A[k * lda + i];
                s += static_cast<long double>(aik) * static_cast<long double>(X[c * ldb + k]);
            }
            const long double b = static_cast<long double>(B[c * ldb + i]);
            INFO("row " << i << ", rhs " << c << ": got " << static_cast<double>(s) << ", want "
                        << static_cast<double>(b));
            REQUIRE(std::fabs(s - b) <= tol<T>() * (1 + std::fabs(b)));
        }
}

} // namespace

using namespace mtl::interface;

TEMPLATE_TEST_CASE("lapack::getrs solves after getrf, plain and transposed (#849)",
                   "[interface][lapack][849]", float, double) {
    using T = TestType;
    const std::size_t n = 5, nrhs = 3, lda = 7, ldb = 6;
    std::vector<T> A = colmajor<T>(n, n, lda, 1);
    shift_diagonal(A, n, lda, 3.0L);
    const std::vector<T> A0 = A;
    std::vector<int> ipiv(n);
    REQUIRE(lapack::getrf(static_cast<int>(n), static_cast<int>(n), A.data(), static_cast<int>(lda),
                          ipiv.data()) == 0);
    for (const char trans : {'N', 'T'}) {
        CAPTURE(trans);
        const std::vector<T> B = colmajor<T>(n, nrhs, ldb, 2);
        std::vector<T> X = B;
        REQUIRE(lapack::getrs(trans, static_cast<int>(n), static_cast<int>(nrhs), A.data(),
                              static_cast<int>(lda), ipiv.data(), X.data(), static_cast<int>(ldb)) == 0);
        require_solves(A0, lda, trans == 'T', n, X, B, ldb, nrhs);
    }
    // No bad-argument case: reference LAPACK's xerbla stops the process
    // instead of returning a negative info.
}

TEMPLATE_TEST_CASE("lapack::potrs solves after potrf, either triangle (#849)",
                   "[interface][lapack][849]", float, double) {
    using T = TestType;
    const std::size_t n = 6, nrhs = 2, lda = 8, ldb = 7;
    // A = G G^T + n I: symmetric positive definite.
    const std::vector<T> G = colmajor<T>(n, n, n, 3);
    std::vector<T> A(lda * n, std::numeric_limits<T>::quiet_NaN());
    for (std::size_t j = 0; j < n; ++j)
        for (std::size_t i = 0; i < n; ++i) {
            long double s = i == j ? static_cast<long double>(n) : 0.0L;
            for (std::size_t k = 0; k < n; ++k)
                s += static_cast<long double>(G[k * n + i]) * static_cast<long double>(G[k * n + j]);
            A[j * lda + i] = static_cast<T>(s);
        }
    for (const char uplo : {'L', 'U'}) {
        CAPTURE(uplo);
        std::vector<T> F = A;
        REQUIRE(lapack::potrf(uplo, static_cast<int>(n), F.data(), static_cast<int>(lda)) == 0);
        const std::vector<T> B = colmajor<T>(n, nrhs, ldb, 4);
        std::vector<T> X = B;
        REQUIRE(lapack::potrs(uplo, static_cast<int>(n), static_cast<int>(nrhs), F.data(),
                              static_cast<int>(lda), X.data(), static_cast<int>(ldb)) == 0);
        require_solves(A, lda, false, n, X, B, ldb, nrhs);
    }
}

TEMPLATE_TEST_CASE("lapack::orgqr forms Q from geqrf: Q^T Q = I and Q R = A (#849)",
                   "[interface][lapack][849]", float, double) {
    using T = TestType;
    const std::size_t m = 7, n = 4, lda = 9;
    const std::vector<T> A0 = colmajor<T>(m, n, lda, 5);
    std::vector<T> A = A0;
    std::vector<T> tau(n);
    T wq{};
    REQUIRE(lapack::geqrf(static_cast<int>(m), static_cast<int>(n), A.data(), static_cast<int>(lda),
                          tau.data(), &wq, -1) == 0);
    std::vector<T> work(static_cast<std::size_t>(wq) + n);
    REQUIRE(lapack::geqrf(static_cast<int>(m), static_cast<int>(n), A.data(), static_cast<int>(lda),
                          tau.data(), work.data(), static_cast<int>(work.size())) == 0);
    // R is the upper triangle of the factored A; keep it before orgqr overwrites A.
    std::vector<long double> R(n * n, 0.0L);
    for (std::size_t j = 0; j < n; ++j)
        for (std::size_t i = 0; i <= j; ++i) R[j * n + i] = static_cast<long double>(A[j * lda + i]);

    REQUIRE(lapack::orgqr(static_cast<int>(m), static_cast<int>(n), static_cast<int>(n), A.data(),
                          static_cast<int>(lda), tau.data(), &wq, -1) == 0);
    work.assign(static_cast<std::size_t>(wq) + n, T{});
    REQUIRE(lapack::orgqr(static_cast<int>(m), static_cast<int>(n), static_cast<int>(n), A.data(),
                          static_cast<int>(lda), tau.data(), work.data(), static_cast<int>(work.size())) == 0);

    for (std::size_t a = 0; a < n; ++a)
        for (std::size_t b = 0; b < n; ++b) {
            long double s = 0;
            for (std::size_t i = 0; i < m; ++i)
                s += static_cast<long double>(A[a * lda + i]) * static_cast<long double>(A[b * lda + i]);
            INFO("(Q^T Q)(" << a << ", " << b << ") = " << static_cast<double>(s));
            REQUIRE(std::fabs(s - (a == b ? 1.0L : 0.0L)) <= tol<T>());
        }
    for (std::size_t j = 0; j < n; ++j)
        for (std::size_t i = 0; i < m; ++i) {
            long double s = 0;
            for (std::size_t k = 0; k <= j; ++k) s += static_cast<long double>(A[k * lda + i]) * R[j * n + k];
            const long double want = static_cast<long double>(A0[j * lda + i]);
            INFO("(Q R)(" << i << ", " << j << ") = " << static_cast<double>(s));
            REQUIRE(std::fabs(s - want) <= tol<T>() * (1 + std::fabs(want)));
        }
}

TEMPLATE_TEST_CASE("lapack::trtrs: every triangle, transpose and diagonal (#849)",
                   "[interface][lapack][849]", float, double) {
    using T = TestType;
    const std::size_t n = 6, nrhs = 2, lda = 8, ldb = 7;
    for (const char uplo : {'U', 'L'})
        for (const char trans : {'N', 'T'})
            for (const char diag : {'N', 'U'}) {
                CAPTURE(uplo, trans, diag);
                // The triangle trtrs must not read is NaN, and so is the
                // diagonal when it is taken as unit.
                std::vector<T> A(lda * n, std::numeric_limits<T>::quiet_NaN());
                std::vector<T> Afull(lda * n, T(0));   // the triangle as it is used
                for (std::size_t j = 0; j < n; ++j)
                    for (std::size_t i = 0; i < n; ++i) {
                        const bool stored = uplo == 'U' ? i <= j : i >= j;
                        if (!stored) continue;
                        const T v = i == j ? static_cast<T>(2.5L + val(6, i, j) / 2)
                                           : static_cast<T>(val(6, i, j) / static_cast<long double>(n));
                        A[j * lda + i] = (i == j && diag == 'U') ? std::numeric_limits<T>::quiet_NaN() : v;
                        Afull[j * lda + i] = (i == j && diag == 'U') ? T(1) : v;
                    }
                const std::vector<T> B = colmajor<T>(n, nrhs, ldb, 7);
                std::vector<T> X = B;
                REQUIRE(lapack::trtrs(uplo, trans, diag, static_cast<int>(n), static_cast<int>(nrhs), A.data(),
                                      static_cast<int>(lda), X.data(), static_cast<int>(ldb)) == 0);
                require_solves(Afull, lda, trans == 'T', n, X, B, ldb, nrhs);
            }
    // A zero on the diagonal: info = its 1-based index, nothing solved.
    std::vector<T> A = colmajor<T>(n, n, lda, 8);
    for (std::size_t i = 0; i < n; ++i) A[i * lda + i] = T(2);
    A[3 * lda + 3] = T(0);
    std::vector<T> X = colmajor<T>(n, nrhs, ldb, 7);
    REQUIRE(lapack::trtrs('U', 'N', 'N', static_cast<int>(n), static_cast<int>(nrhs), A.data(),
                          static_cast<int>(lda), X.data(), static_cast<int>(ldb)) == 4);
}

#endif // MTL5_HAS_LAPACK

#ifdef MTL5_HAS_BLAS

TEMPLATE_TEST_CASE("blas::copy with strides (#849)", "[interface][blas][849]", float, double) {
    using T = TestType;
    const std::vector<T> x = {1, -2, 3, -4, 5, -6, 7};
    std::vector<T> y(8, T(-99));
    // Every second element of x (1, 3, 5, 7) into every second slot of y.
    mtl::interface::blas::copy(4, x.data(), 2, y.data(), 2);
    REQUIRE(y == std::vector<T>{1, -99, 3, -99, 5, -99, 7, -99});
    // n = 0 copies nothing.
    mtl::interface::blas::copy(0, x.data(), 1, y.data(), 1);
    REQUIRE(y[0] == T(1));
    // incy = 1 from incx = 3: 1, -4, 7.
    std::vector<T> z(3, T(0));
    mtl::interface::blas::copy(3, x.data(), 3, z.data(), 1);
    REQUIRE(z == std::vector<T>{1, -4, 7});
}

TEMPLATE_TEST_CASE("blas::asum sums magnitudes (#849)", "[interface][blas][849]", float, double) {
    using T = TestType;
    const std::vector<T> x = {1, -2, 3, -4, 5, -6, 7};
    REQUIRE(mtl::interface::blas::asum(7, x.data(), 1) == T(28));
    REQUIRE(mtl::interface::blas::asum(4, x.data(), 2) == T(16));   // 1 + 3 + 5 + 7
    REQUIRE(mtl::interface::blas::asum(3, x.data() + 1, 2) == T(12));   // 2 + 4 + 6
    REQUIRE(mtl::interface::blas::asum(0, x.data(), 1) == T(0));
}

TEMPLATE_TEST_CASE("blas::iamax: 1-based index of the first largest magnitude (#849)",
                   "[interface][blas][849]", float, double) {
    using T = TestType;
    const std::vector<T> x = {1, -9, 3, 9, -2};
    REQUIRE(mtl::interface::blas::iamax(5, x.data(), 1) == 2);   // -9 first, before 9
    REQUIRE(mtl::interface::blas::iamax(3, x.data(), 2) == 2);   // 1, 3, -2: the 3
    REQUIRE(mtl::interface::blas::iamax(1, x.data(), 1) == 1);
    REQUIRE(mtl::interface::blas::iamax(0, x.data(), 1) == 0);   // n < 1
}

#endif // MTL5_HAS_BLAS

#if !defined(MTL5_HAS_BLAS) && !defined(MTL5_HAS_LAPACK)
TEST_CASE("lapack and blas wrappers: placeholder (no BLAS or LAPACK)", "[interface][849]") {
    REQUIRE(true);
}
#endif
