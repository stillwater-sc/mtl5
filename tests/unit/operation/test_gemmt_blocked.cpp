// gemmt's blocked path (#744): above detail::gemmt_block rows, dense
// float/double operands go through a full-speed GEMM -- ?gemm in a BLAS build,
// the native blocked GEMM otherwise -- by halving the triangle recursively.
//
// The rounding is the GEMM's, so results are checked against a long-double
// reference at a gamma_k-style tolerance, not bit for bit. Shapes straddle the
// block (one block plus one row, several levels with odd halves), k ranges from
// 1 up past n, and every orientation of A, B and C is covered, plus transposed
// views. The contract checks are exact:
//   - with gemmt_fill::keep, the other triangle still holds its sentinels, so
//     neither a panel nor a diagonal tile wrote across the diagonal;
//   - with beta == 0, C starts as NaN and none of it reaches the result;
//   - with gemmt_fill::mirror, C is exactly symmetric.
//
// MTL5_NATIVE_FAST_GEMM is defined before the headers, as the other
// native-fast tests do. In a BLAS build the BLAS branches are taken instead;
// with a ?gemmt that is not OpenBLAS's, gemmt calls it and this test checks
// that path to the same tolerance.
#define MTL5_NATIVE_FAST_GEMM 1

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/parameter.hpp>
#include <mtl/operation/gemmt.hpp>
#include <mtl/operation/trans.hpp>
#include <mtl/tag/orientation.hpp>

using namespace mtl;

namespace {

template <typename T> using rm = mat::dense2D<T, mat::parameters<tag::row_major>>;
template <typename T> using cm = mat::dense2D<T, mat::parameters<tag::col_major>>;

// The blocked path, or ?gemmt where the build calls it instead.
template <typename A, typename B, typename C>
constexpr bool fast_v = detail::gemmt_uses_blocked_v<void, A, B, C> || detail::gemmt_dispatches_blas_v<void, A, B, C>;
static_assert(fast_v<rm<double>, rm<double>, rm<double>>);
static_assert(fast_v<cm<float>, rm<float>, cm<float>>);
static_assert(fast_v<rm<double>, mat::view::transposed_view<cm<double>>, rm<double>>);
static_assert(!detail::gemmt_uses_blocked_v<void, rm<long double>, rm<long double>, rm<long double>>);
static_assert(!detail::gemmt_uses_blocked_v<double, rm<float>, rm<float>, rm<float>>);   // explicit accumulator
static_assert(!detail::gemmt_uses_blocked_v<void, rm<float>, rm<double>, rm<double>>);   // mixed types
static_assert(!detail::gemmt_uses_blocked_v<void, rm<double>, mat::view::transposed_view<mat::compressed2D<double>>,
                                            rm<double>>);

double val(std::size_t seed, std::size_t i, std::size_t j) {
    std::size_t h = seed * 0x9E3779B97F4A7C15ull + i * 0xBF58476D1CE4E5B9ull + j * 0x94D049BB133111EBull;
    h ^= h >> 29; h *= 0xD6E8FEB86659FD93ull; h ^= h >> 32;
    return static_cast<double>(h % 1000003) / 500001.5 - 1.0;
}

template <typename M>
M filled(std::size_t r, std::size_t c, std::size_t seed) {
    M X(r, c);
    for (std::size_t i = 0; i < r; ++i)
        for (std::size_t j = 0; j < c; ++j) X(i, j) = typename M::value_type(val(seed, i, j));
    return X;
}

/// A*B and sum |a b| in long double, from the same values every orientation holds.
struct reference {
    std::size_t n;
    std::vector<long double> s, abs;
    reference(std::size_t n_, std::size_t k, std::size_t sa, std::size_t sb) : n(n_), s(n_ * n_), abs(n_ * n_) {
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j) {
                long double t = 0, a = 0;
                for (std::size_t l = 0; l < k; ++l) {
                    // The double values: a float operand differs by its input
                    // rounding, which the tolerance covers.
                    const long double p = static_cast<long double>(val(sa, i, l)) * val(sb, l, j);
                    t += p; a += std::fabs(p);
                }
                s[i * n + j] = t; abs[i * n + j] = a;
            }
    }
};

bool in_triangle(uplo t, std::size_t i, std::size_t j) { return t == uplo::lower ? j <= i : j >= i; }

/// Runs gemmt on one combination and checks it against the reference R.
/// A or B may be a transposed view of a stored transpose.
template <typename T, typename MC, typename MA, typename MB>
void check_one(const MA& A, const MB& B, std::size_t n, std::size_t k, const reference& R,
               uplo t, gemmt_fill fill, T beta) {
    const T alpha = T(-1.25);
    MC C(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            C(i, j) = beta == T(0) ? std::numeric_limits<T>::quiet_NaN()   // must not be read
                                   : T(val(9, std::max(i, j), std::min(i, j)));
    MC C0 = C;
    if (fill == gemmt_fill::keep)   // sentinels: the other triangle must survive
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                if (!in_triangle(t, i, j)) C(i, j) = C0(i, j) = T(1000 + double(i) - double(j));
    gemmt(alpha, A, B, beta, C, t, fill);

    const long double eps = std::numeric_limits<T>::epsilon();
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            const bool tri = in_triangle(t, i, j);
            if (!tri && fill == gemmt_fill::keep) {
                if (!(C(i, j) == C0(i, j))) { INFO("other triangle written at (" << i << ", " << j << ")"); REQUIRE(false); }
                continue;
            }
            if (!tri) {   // mirrored
                if (!(C(i, j) == C(j, i))) { INFO("not symmetric at (" << i << ", " << j << ")"); REQUIRE(false); }
                continue;
            }
            const long double c = beta == T(0) ? 0.0L : static_cast<long double>(beta) * static_cast<long double>(C0(i, j));
            const long double want = static_cast<long double>(alpha) * R.s[i * n + j] + c;
            const long double scale = std::fabs(static_cast<long double>(alpha)) * R.abs[i * n + j] + std::fabs(c) + 1;
            // gamma_k for the sum, plus the rounding of the inputs to T (2 eps per product).
            const long double tol = 8.0L * static_cast<long double>(k + 4) * eps * scale;
            const long double got = static_cast<long double>(C(i, j));
            if (!(std::isfinite(static_cast<double>(got)) && std::fabs(got - want) <= tol)) {
                INFO("(" << i << ", " << j << ") got " << static_cast<double>(got) << " want "
                         << static_cast<double>(want) << " tol " << static_cast<double>(tol));
                REQUIRE(false);
            }
        }
}

template <typename T, typename MA, typename MB, typename MC>
void sweep(std::size_t n, std::size_t k, const reference& R) {
    const MA A = filled<MA>(n, k, 1);
    const MB B = filled<MB>(k, n, 2);
    for (const uplo t : {uplo::lower, uplo::upper})
        for (const gemmt_fill fill : {gemmt_fill::keep, gemmt_fill::mirror})
            for (const T beta : {T(0), T(0.5)}) {
                CAPTURE(n, k, t == uplo::lower, fill == gemmt_fill::keep, static_cast<double>(beta));
                check_one<T, MC>(A, B, n, k, R, t, fill, beta);
            }
}

template <typename T>
void all_orientations(std::size_t n, std::size_t k) {
    const reference R(n, k, 1, 2);
    sweep<T, rm<T>, rm<T>, rm<T>>(n, k, R);
    sweep<T, rm<T>, rm<T>, cm<T>>(n, k, R);
    sweep<T, rm<T>, cm<T>, rm<T>>(n, k, R);
    sweep<T, rm<T>, cm<T>, cm<T>>(n, k, R);
    sweep<T, cm<T>, rm<T>, rm<T>>(n, k, R);
    sweep<T, cm<T>, rm<T>, cm<T>>(n, k, R);
    sweep<T, cm<T>, cm<T>, rm<T>>(n, k, R);
    sweep<T, cm<T>, cm<T>, cm<T>>(n, k, R);
}

} // namespace

TEST_CASE("blocked gemmt: every orientation across block boundaries, double", "[operation][gemmt][744]") {
    constexpr std::size_t nb = detail::gemmt_block;
    all_orientations<double>(nb + 1, 1);
    all_orientations<double>(nb + 1, 70);
    all_orientations<double>(2 * nb + 37, 50);
}

TEST_CASE("blocked gemmt: every orientation across block boundaries, float", "[operation][gemmt][744]") {
    constexpr std::size_t nb = detail::gemmt_block;
    all_orientations<float>(nb + 1, 33);
    all_orientations<float>(2 * nb + 37, 9);
}

TEST_CASE("blocked gemmt: several levels of recursion, k past n", "[operation][gemmt][744]") {
    const std::size_t n = 4 * detail::gemmt_block + 45, k = n + 31;   // odd halves at each level
    const reference R(n, k, 1, 2);
    sweep<double, rm<double>, rm<double>, rm<double>>(n, k, R);
    sweep<double, cm<double>, cm<double>, cm<double>>(n, k, R);
}

TEST_CASE("blocked gemmt: transposed views of A and B", "[operation][gemmt][744]") {
    const std::size_t n = 2 * detail::gemmt_block + 3, k = 41;
    const reference R(n, k, 1, 2);
    // At(l, i) = A(i, l) and Bt(j, l) = B(l, j), so trans() of them are A and B.
    rm<double> At(k, n);
    cm<double> Bt(n, k);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t l = 0; l < k; ++l) { At(l, i) = val(1, i, l); Bt(i, l) = val(2, l, i); }
    const rm<double> A = filled<rm<double>>(n, k, 1);
    const cm<double> B = filled<cm<double>>(k, n, 2);
    for (const uplo t : {uplo::lower, uplo::upper})
        for (const double beta : {0.0, 0.5}) {
            CAPTURE(t == uplo::lower, beta);
            check_one<double, rm<double>>(trans(At), B, n, k, R, t, gemmt_fill::keep, beta);
            check_one<double, cm<double>>(A, trans(Bt), n, k, R, t, gemmt_fill::keep, beta);
            check_one<double, rm<double>>(trans(At), trans(Bt), n, k, R, t, gemmt_fill::mirror, beta);
        }
}

TEST_CASE("blocked gemmt: alpha == 0 is beta C without reading A or B", "[operation][gemmt][744]") {
    const std::size_t n = 2 * detail::gemmt_block + 5, k = 20;
    rm<double> A(n, k), B(k, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t l = 0; l < k; ++l) A(i, l) = B(l, i) = std::numeric_limits<double>::quiet_NaN();
    rm<double> C = filled<rm<double>>(n, n, 3);
    const rm<double> C0 = C;
    gemmt(0.0, A, B, 2.0, C, uplo::lower, gemmt_fill::keep);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) REQUIRE(C(i, j) == (j <= i ? 2.0 * C0(i, j) : C0(i, j)));
}

TEST_CASE("blocked gemmt: congruence above the block size", "[operation][gemmt][congruence][744]") {
    // C = A P A^T: the second product goes to gemmt as trans(A) (no copy).
    const std::size_t m = 2 * detail::gemmt_block + 11, n = 23;
    const rm<double> A = filled<rm<double>>(m, n, 4);
    rm<double> P(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) P(i, j) = val(5, std::max(i, j), std::min(i, j));
    rm<double> C(m, m);
    congruence(A, P, C);
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j <= i; ++j) {
            long double s = 0, abs = 0;
            for (std::size_t p = 0; p < n; ++p)
                for (std::size_t q = 0; q < n; ++q) {
                    const long double t = static_cast<long double>(A(i, p)) * P(p, q) * A(j, q);
                    s += t; abs += std::fabs(t);
                }
            const long double tol = 16.0L * (2 * n + 4) * std::numeric_limits<double>::epsilon() * (abs + 1);
            INFO("(" << i << ", " << j << ")");
            REQUIRE(std::fabs(static_cast<long double>(C(i, j)) - s) <= tol);
            REQUIRE(C(i, j) == C(j, i));
        }
}
