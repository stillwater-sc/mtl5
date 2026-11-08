// The generic trsm, trmm, symm, symv, syrk and syr2k walk row-major operands
// row by row (#724), and the result is BIT-IDENTICAL to the loops they replace.
//
// Before #724 the generic paths were textbook loops whose inner access ran down
// a column of a row-major matrix (a stride of n per access), which is why trsm
// and trmm ran at 2% of the roofline and symm at 1%. The row forms keep every
// entry's summation order and expressions, so the bits do not change; this
// test holds them to that, against verbatim copies of the old loops below.
//
// In a BLAS build, row-major float and double now go to BLAS (#724 part 1), so
// those cases are only checked to be close; long double never reaches BLAS and
// exercises the row loops in every build. The *_uses_row_loop_v traits are
// static_asserted, so a fallback to the old loops fails to compile.
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>

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

static_assert(detail::trsm_uses_row_loop_v<rm<double>> && !detail::trsm_uses_row_loop_v<cm<double>>);
static_assert(detail::trmm_uses_row_loop_v<rm<float>> && !detail::trmm_uses_row_loop_v<cm<float>>);
static_assert(detail::symm_uses_row_loop_v<rm<long double>> && !detail::symm_uses_row_loop_v<cm<long double>>);
static_assert(detail::symv_uses_row_loop_v<rm<double>> && !detail::symv_uses_row_loop_v<cm<double>>);
static_assert(detail::syrk_uses_row_loop_v<rm<double>> && !detail::syrk_uses_row_loop_v<cm<double>>);
static_assert(detail::syr2k_uses_row_loop_v<rm<double>, rm<double>> &&
              !detail::syr2k_uses_row_loop_v<rm<double>, cm<double>>);

// -- the pre-#724 generic loops, verbatim ----------------------------------------

template <typename S, typename MA, typename MB>
void old_trsm(const S& alpha, const MA& A, MB& B, bool upper, bool unit_diag) {
    using size_type = typename MA::size_type;
    const size_type m = A.num_rows();
    const size_type n = B.num_cols();
    for (size_type c = 0; c < n; ++c) {
        if (upper) {
            for (size_type ii = m; ii-- > 0; ) {
                auto s = alpha * B(ii, c);
                for (size_type j = ii + 1; j < m; ++j)
                    s -= A(ii, j) * B(j, c);
                B(ii, c) = unit_diag ? s : s / A(ii, ii);
            }
        } else {
            for (size_type i = 0; i < m; ++i) {
                auto s = alpha * B(i, c);
                for (size_type j = 0; j < i; ++j)
                    s -= A(i, j) * B(j, c);
                B(i, c) = unit_diag ? s : s / A(i, i);
            }
        }
    }
}

template <typename S, typename MA, typename MB>
void old_trmm(const S& alpha, const MA& A, MB& B, bool upper, bool unit_diag) {
    using size_type = typename MA::size_type;
    const size_type m = A.num_rows();
    const size_type n = B.num_cols();
    for (size_type c = 0; c < n; ++c) {
        if (upper) {
            for (size_type i = 0; i < m; ++i) {
                auto acc = unit_diag ? B(i, c) : A(i, i) * B(i, c);
                for (size_type j = i + 1; j < m; ++j)
                    acc += A(i, j) * B(j, c);
                B(i, c) = alpha * acc;
            }
        } else {
            for (size_type ii = m; ii-- > 0; ) {
                auto acc = unit_diag ? B(ii, c) : A(ii, ii) * B(ii, c);
                for (size_type j = 0; j < ii; ++j)
                    acc += A(ii, j) * B(j, c);
                B(ii, c) = alpha * acc;
            }
        }
    }
}

template <typename S, typename MA, typename MB, typename MC>
void old_symm(const S& alpha, const MA& A, const MB& B, const S& beta, MC& C) {
    using value_type = typename MC::value_type;
    using size_type  = typename MA::size_type;
    const size_type m = A.num_rows();
    const size_type n = B.num_cols();
    const bool alpha_zero = (alpha == math::zero<S>());
    const bool beta_zero  = (beta == math::zero<S>());
    auto a = [&A](size_type i, size_type k) -> decltype(auto) { return k <= i ? A(i, k) : A(k, i); };
    for (size_type i = 0; i < m; ++i)
        for (size_type j = 0; j < n; ++j) {
            if (alpha_zero) {
                C(i, j) = beta_zero ? math::zero<value_type>() : value_type(beta * C(i, j));
                continue;
            }
            auto acc = math::zero<value_type>();
            for (size_type k = 0; k < m; ++k)
                acc += a(i, k) * B(k, j);
            C(i, j) = beta_zero ? value_type(alpha * acc) : value_type(alpha * acc + beta * C(i, j));
        }
}

template <typename S, typename M, typename VX, typename VY>
void old_symv(const S& alpha, const M& A, const VX& x, const S& beta, VY& y) {
    using value_type = typename VY::value_type;
    using size_type  = typename M::size_type;
    const size_type n = A.num_rows();
    const bool alpha_zero = (alpha == math::zero<S>());
    const bool beta_zero  = (beta == math::zero<S>());
    auto a = [&A](size_type i, size_type j) -> decltype(auto) { return j <= i ? A(i, j) : A(j, i); };
    for (size_type i = 0; i < n; ++i) {
        if (alpha_zero) {
            y(i) = beta_zero ? math::zero<value_type>() : value_type(beta * y(i));
            continue;
        }
        auto acc = math::zero<value_type>();
        for (size_type j = 0; j < n; ++j)
            acc += a(i, j) * x(j);
        y(i) = beta_zero ? value_type(alpha * acc) : value_type(alpha * acc + beta * y(i));
    }
}

template <typename S, typename MA, typename MC>
void old_syrk(const S& alpha, const MA& A, const S& beta, MC& C) {
    using value_type = typename MC::value_type;
    using size_type  = typename MA::size_type;
    const size_type m = A.num_rows();
    const size_type k = A.num_cols();
    const bool alpha_zero = (alpha == math::zero<S>());
    const bool beta_zero  = (beta == math::zero<S>());
    for (size_type i = 0; i < m; ++i)
        for (size_type j = 0; j <= i; ++j) {
            if (alpha_zero) {
                C(i, j) = beta_zero ? math::zero<value_type>() : value_type(beta * C(i, j));
                continue;
            }
            auto acc = math::zero<value_type>();
            for (size_type l = 0; l < k; ++l)
                acc += A(i, l) * A(j, l);
            C(i, j) = beta_zero ? value_type(alpha * acc) : value_type(alpha * acc + beta * C(i, j));
        }
    for (size_type i = 0; i < m; ++i)
        for (size_type j = 0; j < i; ++j)
            C(j, i) = C(i, j);
}

template <typename S, typename MA, typename MB, typename MC>
void old_syr2k(const S& alpha, const MA& A, const MB& B, const S& beta, MC& C) {
    using value_type = typename MC::value_type;
    using size_type  = typename MA::size_type;
    const size_type m = A.num_rows();
    const size_type k = A.num_cols();
    const bool alpha_zero = (alpha == math::zero<S>());
    const bool beta_zero  = (beta == math::zero<S>());
    for (size_type i = 0; i < m; ++i)
        for (size_type j = 0; j <= i; ++j) {
            if (alpha_zero) {
                C(i, j) = beta_zero ? math::zero<value_type>() : value_type(beta * C(i, j));
                continue;
            }
            auto acc = math::zero<value_type>();
            for (size_type l = 0; l < k; ++l)
                acc += A(i, l) * B(j, l) + B(i, l) * A(j, l);
            C(i, j) = beta_zero ? value_type(alpha * acc) : value_type(alpha * acc + beta * C(i, j));
        }
    for (size_type i = 0; i < m; ++i)
        for (size_type j = 0; j < i; ++j)
            C(j, i) = C(i, j);
}

// -- inputs and comparison --------------------------------------------------------

double val(std::size_t seed, std::size_t i, std::size_t j) {
    std::size_t h = seed * 0x9E3779B97F4A7C15ull + i * 0xBF58476D1CE4E5B9ull + j * 0x94D049BB133111EBull;
    h ^= h >> 29; h *= 0xD6E8FEB86659FD93ull; h ^= h >> 32;
    return static_cast<double>(h % 1000003) / 500001.5 - 1.0;   // many distinct mantissas
}

template <typename M>
M filled(std::size_t r, std::size_t c, std::size_t seed) {
    M A(r, c);
    for (std::size_t i = 0; i < r; ++i)
        for (std::size_t j = 0; j < c; ++j) A(i, j) = static_cast<typename M::value_type>(val(seed, i, j));
    return A;
}

template <typename M>
M triangular(std::size_t m, std::size_t seed) {   // well conditioned, full storage
    M A = filled<M>(m, m, seed);
    for (std::size_t i = 0; i < m; ++i) A(i, i) = static_cast<typename M::value_type>(2.0 + val(seed, i, i) / 2);
    return A;
}

/// The same value, and the same sign of zero. Not memcmp: x87 long double has
/// six bytes of padding whose contents are unspecified.
template <typename T>
bool identical(T x, T y) {
    if (std::isnan(x) || std::isnan(y)) return std::isnan(x) && std::isnan(y);
    return x == y && std::signbit(x) == std::signbit(y);
}

template <typename M1, typename M2>
void require_identical(const M1& X, const M2& Y, std::size_t r, std::size_t c) {
    for (std::size_t i = 0; i < r; ++i)
        for (std::size_t j = 0; j < c; ++j) {
            INFO("(" << i << ", " << j << ") " << static_cast<double>(X(i, j)) << " vs "
                     << static_cast<double>(Y(i, j)));
            REQUIRE(identical(X(i, j), Y(i, j)));
        }
}

/// Close, for the cases a BLAS build sends to BLAS instead of the row loop.
template <typename M1, typename M2>
void require_close(const M1& X, const M2& Y, std::size_t r, std::size_t c) {
    using T = typename M1::value_type;
    const double tol = 256 * static_cast<double>(std::numeric_limits<T>::epsilon());
    for (std::size_t i = 0; i < r; ++i)
        for (std::size_t j = 0; j < c; ++j) {
            INFO("(" << i << ", " << j << ")");
            REQUIRE(std::fabs(static_cast<double>(X(i, j)) - static_cast<double>(Y(i, j))) <=
                    tol * (1 + std::fabs(static_cast<double>(Y(i, j)))));
        }
}

template <bool ToBlas, typename M1, typename M2>
void require_same(const M1& X, const M2& Y, std::size_t r, std::size_t c) {
    if constexpr (ToBlas) require_close(X, Y, r, c);
    else                  require_identical(X, Y, r, c);
}

// Shapes chosen so no dimension equals another, and m, n are not multiples of
// a vector width.
constexpr std::size_t M_ = 23, N_ = 13, K_ = 17;

} // namespace

TEMPLATE_TEST_CASE("trsm and trmm row loops are bit-identical to the column loops", "[operation][724]",
                   float, double, long double) {
    using M = rm<TestType>;
    constexpr bool blas = detail::trsm_dispatches_blas_v<void, M, M>;
    for (const bool upper : {false, true})
        for (const bool unit : {false, true}) {
            CAPTURE(upper, unit);
            const M A = triangular<M>(M_, 1);
            const M B0 = filled<M>(M_, N_, 2);
            M X = B0, Y = B0;
            trsm(TestType(0.625), A, X, upper, unit);
            old_trsm(TestType(0.625), A, Y, upper, unit);
            require_same<blas>(X, Y, M_, N_);
            X = B0; Y = B0;
            trmm(TestType(-1.375), A, X, upper, unit);
            old_trmm(TestType(-1.375), A, Y, upper, unit);
            require_same<blas>(X, Y, M_, N_);
        }
}

TEST_CASE("trsm and trmm row loops: a double alpha over float matrices", "[operation][724]") {
    // alpha * B(i, c) is double, so the column loop's `s` is double; the row
    // loop's buffer must be too, or each step would round to float.
    using M = rm<float>;
    const M A = triangular<M>(M_, 3);
    const M B0 = filled<M>(M_, N_, 4);
    M X = B0, Y = B0;
    trsm(0.1, A, X, false);
    old_trsm(0.1, A, Y, false, false);
    if constexpr (!detail::trsm_dispatches_blas_v<void, M, M>) require_identical(X, Y, M_, N_);
    else                                                       require_close(X, Y, M_, N_);
}

TEMPLATE_TEST_CASE("symm and symv row loops are bit-identical to the old loops", "[operation][724]",
                   float, double, long double) {
    using M = rm<TestType>;
    using V = vec::dense_vector<TestType>;
    const M A = filled<M>(M_, M_, 5);   // full storage: only the lower triangle is read
    const M B = filled<M>(M_, N_, 6);
    const M C0 = filled<M>(M_, N_, 7);
    for (const TestType beta : {TestType(0), TestType(0.75)}) {
        CAPTURE(static_cast<double>(beta));
        M X = C0, Y = C0;
        symm(TestType(1.25), A, B, beta, X);
        old_symm(TestType(1.25), A, B, beta, Y);
        require_same<detail::symm_dispatches_blas_v<void, M, M, M>>(X, Y, M_, N_);

        V x(M_), y0(M_);
        for (std::size_t i = 0; i < M_; ++i) { x(i) = TestType(val(8, i, 0)); y0(i) = TestType(val(9, i, 0)); }
        V u = y0, w = y0;
        symv(TestType(1.25), A, x, beta, u);
        old_symv(TestType(1.25), A, x, beta, w);
        for (std::size_t i = 0; i < M_; ++i) {
            INFO("y(" << i << ")");
            if constexpr (detail::symv_dispatches_blas_v<void, M, V, V>)
                REQUIRE(std::fabs(double(u(i)) - double(w(i))) <= 1e-4 * (1 + std::fabs(double(w(i)))));
            else
                REQUIRE(identical(u(i), w(i)));
        }
    }
}

TEMPLATE_TEST_CASE("syrk and syr2k row loops are bit-identical to the old loops", "[operation][724]",
                   float, double, long double) {
    using M = rm<TestType>;
    const M A = filled<M>(M_, K_, 10);
    const M B = filled<M>(M_, K_, 11);
    M C0(M_, M_);
    for (std::size_t i = 0; i < M_; ++i)
        for (std::size_t j = 0; j < M_; ++j) C0(i, j) = TestType(val(12, std::max(i, j), std::min(i, j)));
    for (const TestType beta : {TestType(0), TestType(-0.5)}) {
        CAPTURE(static_cast<double>(beta));
        M X = C0, Y = C0;
        syrk(TestType(0.875), A, beta, X);
        old_syrk(TestType(0.875), A, beta, Y);
        require_same<detail::syrk_dispatches_blas_v<void, M, M>>(X, Y, M_, M_);
        X = C0; Y = C0;
        syr2k(TestType(0.875), A, B, beta, X);
        old_syr2k(TestType(0.875), A, B, beta, Y);
        require_same<detail::syr2k_dispatches_blas_v<void, M, M, M>>(X, Y, M_, M_);
    }
}

TEST_CASE("row loops: alpha == 0 and empty shapes", "[operation][724]") {
    using M = rm<long double>;   // never BLAS: the row loops in every build
    const M A = triangular<M>(4, 13);
    M B = filled<M>(4, 3, 14);
    trmm(0.0L, A, B, false);
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 3; ++j) REQUIRE(B(i, j) == 0.0L);

    M E(0, 0), F(0, 5), G(4, 0);
    trsm(1.0L, E, F, true);
    trmm(1.0L, E, F, true);
    trsm(1.0L, A, G, false);
    M C(4, 0);
    symm(1.0L, A, G, 0.0L, C);
    M S(0, 0), Z(0, 3);
    syrk(1.0L, Z, 0.0L, S);
    syr2k(1.0L, Z, Z, 0.0L, S);
    vec::dense_vector<long double> x(0), y(0);
    symv(1.0L, E, x, 0.0L, y);
    SUCCEED("no out-of-range access on empty shapes");
}
