// gemmt and congruence (#656). The generic path is checked bit for bit against
// mult_generic, the reference GEMM loop, on the triangle it forms; every path
// (generic, and ?gemmt where the BLAS has it) is checked against a long-double
// reference over all operand orientations, both triangles and both fills.
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <vector>

#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/mat/parameter.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/tag/orientation.hpp>
#include <mtl/operation/gemmt.hpp>
#include <mtl/operation/mult.hpp>
#include <mtl/operation/trans.hpp>

using namespace mtl;

namespace {

/// A class-type scalar that rounds to float: not arithmetic, so it never
/// reaches BLAS or the native blocked GEMM, and its sums round at every step,
/// so agreeing bit for bit is not an accident of exact arithmetic.
struct fl {
    float v = 0.0f;
    fl() = default;
    constexpr fl(double d) : v(static_cast<float>(d)) {}
    friend fl operator+(fl a, fl b) { return double(a.v + b.v); }
    friend fl operator-(fl a, fl b) { return double(a.v - b.v); }
    friend fl operator*(fl a, fl b) { return double(a.v * b.v); }
    friend fl operator/(fl a, fl b) { return double(a.v / b.v); }
    friend fl operator-(fl a) { return double(-a.v); }
    fl& operator+=(fl o) { return *this = *this + o; }
    fl& operator-=(fl o) { return *this = *this - o; }
    fl& operator*=(fl o) { return *this = *this * o; }
    fl& operator/=(fl o) { return *this = *this / o; }
    friend bool operator==(fl a, fl b) { return a.v == b.v; }
    friend auto operator<=>(fl a, fl b) { return a.v <=> b.v; }
};

double dbl(double x) { return x; }
double dbl(float x) { return x; }
double dbl(fl x) { return x.v; }

template <typename T, typename O = tag::row_major>
using dmat = mat::dense2D<T, mat::parameters<O>>;

template <typename M>
void fill_random(M& A, std::mt19937_64& g) {
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    for (std::size_t i = 0; i < A.num_rows(); ++i)
        for (std::size_t j = 0; j < A.num_cols(); ++j)
            A(i, j) = typename M::value_type(u(g));
}

/// A random symmetric n x n matrix.
template <typename M>
void fill_symmetric(M& P, std::mt19937_64& g) {
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    for (std::size_t i = 0; i < P.num_rows(); ++i)
        for (std::size_t j = 0; j <= i; ++j)
            P(i, j) = P(j, i) = typename M::value_type(u(g));
}

/// A sparse-looking dense matrix: identity blocks and a few strips, as the
/// Jacobians in a covariance update are. Zero entries are exact zeros.
template <typename M>
void fill_jacobian(M& A, std::mt19937_64& g) {
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    for (std::size_t i = 0; i < A.num_rows(); ++i)
        for (std::size_t j = 0; j < A.num_cols(); ++j)
            A(i, j) = typename M::value_type(i == j ? 1.0 : ((i + 2 * j) % 5 == 0 ? u(g) : 0.0));
}

template <typename M>
mat::compressed2D<typename M::value_type> to_crs(const M& A) {
    mat::compressed2D<typename M::value_type> S(A.num_rows(), A.num_cols());
    {
        mat::inserter<mat::compressed2D<typename M::value_type>> ins(S);
        for (std::size_t i = 0; i < A.num_rows(); ++i)
            for (std::size_t j = 0; j < A.num_cols(); ++j)
                if (dbl(A(i, j)) != 0.0)
                    ins[i][j] << A(i, j);
    }
    return S;
}

template <typename M1, typename M2>
bool same_bits(const M1& X, const M2& Y) {
    for (std::size_t i = 0; i < X.num_rows(); ++i)
        for (std::size_t j = 0; j < X.num_cols(); ++j)
            if (!(X(i, j) == Y(i, j)))
                return false;
    return true;
}

template <typename M>
bool exactly_symmetric(const M& C) {
    for (std::size_t i = 0; i < C.num_rows(); ++i)
        for (std::size_t j = 0; j < i; ++j)
            if (!(C(i, j) == C(j, i)))
                return false;
    return true;
}

bool in_triangle(uplo t, std::size_t i, std::size_t j) {
    return t == uplo::lower ? j <= i : j >= i;
}

/// max |C - (alpha*A*B + beta*C0)| over the triangle, against a long-double
/// sum, divided by (|alpha| * sum_k |A(i,k) B(k,j)| + |beta C0(i,j)|).
template <typename MA, typename MB, typename MC, typename M0>
double triangle_error(double alpha, const MA& A, const MB& B, double beta,
                      const M0& C0, const MC& C, uplo t) {
    double worst = 0.0;
    const std::size_t n = C.num_rows(), k = A.num_cols();
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            if (!in_triangle(t, i, j))
                continue;
            long double s = 0.0L, m = 0.0L;
            for (std::size_t l = 0; l < k; ++l) {
                const long double p = static_cast<long double>(dbl(A(i, l))) * dbl(B(l, j));
                s += p;
                m += std::fabs(p);
            }
            const long double ref = alpha * s + (beta == 0.0 ? 0.0L : beta * static_cast<long double>(dbl(C0(i, j))));
            const long double scale = std::fabs(alpha) * m + std::fabs(beta * dbl(C0(i, j))) + 1e-300L;
            worst = std::max(worst, static_cast<double>(std::fabs(dbl(C(i, j)) - ref) / scale));
        }
    return worst;
}

} // namespace

TEST_CASE("gemmt: triangle bit-identical to mult_generic", "[operation][gemmt]") {
    std::mt19937_64 g(656);
    for (std::size_t n : {1u, 2u, 7u, 16u})
        for (std::size_t k : {1u, 5u, 13u}) {
            dmat<fl> A(n, k), B(k, n), R(n, n);
            fill_random(A, g);
            fill_random(B, g);
            detail::mult_generic(A, B, R);
            for (uplo t : {uplo::lower, uplo::upper}) {
                dmat<fl> C(n, n);
                fill_random(C, g);   // beta == 0: must not be read
                gemmt(fl(1.0), A, B, fl(0.0), C, t, gemmt_fill::keep);
                for (std::size_t i = 0; i < n; ++i)
                    for (std::size_t j = 0; j < n; ++j)
                        if (in_triangle(t, i, j))
                            REQUIRE(C(i, j) == R(i, j));
            }
        }
    // A column-major B takes the dot-product loop rather than the i-k-j one.
    for (std::size_t n : {1u, 6u, 11u}) {
        dmat<fl> A(n, 7), R(n, n), C(n, n);
        dmat<fl, tag::col_major> B(7, n);
        fill_random(A, g);
        fill_random(B, g);
        detail::mult_generic(A, B, R);
        for (uplo t : {uplo::lower, uplo::upper}) {
            gemmt(fl(1.0), A, B, fl(0.0), C, t);
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t j = 0; j < n; ++j)
                    if (in_triangle(t, i, j))
                        REQUIRE(C(i, j) == R(i, j));
        }
    }
#if !(defined(MTL5_HAS_BLAS) && defined(MTL5_BLAS_HAS_GEMMT))
    // double takes the generic path when no ?gemmt is linked.
    dmat<double> A(9, 6), B(6, 9), R(9, 9), C(9, 9);
    fill_random(A, g);
    fill_random(B, g);
    detail::mult_generic(A, B, R);
    gemmt(1.0, A, B, 0.0, C, uplo::lower, gemmt_fill::keep);
    for (std::size_t i = 0; i < 9; ++i)
        for (std::size_t j = 0; j <= i; ++j)
            REQUIRE(C(i, j) == R(i, j));
#endif
}

TEST_CASE("gemmt: accumulator triangle bit-identical to mult_generic<Acc>", "[operation][gemmt]") {
    std::mt19937_64 g(6561);
    dmat<float> A(8, 11), B(11, 8), R(8, 8), C(8, 8);
    fill_random(A, g);
    fill_random(B, g);
    detail::mult_generic<double>(A, B, R);
    gemmt<double>(1.0f, A, B, 0.0f, C, uplo::upper);
    for (std::size_t i = 0; i < 8; ++i)
        for (std::size_t j = i; j < 8; ++j)
            REQUIRE(C(i, j) == R(i, j));
    REQUIRE(exactly_symmetric(C));
}

TEST_CASE("gemmt: every orientation, both triangles, both fills", "[operation][gemmt]") {
    std::mt19937_64 g(65);
    const std::size_t n = 6, k = 4;
    const double eps = std::numeric_limits<double>::epsilon();
    auto run = [&](const auto& A, const auto& B, auto C, const char* label) {
        INFO(label);
        auto C0 = C;
        for (uplo t : {uplo::lower, uplo::upper})
            for (double beta : {0.0, 0.5}) {
                // mirror: exactly symmetric, triangle correct
                C = C0;
                gemmt(1.5, A, B, beta, C, t);
                REQUIRE(exactly_symmetric(C));
                REQUIRE(triangle_error(1.5, A, B, beta, C0, C, t) <= 4 * k * eps);
                // keep: the other triangle untouched
                C = C0;
                gemmt(1.5, A, B, beta, C, t, gemmt_fill::keep);
                REQUIRE(triangle_error(1.5, A, B, beta, C0, C, t) <= 4 * k * eps);
                for (std::size_t i = 0; i < n; ++i)
                    for (std::size_t j = 0; j < n; ++j)
                        if (!in_triangle(t, i, j))
                            REQUIRE(C(i, j) == C0(i, j));
            }
    };
    dmat<double, tag::row_major> Ar(n, k), Br(k, n), Cr(n, n), Atr(k, n), Btr(n, k);
    dmat<double, tag::col_major> Ac(n, k), Bc(k, n), Cc(n, n), Atc(k, n), Btc(n, k);
    fill_random(Ar, g); fill_random(Br, g); fill_random(Cr, g);
    fill_random(Atr, g); fill_random(Btr, g);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < k; ++j) {
            Ac(i, j) = Ar(i, j); Bc(j, i) = Br(j, i);
            Atc(j, i) = Atr(j, i); Btc(i, j) = Btr(i, j);
        }
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            Cc(i, j) = Cr(i, j);

    run(Ar, Br, Cr, "row row -> row");
    run(Ac, Bc, Cc, "col col -> col");
    run(Ar, Bc, Cc, "row col -> col");
    run(Ac, Br, Cr, "col row -> row");
    run(Ar, trans(Btr), Cr, "row trans(row) -> row");
    run(Ac, trans(Btc), Cc, "col trans(col) -> col");
    run(trans(Atr), Bc, Cr, "trans(row) col -> row");
    run(trans(Atc), trans(Btr), Cc, "trans(col) trans(row) -> col");
}

TEST_CASE("gemmt: alpha == 0 and beta == 0 follow BLAS", "[operation][gemmt]") {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (int orient = 0; orient < 2; ++orient) {
        auto check = [&](auto A, auto B, auto C) {
            for (std::size_t i = 0; i < 3; ++i)
                for (std::size_t j = 0; j < 3; ++j) {
                    A(i, j) = 1.0 + double(i);
                    B(i, j) = 2.0 - double(j);
                    C(i, j) = nan;
                }
            gemmt(1.0, A, B, 0.0, C);   // C is not read
            for (std::size_t i = 0; i < 3; ++i)
                for (std::size_t j = 0; j < 3; ++j)
                    REQUIRE(std::isfinite(C(i, j)));
            for (std::size_t i = 0; i < 3; ++i)
                for (std::size_t j = 0; j < 3; ++j)
                    A(i, j) = nan;
            gemmt(0.0, A, B, 2.0, C);   // A and B are not read
            REQUIRE(std::isfinite(C(2, 1)));
            gemmt(0.0, A, B, 0.0, C);
            for (std::size_t i = 0; i < 3; ++i)
                for (std::size_t j = 0; j < 3; ++j)
                    REQUIRE(C(i, j) == 0.0);
        };
        if (orient == 0)
            check(dmat<double>(3, 3), dmat<double>(3, 3), dmat<double>(3, 3));
        else
            check(dmat<double, tag::col_major>(3, 3), dmat<double, tag::col_major>(3, 3),
                  dmat<double, tag::col_major>(3, 3));
    }
}

TEST_CASE("gemmt: edge cases", "[operation][gemmt]") {
    dmat<double> A0(0, 3), B0(3, 0), C0(0, 0);
    gemmt(1.0, A0, B0, 0.0, C0);   // n == 0
    dmat<double> A1(1, 1), B1(1, 1), C1(1, 1);
    A1(0, 0) = 3.0; B1(0, 0) = 4.0; C1(0, 0) = 1.0;
    gemmt(2.0, A1, B1, 1.0, C1);
    REQUIRE(C1(0, 0) == 25.0);
    // k == 0: C = beta * C
    for (int orient = 0; orient < 2; ++orient) {
        auto check = [](auto A, auto B, auto C) {
            for (std::size_t i = 0; i < 2; ++i)
                for (std::size_t j = 0; j < 2; ++j)
                    C(i, j) = 1.0;
            gemmt(1.0, A, B, 3.0, C);
            for (std::size_t i = 0; i < 2; ++i)
                for (std::size_t j = 0; j < 2; ++j)
                    REQUIRE(C(i, j) == 3.0);
        };
        if (orient == 0)
            check(dmat<double>(2, 0), dmat<double>(0, 2), dmat<double>(2, 2));
        else
            check(dmat<double, tag::col_major>(2, 0), dmat<double, tag::col_major>(0, 2),
                  dmat<double, tag::col_major>(2, 2));
    }
}

TEST_CASE("congruence: generic path bit-identical to mult(mult(A, P), trans(A))", "[operation][gemmt][congruence]") {
    std::mt19937_64 g(4448);
    for (std::size_t m : {1u, 3u, 9u})
        for (std::size_t n : {1u, 4u, 15u}) {
            dmat<fl> A(m, n), P(n, n), T(m, n), R(m, m), C(m, m);
            fill_random(A, g);
            fill_symmetric(P, g);
            detail::mult_generic(A, P, T);
            detail::mult_generic(T, trans(A), R);
            congruence(A, P, C);
            REQUIRE(exactly_symmetric(C));
            for (std::size_t i = 0; i < m; ++i)
                for (std::size_t j = 0; j <= i; ++j)
                    REQUIRE(C(i, j) == R(i, j));
        }
}

TEST_CASE("congruence: CRS and dense A agree bit for bit", "[operation][gemmt][congruence]") {
    std::mt19937_64 g(451);
    auto run = [&](auto tag_value) {
        using T = decltype(tag_value);
        for (std::size_t m : {1u, 6u, 15u})
            for (std::size_t n : {2u, 15u, 21u}) {
                dmat<T> A(m, n), P(n, n), Cd(m, m), Cs(m, m);
                fill_jacobian(A, g);
                fill_symmetric(P, g);
                const auto S = to_crs(A);
                congruence(A, P, Cd);
                congruence(S, P, Cs);
                REQUIRE(same_bits(Cd, Cs));
                REQUIRE(exactly_symmetric(Cs));
                // alpha/beta form: C = 2 A P A^T + 0.5 C
                dmat<T> Dd(m, m), Ds(m, m);
                fill_symmetric(Dd, g);
                for (std::size_t i = 0; i < m; ++i)
                    for (std::size_t j = 0; j < m; ++j)
                        Ds(i, j) = Dd(i, j);
                congruence(T(2.0), A, P, T(0.5), Dd);
                congruence(T(2.0), S, P, T(0.5), Ds);
                REQUIRE(same_bits(Dd, Ds));
            }
    };
    run(fl{});
#if !defined(MTL5_HAS_BLAS) && !defined(MTL5_NATIVE_FAST_GEMM)
    // With BLAS the dense side's A*P is a dgemm, and with ?gemmt its second
    // product is too; either sums in the vendor's order, not the CRS kernel's.
    // With MTL5_NATIVE_FAST_GEMM it is gemm_blocked, whose micro-kernel fuses
    // each multiply-add where the CRS kernel rounds the product first.
    run(double{});
#endif
}

TEST_CASE("congruence: diagonal middle term", "[operation][gemmt][congruence]") {
    std::mt19937_64 g(1989);
    const std::size_t m = 7, n = 5;
    dmat<fl> K(m, n), D(n, n), Cv(m, m), Cm(m, m);
    vec::dense_vector<fl> r(n);
    fill_random(K, g);
    std::uniform_real_distribution<double> u(0.1, 2.0);
    for (std::size_t i = 0; i < n; ++i) {
        r(i) = fl(u(g));
        for (std::size_t j = 0; j < n; ++j)
            D(i, j) = fl(i == j ? double(r(i).v) : 0.0);
    }
    congruence(K, r, Cv);
    congruence(K, D, Cm);
    REQUIRE(same_bits(Cv, Cm));
    REQUIRE(exactly_symmetric(Cv));

    // CRS K, and the Joseph-form accumulation C += K diag(r) K^T
    dmat<fl> J(m, n), Cd(m, m), Cs(m, m);
    fill_jacobian(J, g);
    const auto S = to_crs(J);
    fill_symmetric(Cd, g);
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < m; ++j)
            Cs(i, j) = Cd(i, j);
    congruence(fl(1.0), J, r, fl(1.0), Cd);
    congruence(fl(1.0), S, r, fl(1.0), Cs);
    REQUIRE(same_bits(Cd, Cs));
}

TEST_CASE("congruence: column-major double against a reference", "[operation][gemmt][congruence]") {
    std::mt19937_64 g(200);
    const std::size_t m = 12, n = 9;
    dmat<double, tag::col_major> A(m, n), P(n, n), C(m, m), T(m, n);
    fill_random(A, g);
    fill_symmetric(P, g);
    congruence(A, P, C);
    REQUIRE(exactly_symmetric(C));
    detail::mult_generic(A, P, T);
    const double eps = std::numeric_limits<double>::epsilon();
    REQUIRE(triangle_error(1.0, T, trans(A), 0.0, C, C, uplo::lower) <= 8 * n * eps);
}

TEST_CASE("congruence: empty", "[operation][gemmt][congruence]") {
    dmat<double> A(0, 3), P(3, 3), C(0, 0);
    congruence(A, P, C);
    dmat<double> B(2, 0), Q(0, 0), D(2, 2);
    D(0, 0) = D(0, 1) = D(1, 0) = D(1, 1) = 7.0;
    congruence(B, Q, D);   // A*P*A^T of an empty inner dimension is zero
    REQUIRE(D(0, 0) == 0.0);
    REQUIRE(D(1, 0) == 0.0);
    REQUIRE(D(0, 1) == 0.0);
}
