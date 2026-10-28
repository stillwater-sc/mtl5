// MTL5 -- Tests for the Sylvester solvers sylvester_triangular() and sylvester() (#650)
//
// Each test builds C from a known X0, solves, and checks the residual
// ||A X + sgn X B - C|| relative to (||A|| + ||B||) ||X|| + ||C||, which a
// backward stable solver keeps at a small multiple of eps. Where A and -sgn B
// are well separated the forward error against X0 is checked as well. With
// LAPACK the default float/double calls take trsyl and the Accumulator calls
// the in-house path, so both are checked.
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <complex>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>
#include <mtl/mtl.hpp>

using namespace mtl;

namespace {

/// A class-typed real with no std::numeric_limits specialization and only the
/// operations the solvers may use: + - * /, comparisons, and abs and sqrt by ADL.
struct unlimited {
    double v = 0.0;
    unlimited() = default;
    constexpr unlimited(double d) : v(d) {}
    friend unlimited operator+(unlimited a, unlimited b) { return a.v + b.v; }
    friend unlimited operator-(unlimited a, unlimited b) { return a.v - b.v; }
    friend unlimited operator*(unlimited a, unlimited b) { return a.v * b.v; }
    friend unlimited operator/(unlimited a, unlimited b) { return a.v / b.v; }
    friend unlimited operator-(unlimited a) { return -a.v; }
    unlimited& operator+=(unlimited o) { v += o.v; return *this; }
    unlimited& operator-=(unlimited o) { v -= o.v; return *this; }
    unlimited& operator*=(unlimited o) { v *= o.v; return *this; }
    unlimited& operator/=(unlimited o) { v /= o.v; return *this; }
    friend bool operator==(unlimited a, unlimited b) { return a.v == b.v; }
    friend auto operator<=>(unlimited a, unlimited b) { return a.v <=> b.v; }
    friend unlimited sqrt(unlimited a) { return std::sqrt(a.v); }
    friend unlimited abs(unlimited a) { return std::abs(a.v); }
};
static_assert(!std::numeric_limits<unlimited>::is_specialized);

long double ld(float x) { return x; }
long double ld(double x) { return x; }
long double ld(long double x) { return x; }
long double ld(unlimited x) { return x.v; }
template <typename R>
std::complex<long double> ldc(const std::complex<R>& z) { return {ld(z.real()), ld(z.imag())}; }
template <typename T>
std::complex<long double> ldc(const T& x) { return {ld(x), 0.0L}; }

template <typename T>
long double eps_of() {
    return ld(detail::machine_epsilon<magnitude_t<T>>());
}

template <typename T>
mat::dense2D<T> random_matrix(std::size_t m, std::size_t n, std::mt19937_64& g) {
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    mat::dense2D<T> A(m, n);
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            if constexpr (is_complex_v<T>) A(i, j) = T(u(g), u(g));
            else                           A(i, j) = T(u(g));
        }
    return A;
}

/// The real Schur factor T of a random matrix shifted by `shift` on the
/// diagonal, so that the spectra of A and -sgn B can be kept apart.
template <typename T>
mat::dense2D<T> quasi_triangular(std::size_t n, double shift, std::mt19937_64& g) {
    auto A = random_matrix<T>(n, n, g);
    for (std::size_t i = 0; i < n; ++i) A(i, i) = A(i, i) + T(shift);
    return schur(A).T;
}

/// m x n product A X + sgn X B, in the element type.
template <typename T>
mat::dense2D<T> apply(const mat::dense2D<T>& A, const mat::dense2D<T>& B, const mat::dense2D<T>& X,
                      sylvester_sign sign) {
    const std::size_t m = X.num_rows(), n = X.num_cols();
    mat::dense2D<T> C(m, n);
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            T s(0);
            for (std::size_t k = 0; k < m; ++k) s += A(i, k) * X(k, j);
            T t(0);
            for (std::size_t k = 0; k < n; ++k) t += X(i, k) * B(k, j);
            C(i, j) = sign == sylvester_sign::plus ? s + t : s - t;
        }
    return C;
}

template <typename T>
long double fro(const mat::dense2D<T>& A) {
    long double s = 0;
    for (std::size_t i = 0; i < A.num_rows(); ++i)
        for (std::size_t j = 0; j < A.num_cols(); ++j) s += std::norm(ldc(A(i, j)));
    return std::sqrt(s);
}

/// ||A X + sgn X B - C||_F / ((||A||_F + ||B||_F) ||X||_F + ||C||_F), in eps,
/// formed in long double.
template <typename T>
long double residual(const mat::dense2D<T>& A, const mat::dense2D<T>& B, const mat::dense2D<T>& C,
                     const mat::dense2D<T>& X, sylvester_sign sign) {
    const std::size_t m = X.num_rows(), n = X.num_cols();
    const long double sg = sign == sylvester_sign::plus ? 1.0L : -1.0L;
    long double r = 0;
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            std::complex<long double> s = -ldc(C(i, j));
            for (std::size_t k = 0; k < m; ++k) s += ldc(A(i, k)) * ldc(X(k, j));
            for (std::size_t k = 0; k < n; ++k) s += sg * ldc(X(i, k)) * ldc(B(k, j));
            r += std::norm(s);
        }
    const long double d = (fro(A) + fro(B)) * fro(X) + fro(C);
    return d == 0 ? 0 : std::sqrt(r) / d / eps_of<T>();
}

template <typename T>
long double forward(const mat::dense2D<T>& X, const mat::dense2D<T>& X0) {
    long double r = 0;
    for (std::size_t i = 0; i < X.num_rows(); ++i)
        for (std::size_t j = 0; j < X.num_cols(); ++j) r += std::norm(ldc(X(i, j)) - ldc(X0(i, j)));
    return std::sqrt(r) / fro(X0);
}

/// Solve quasi-triangular systems of every size pairing up to 7 x 7, with and
/// without an Accumulator. A's eigenvalues sit near +3 and -sgn B's near -3,
/// so the problem is well conditioned and X0 is recovered.
template <typename T, typename Accumulator>
void check_triangular_family(std::uint64_t seed) {
    std::mt19937_64 g(seed);
    long double worst = 0;
    int blocks2 = 0;
    for (auto sign : {sylvester_sign::plus, sylvester_sign::minus})
        for (std::size_t m = 1; m <= 7; ++m)
            for (std::size_t n = 1; n <= 7; ++n) {
                auto A = quasi_triangular<T>(m, 3.0, g);
                auto B = quasi_triangular<T>(n, sign == sylvester_sign::plus ? 3.0 : -3.0, g);
                for (std::size_t k = 0; k + 1 < m; ++k) blocks2 += !(A(k + 1, k) == T(0));
                auto X0 = random_matrix<T>(m, n, g);
                auto C  = apply(A, B, X0, sign);
                auto [X, perturbed] = sylvester_triangular<Accumulator>(A, B, C, sign);
                INFO("m " << m << ", n " << n << ", sign " << int(sign));
                CHECK_FALSE(perturbed);
                const long double res = residual(A, B, C, X, sign);
                CHECK(res <= 10.0L * (m + n));
                CHECK(forward(X, X0) <= 1e3L * (m + n) * eps_of<T>());
                if (res > worst) worst = res;
            }
    CHECK(blocks2 > 10);   // the 2x2 block cases were exercised
    INFO("worst residual " << worst << " eps");
    CHECK(worst <= 20.0L);
}

} // namespace

TEST_CASE("sylvester_triangular: real quasi-triangular A and B", "[operation][sylvester]") {
    check_triangular_family<float, void>(1);
    check_triangular_family<double, void>(2);
    check_triangular_family<long double, void>(3);
    check_triangular_family<double, double>(4);           // in-house path with LAPACK
    check_triangular_family<float, double>(5);            // wider accumulator
    check_triangular_family<unlimited, void>(6);          // no numeric_limits
}

TEST_CASE("sylvester_triangular: every 1x1/2x2 block pairing", "[operation][sylvester]") {
    // 1x1 blocks, and 2x2 blocks holding complex pairs: in LAPACK's standard
    // form (equal diagonal) and not (A2n, B2n, eigenvalues 2 +- i sqrt(7) and
    // -1 +- i sqrt(2)), which trsyl and the in-house path both accept.
    const mat::dense2D<double> A2{{1.0, 4.0}, {-1.0, 1.0}};
    const mat::dense2D<double> B2{{-2.0, 0.5}, {-3.0, -2.0}};
    const mat::dense2D<double> A2n{{1.0, 4.0}, {-2.0, 3.0}};
    const mat::dense2D<double> B2n{{0.0, 1.0}, {-3.0, -2.0}};
    const mat::dense2D<double> A1{{4.0}};
    const mat::dense2D<double> B1{{0.5}};
    std::mt19937_64 g(7);
    for (const auto* A : {&A1, &A2, &A2n})
        for (const auto* B : {&B1, &B2, &B2n}) {
            auto X0 = random_matrix<double>(A->num_rows(), B->num_rows(), g);
            auto C  = apply(*A, *B, X0, sylvester_sign::plus);
            for (bool inhouse : {false, true}) {
                auto [X, perturbed] = inhouse ? sylvester_triangular<double>(*A, *B, C)
                                              : sylvester_triangular(*A, *B, C);
                INFO(A->num_rows() << "x" << B->num_rows() << (inhouse ? " in-house" : ""));
                CHECK_FALSE(perturbed);
                CHECK(residual(*A, *B, C, X, sylvester_sign::plus) <= 10.0L);
                CHECK(forward(X, X0) <= 1e-14L);
            }
        }
}

TEST_CASE("sylvester_triangular: complex upper triangular A and B", "[operation][sylvester]") {
    using C = std::complex<double>;
    std::mt19937_64 g(8);
    for (auto sign : {sylvester_sign::plus, sylvester_sign::minus})
        for (std::size_t m : {1u, 4u, 9u})
            for (std::size_t n : {1u, 5u, 8u}) {
                // rsf2csf's triangular factors of shifted real matrices.
                auto Ar = random_matrix<double>(m, m, g);
                auto Br = random_matrix<double>(n, n, g);
                for (std::size_t i = 0; i < m; ++i) Ar(i, i) += 3.0;
                for (std::size_t i = 0; i < n; ++i) Br(i, i) += sign == sylvester_sign::plus ? 3.0 : -3.0;
                auto [QA, TA] = schur(Ar);
                auto [QB, TB] = schur(Br);
                auto A = rsf2csf(QA, TA).T;
                auto B = rsf2csf(QB, TB).T;
                auto X0 = random_matrix<C>(m, n, g);
                auto Cm = apply(A, B, X0, sign);
                for (bool inhouse : {false, true}) {
                    auto [X, perturbed] = inhouse ? sylvester_triangular<C>(A, B, Cm, sign)
                                                  : sylvester_triangular(A, B, Cm, sign);
                    INFO("m " << m << ", n " << n << (inhouse ? " in-house" : ""));
                    CHECK_FALSE(perturbed);
                    CHECK(residual(A, B, Cm, X, sign) <= 10.0L * (m + n));
                    CHECK(forward(X, X0) <= 1e-12L);
                }
            }
}

TEST_CASE("sylvester_triangular: singular and nearly singular equations", "[operation][sylvester]") {
    // A and -B share the eigenvalue 1: A X + X B = C has no unique solution.
    const mat::dense2D<double> A{{1.0, 2.0}, {0.0, 3.0}};
    const mat::dense2D<double> B{{-1.0, 1.0}, {0.0, 5.0}};
    const mat::dense2D<double> C{{1.0, 1.0}, {1.0, 1.0}};
    for (bool inhouse : {false, true}) {
        auto [X, perturbed] = inhouse ? sylvester_triangular<double>(A, B, C) : sylvester_triangular(A, B, C);
        INFO((inhouse ? "in-house" : "default"));
        CHECK(perturbed);
        for (std::size_t i = 0; i < 2; ++i)
            for (std::size_t j = 0; j < 2; ++j) CHECK(std::isfinite(X(i, j)));
    }
    // The same A with sylvester_sign::minus and B = A: A X - X A = C.
    for (bool inhouse : {false, true}) {
        auto r = inhouse ? sylvester_triangular<double>(A, A, C, sylvester_sign::minus)
                         : sylvester_triangular(A, A, C, sylvester_sign::minus);
        CHECK(r.perturbed);
    }
    // A 2x2 block of A (eigenvalues +-i) against one of -B (also +-i).
    const mat::dense2D<double> Ac{{0.0, 1.0}, {-1.0, 0.0}};
    auto r = sylvester_triangular<double>(Ac, Ac, C, sylvester_sign::minus);
    CHECK(r.perturbed);
    // Well apart: no flag.
    CHECK_FALSE(sylvester_triangular<double>(A, A, C).perturbed);
}

TEST_CASE("sylvester_triangular: shapes and errors", "[operation][sylvester]") {
    const mat::dense2D<double> A{{1.0, 2.0}, {0.0, 3.0}};
    auto empty = sylvester_triangular(A, mat::dense2D<double>(0, 0), mat::dense2D<double>(2, 0));
    CHECK(empty.X.num_rows() == 2);
    CHECK(empty.X.num_cols() == 0);
    auto empty2 = sylvester_triangular(mat::dense2D<double>(0, 0), A, mat::dense2D<double>(0, 2));
    CHECK(empty2.X.num_rows() == 0);

    // Column-major input.
    mat::dense2D<double, mat::parameters<tag::col_major>> Acm(2, 2), Ccm(2, 3);
    mat::dense2D<double> Bm{{2.0, 1.0, 0.0}, {0.0, 1.0, 1.0}, {0.0, -1.0, 1.0}};
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 2; ++j) Acm(i, j) = A(i, j);
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 3; ++j) Ccm(i, j) = double(i + 2 * j + 1);
    auto [X, perturbed] = sylvester_triangular(Acm, Bm, Ccm);
    CHECK_FALSE(perturbed);
    mat::dense2D<double> Ar(A), Cr(2, 3);
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 3; ++j) Cr(i, j) = Ccm(i, j);
    CHECK(residual(Ar, Bm, Cr, X, sylvester_sign::plus) <= 10.0L);

    const mat::dense2D<double> lower{{1.0, 0.0, 0.0}, {1.0, 1.0, 0.0}, {1.0, 1.0, 1.0}};
    const mat::dense2D<double> adjacent{{1.0, 1.0, 1.0}, {1.0, 1.0, 1.0}, {0.0, 1.0, 1.0}};
    const mat::dense2D<double> I3{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
    const mat::dense2D<double> C3(3, 3);
    REQUIRE_THROWS_AS(sylvester_triangular(lower, I3, C3), std::invalid_argument);
    REQUIRE_THROWS_AS(sylvester_triangular(I3, adjacent, C3), std::invalid_argument);
    REQUIRE_THROWS_AS(sylvester_triangular(I3, I3, mat::dense2D<double>(3, 2)), std::invalid_argument);
    REQUIRE_THROWS_AS(sylvester_triangular(mat::dense2D<double>(2, 3), I3, C3), std::invalid_argument);
}

TEST_CASE("sylvester: general real A and B (Bartels-Stewart)", "[operation][sylvester]") {
    std::mt19937_64 g(9);
    for (auto sign : {sylvester_sign::plus, sylvester_sign::minus})
        for (std::size_t m : {1u, 3u, 8u})
            for (std::size_t n : {2u, 6u, 11u}) {
                auto A = random_matrix<double>(m, m, g);
                auto B = random_matrix<double>(n, n, g);
                for (std::size_t i = 0; i < m; ++i) A(i, i) += 3.0;
                for (std::size_t i = 0; i < n; ++i) B(i, i) += sign == sylvester_sign::plus ? 3.0 : -3.0;
                auto X0 = random_matrix<double>(m, n, g);
                auto C  = apply(A, B, X0, sign);
                for (bool inhouse : {false, true}) {
                    auto [X, perturbed] = inhouse ? sylvester<double>(A, B, C, sign) : sylvester(A, B, C, sign);
                    INFO("m " << m << ", n " << n << ", sign " << int(sign) << (inhouse ? " in-house" : ""));
                    CHECK_FALSE(perturbed);
                    CHECK(residual(A, B, C, X, sign) <= 10.0L * (m + n));
                    CHECK(forward(X, X0) <= 1e-12L);
                }
            }

    // Generic element types: float and the type without numeric_limits.
    for (std::size_t n : {3u, 7u}) {
        auto Af = random_matrix<float>(n, n, g);
        auto Bf = random_matrix<float>(n, n, g);
        for (std::size_t i = 0; i < n; ++i) { Af(i, i) += 3.0f; Bf(i, i) += 3.0f; }
        auto Xf = random_matrix<float>(n, n, g);
        auto Cf = apply(Af, Bf, Xf, sylvester_sign::plus);
        CHECK(residual(Af, Bf, Cf, sylvester(Af, Bf, Cf).X, sylvester_sign::plus) <= 10.0L * n);

        mat::dense2D<unlimited> Au(n, n), Bu(n, n), Xu(n, n);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j) {
                Au(i, j) = unlimited(double(Af(i, j)));
                Bu(i, j) = unlimited(double(Bf(i, j)));
                Xu(i, j) = unlimited(double(Xf(i, j)));
            }
        auto Cu = apply(Au, Bu, Xu, sylvester_sign::plus);
        auto [X, perturbed] = sylvester(Au, Bu, Cu);
        CHECK_FALSE(perturbed);
        CHECK(residual(Au, Bu, Cu, X, sylvester_sign::plus) <= 10.0L * n);
        CHECK(forward(X, Xu) <= 1e-12L);
    }

    REQUIRE_THROWS_AS(sylvester(mat::dense2D<double>(2, 3), mat::dense2D<double>(2, 2),
                                mat::dense2D<double>(2, 2)),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(sylvester(mat::dense2D<double>(2, 2), mat::dense2D<double>(2, 2),
                                mat::dense2D<double>(3, 2)),
                      std::invalid_argument);
    auto e = sylvester(mat::dense2D<double>(0, 0), mat::dense2D<double>(2, 2), mat::dense2D<double>(0, 2));
    CHECK(e.X.num_rows() == 0);
}
