// MTL5 -- Tests for the real Schur decomposition schur() and rsf2csf() (#650)
//
// A Schur form is not unique, so the tests check its properties: the backward
// error ||A - Q T Q^T|| / ||A||, the orthogonality of Q, T's quasi-triangular
// shape with 2x2 blocks in standard form, and its eigenvalues. With LAPACK the
// default calls take gees and the Accumulator calls the in-house path, so
// both are checked.
#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <numbers>
#include <random>
#include <stdexcept>
#include <vector>
#include <mtl/mtl.hpp>

using namespace mtl;

namespace {

/// A class-typed real with no std::numeric_limits specialization and only the
/// operations schur() may use: + - * /, comparisons, and abs and sqrt by ADL.
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

template <typename T>
long double eps_of() {
    return ld(detail::machine_epsilon<T>());
}

struct quality {
    long double backward;    ///< ||A - Q T Q^T||_F / ||A||_F, in units of n eps
    long double orthogonal;  ///< ||Q^T Q - I||_F, in units of n eps
    bool shape;              ///< quasi-triangular, 2x2 blocks in standard form
};

/// Measured in the element type, normalized by max |a_ij| so that matrices
/// near overflow or underflow can be checked.
template <typename T>
quality check(const mat::dense2D<T>& A, const mat::dense2D<T>& Q, const mat::dense2D<T>& R) {
    using std::abs;
    const std::size_t n = A.num_rows();
    long double amax = 0;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) amax = std::max(amax, std::abs(ld(A(i, j))));
    if (amax == 0) amax = 1;
    long double na = 0, nb = 0, no = 0;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            T s = T(0), o = T(0);
            for (std::size_t k = 0; k < n; ++k) {
                T rq = T(0);
                for (std::size_t l = 0; l < n; ++l) rq = rq + R(k, l) * Q(j, l);
                s = s + Q(i, k) * rq;
                o = o + Q(k, i) * Q(k, j);
            }
            const long double d = ld(T(A(i, j) - s)) / amax;
            const long double a = ld(A(i, j)) / amax;
            const long double e = ld(T(o - T(i == j ? 1.0 : 0.0)));
            nb += d * d; na += a * a; no += e * e;
        }
    bool shape = true;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j + 1 < i; ++j)
            if (!(R(i, j) == T(0))) shape = false;
    for (std::size_t k = 0; k + 1 < n; ++k)
        if (!(R(k + 1, k) == T(0))) {
            if (k + 2 < n && !(R(k + 2, k + 1) == T(0))) shape = false;   // blocks do not overlap
            if (!(R(k, k) == R(k + 1, k + 1))) shape = false;              // equal diagonal
            if ((R(k, k + 1) < T(0)) == (R(k + 1, k) < T(0))) shape = false;  // b*c < 0
        }
    const long double un = (n ? n : 1) * eps_of<T>();
    return {na > 0 ? std::sqrt(nb / na) / un : std::sqrt(nb), std::sqrt(no) / un, shape};
}

/// Matrix families, n x n, with entries scaled by `scale`.
enum family { dense_random, nonnormal, graded, companion, jordan, cyclic, upper, diagonal, skew_tridiagonal };
constexpr family all_families[] = {dense_random, nonnormal, graded, companion, jordan,
                                   cyclic, upper, diagonal, skew_tridiagonal};

template <typename T>
mat::dense2D<T> make(family f, std::size_t n, std::mt19937_64& g, double scale = 1.0) {
    mat::dense2D<T> A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            const double v = double(g() >> 11) * 0x1.0p-53 - 0.5;
            double a = 0.0;
            switch (f) {
            case dense_random:  a = v; break;
            case nonnormal: a = (j + 1 >= i) ? v * (1.0 + 10.0 * (j > i)) : 0.0; break;
            case graded:    a = v * std::pow(10.0, 0.5 * (double(i) - double(j))); break;
            case companion: a = (i == 0) ? -double(j + 1) / double(n) : (i == j + 1 ? 1.0 : 0.0); break;
            case jordan:    a = (i == j) ? 2.0 : (j == i + 1 ? 1.0 : 0.0); break;
            case cyclic:    a = (i + 1 == j || (i + 1 == n && j == 0)) ? 1.0 : 0.0; break;
            case upper:     a = (j >= i) ? v : 0.0; break;
            case diagonal:  a = (i == j) ? 0.25 * double(i % 3) : 0.0; break;
            case skew_tridiagonal: a = (i == j + 1) ? 1.0 : (j == i + 1 ? -1.0 : 0.0); break;
            }
            A(i, j) = T(a * scale);
        }
    return A;
}

template <typename T, typename F>
void check_families(F&& schur_fn, std::size_t nmax, long double bound, double scale = 1.0) {
    std::mt19937_64 g(7);
    for (family f : all_families)
        for (std::size_t n = 1; n <= nmax; ++n) {
            auto A = make<T>(f, n, g, scale);
            auto [Q, R] = schur_fn(A);
            auto q = check(A, Q, R);
            INFO("family " << int(f) << ", n = " << n << ", scale " << scale);
            CHECK(q.backward <= bound);
            CHECK(q.orthogonal <= bound);
            CHECK(q.shape);
        }
}

/// Sort eigenvalues by real part, then imaginary part, treating real parts
/// within 1e-9 as equal so that a conjugate pair stays together.
void sort_eigs(std::vector<std::complex<long double>>& e) {
    std::sort(e.begin(), e.end(), [](auto x, auto y) {
        return std::abs(x.real() - y.real()) > 1e-9L ? x.real() < y.real() : x.imag() < y.imag();
    });
}

/// The eigenvalues read from T's blocks, sorted by sort_eigs.
template <typename T>
std::vector<std::complex<long double>> block_eigenvalues(const mat::dense2D<T>& R) {
    std::vector<std::complex<long double>> e;
    const std::size_t n = R.num_rows();
    for (std::size_t k = 0; k < n; ) {
        if (k + 1 < n && !(R(k + 1, k) == T(0))) {
            const long double a = ld(R(k, k));
            const long double im = std::sqrt(std::abs(ld(R(k, k + 1)))) * std::sqrt(std::abs(ld(R(k + 1, k))));
            e.emplace_back(a, im);
            e.emplace_back(a, -im);
            k += 2;
        } else {
            e.emplace_back(ld(R(k, k)), 0.0L);
            ++k;
        }
    }
    sort_eigs(e);
    return e;
}

} // namespace

TEST_CASE("schur: backward stable across matrix families", "[operation][schur]") {
    // Measured worst cases are 3.2 n eps (double, float, long double).
    auto def = [](const auto& A) { return schur(A); };
    check_families<double>(def, 16, 10);
    check_families<float>(def, 16, 10);
    check_families<long double>(def, 16, 10);
}

TEST_CASE("schur: the in-house path, through an Accumulator", "[operation][schur]") {
    // A non-void Accumulator takes the in-house path even where LAPACK is
    // available, so with LAPACK this covers what the default call does not.
    auto inhouse = [](const auto& A) { return schur<double>(A); };
    check_families<double>(inhouse, 16, 10);
    check_families<float>(inhouse, 16, 10);
}

TEST_CASE("schur: entries near overflow and underflow", "[operation][schur]") {
    // Without scaling, products of two entries overflow at 1e200 and underflow
    // at 1e-200, and the iteration does not converge.
    auto inhouse = [](const auto& A) { return schur<double>(A); };
    auto def = [](const auto& A) { return schur(A); };
    for (double scale : {1e200, 1e-200, 1e300}) {
        check_families<double>(inhouse, 10, 10, scale);
        check_families<double>(def, 10, 10, scale);
    }
}

TEST_CASE("schur: a scalar without numeric_limits", "[operation][schur]") {
    // The epsilon is measured, and only + - * /, comparisons, abs and sqrt are
    // used, so the generic path compiles and converges for such a type.
    REQUIRE(detail::machine_epsilon<unlimited>().v == std::numeric_limits<double>::epsilon());
    auto def = [](const auto& A) { return schur(A); };
    check_families<unlimited>(def, 12, 10);
}

TEST_CASE("schur: eigenvalues from the blocks", "[operation][schur]") {
    // The cyclic permutation of order n has the n-th roots of unity as its
    // eigenvalues; it is a classic case for exceptional shifts.
    std::mt19937_64 g(1);
    for (std::size_t n : {2u, 3u, 5u, 8u, 13u}) {
        auto A = make<double>(cyclic, n, g);
        for (auto [Q, R] : {schur(A), schur<double>(A)}) {
            auto e = block_eigenvalues(R);
            std::vector<std::complex<long double>> ref;
            for (std::size_t k = 0; k < n; ++k) {
                const long double t = 2 * std::numbers::pi_v<long double> * k / n;
                ref.emplace_back(std::cos(t), std::sin(t));
            }
            sort_eigs(ref);
            REQUIRE(e.size() == n);
            for (std::size_t k = 0; k < n; ++k) {
                INFO("n = " << n << ", k = " << k);
                CHECK(std::abs(e[k] - ref[k]) < 1e-13L);
            }
        }
    }

    // A real matrix and the general eigensolver agree.
    auto A = make<double>(dense_random, 9, g);
    auto e = block_eigenvalues(schur(A).T);
    auto ev = eigenvalue(A, 1e-14);
    std::vector<std::complex<long double>> ref;
    for (std::size_t k = 0; k < ev.size(); ++k) ref.emplace_back(ev(k).real(), ev(k).imag());
    sort_eigs(ref);
    for (std::size_t k = 0; k < ref.size(); ++k) CHECK(std::abs(e[k] - ref[k]) < 1e-12L);
}

TEST_CASE("schur: 0x0, 1x1, zero and 2x2 matrices", "[operation][schur]") {
    auto [Q0, T0] = schur(mat::dense2D<double>(0, 0));
    REQUIRE(Q0.num_rows() == 0);
    REQUIRE(T0.num_rows() == 0);

    mat::dense2D<double> one(1, 1);
    one(0, 0) = -2.5;
    auto [Q1, T1] = schur(one);
    REQUIRE(T1(0, 0) == -2.5);
    REQUIRE(std::abs(Q1(0, 0)) == 1.0);

    mat::dense2D<double> Z(4, 4);
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j) Z(i, j) = 0.0;
    for (auto [Q, R] : {schur(Z), schur<double>(Z)})
        for (std::size_t i = 0; i < 4; ++i)
            for (std::size_t j = 0; j < 4; ++j) {
                REQUIRE(R(i, j) == 0.0);
                REQUIRE(std::abs(Q(i, j)) == (i == j ? 1.0 : 0.0));
            }

    // Every branch of the 2x2 standardization (LAPACK dlanv2): c == 0, b == 0,
    // already standard, real distinct, complex, and a real double root reached
    // through the equal-diagonal branch.
    const double blocks[][4] = {
        {1, 2, 0, 3}, {1, 0, 2, 3}, {0, -1, 1, 0}, {1, 2, 3, 4},
        {1, -5, 2, 3}, {1, 1, -1, 3}, {2, 3, -1, 1},
    };
    for (const auto& b : blocks) {
        mat::dense2D<double> A(2, 2);
        A(0, 0) = b[0]; A(0, 1) = b[1]; A(1, 0) = b[2]; A(1, 1) = b[3];
        for (auto [Q, R] : {schur(A), schur<double>(A)}) {
            auto q = check(A, Q, R);
            INFO("block {" << b[0] << ", " << b[1] << ", " << b[2] << ", " << b[3] << "}");
            CHECK(q.backward <= 4);
            CHECK(q.orthogonal <= 4);
            CHECK(q.shape);
            // Real eigenvalues give a triangular T, complex ones a 2x2 block.
            const double disc = (b[0] - b[3]) * (b[0] - b[3]) + 4 * b[1] * b[2];
            if (disc > 0) CHECK(R(1, 0) == 0.0);
            if (disc < 0) CHECK(R(1, 0) != 0.0);
        }
    }
}

TEST_CASE("schur: column-major input", "[operation][schur]") {
    using colmat = mat::dense2D<double, mat::parameters<tag::col_major>>;
    std::mt19937_64 g(3);
    auto A = make<double>(nonnormal, 7, g);
    colmat C(7, 7);
    for (std::size_t i = 0; i < 7; ++i)
        for (std::size_t j = 0; j < 7; ++j) C(i, j) = A(i, j);
    for (auto [Q, R] : {schur(C), schur<double>(C)}) {
        auto q = check(A, Q, R);
        CHECK(q.backward <= 10);
        CHECK(q.orthogonal <= 10);
        CHECK(q.shape);
    }
}

TEST_CASE("schur: errors", "[operation][schur]") {
    REQUIRE_THROWS_AS(schur(mat::dense2D<double>(2, 3)), std::invalid_argument);
    mat::dense2D<double> A(3, 3);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j) A(i, j) = double(i + j);
    for (double bad : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
        A(1, 2) = bad;
        REQUIRE_THROWS_AS(schur(A), std::invalid_argument);
        REQUIRE_THROWS_AS(schur<double>(A), std::invalid_argument);
    }
}

TEST_CASE("schur: hessenberg_extract_Q, real and complex", "[operation][schur]") {
    std::mt19937_64 g(5);
    auto u = [&] { return double(g() >> 11) * 0x1.0p-53 - 0.5; };
    const std::size_t n = 6;

    mat::dense2D<double> A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) A(i, j) = u();
    auto W = A;
    vec::dense_vector<double> tau;
    hessenberg_factor(W, tau);
    auto Q = hessenberg_extract_Q(W, tau);
    auto H = hessenberg_extract_H(W);
    // H is Hessenberg, not quasi-triangular, so check only the factorization.
    auto q = check(A, Q, H);
    CHECK(q.backward <= 10);
    CHECK(q.orthogonal <= 10);

    // Complex: A = Q H Q^H needs the conjugated reflectors.
    using C = std::complex<double>;
    mat::dense2D<C> Ac(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) Ac(i, j) = C(u(), u());
    auto Wc = Ac;
    vec::dense_vector<C> tauc;
    hessenberg_factor(Wc, tauc);
    auto Qc = hessenberg_extract_Q(Wc, tauc);
    auto Hc = hessenberg_extract_H(Wc);
    double back = 0, orth = 0, na = 0;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            C s(0), o(0);
            for (std::size_t k = 0; k < n; ++k) {
                for (std::size_t l = 0; l < n; ++l) s += Qc(i, k) * Hc(k, l) * std::conj(Qc(j, l));
                o += std::conj(Qc(k, i)) * Qc(k, j);
            }
            back += std::norm(Ac(i, j) - s);
            na += std::norm(Ac(i, j));
            orth += std::norm(o - C(i == j ? 1.0 : 0.0));
        }
    const double eps = std::numeric_limits<double>::epsilon();
    CHECK(std::sqrt(back / na) <= 10 * n * eps);
    CHECK(std::sqrt(orth) <= 10 * n * eps);
}

TEST_CASE("rsf2csf: complex triangular Schur form", "[operation][schur]") {
    using C = std::complex<double>;
    std::mt19937_64 g(11);
    for (family f : {dense_random, cyclic, skew_tridiagonal, companion}) {
        auto A = make<double>(f, 9, g);
        auto [Q, R] = schur(A);
        auto [U, S] = rsf2csf(Q, R);
        const std::size_t n = A.num_rows();
        const double eps = std::numeric_limits<double>::epsilon();
        double back = 0, orth = 0, na = 0;
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j) {
                if (i > j) CHECK(S(i, j) == C(0));
                C s(0), o(0);
                for (std::size_t k = 0; k < n; ++k) {
                    for (std::size_t l = 0; l < n; ++l) s += U(i, k) * S(k, l) * std::conj(U(j, l));
                    o += std::conj(U(k, i)) * U(k, j);
                }
                back += std::norm(C(A(i, j)) - s);
                na += std::norm(C(A(i, j)));
                orth += std::norm(o - C(i == j ? 1.0 : 0.0));
            }
        INFO("family " << int(f));
        CHECK(std::sqrt(back / na) <= 10 * n * eps);
        CHECK(std::sqrt(orth) <= 10 * n * eps);

        // S's diagonal holds the eigenvalues, the positive imaginary part first
        // in each pair that came from a 2x2 block.
        auto e = block_eigenvalues(R);
        std::vector<std::complex<long double>> d;
        for (std::size_t k = 0; k < n; ++k) d.emplace_back(S(k, k).real(), S(k, k).imag());
        for (std::size_t k = 0; k + 1 < n; ++k)
            if (R(k + 1, k) != 0.0) {
                CHECK(S(k, k).imag() > 0);
                CHECK(std::abs(S(k + 1, k + 1) - std::conj(S(k, k))) <= 10 * eps);
            }
        sort_eigs(d);
        for (std::size_t k = 0; k < n; ++k) CHECK(std::abs(d[k] - e[k]) <= 1e-13L);
    }
    REQUIRE_THROWS_AS(rsf2csf(mat::dense2D<double>(2, 2), mat::dense2D<double>(3, 3)), std::invalid_argument);
}
