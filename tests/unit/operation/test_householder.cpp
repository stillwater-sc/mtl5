// MTL5 -- Tests for the default Householder reflector householder(x)
//
// H = I - beta v v^T with v(0) = 1 must be orthogonal (beta v^T v == 2) and
// reflect x onto a multiple of e_1. The tests check both directly, including
// in a type with little range below 1, where forming beta from v0^2 lost it.
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <complex>
#include <initializer_list>
#include <limits>
#include <random>
#include <mtl/mtl.hpp>

using namespace mtl;

namespace {

/// A real with little range below 1, like half precision or a short posit:
/// every result smaller in magnitude than 2^-20 is flushed to zero.
struct short_range {
    double v = 0.0;
    short_range() = default;
    constexpr short_range(double d) : v(std::abs(d) < 0x1.0p-20 ? 0.0 : d) {}
    friend short_range operator+(short_range a, short_range b) { return a.v + b.v; }
    friend short_range operator-(short_range a, short_range b) { return a.v - b.v; }
    friend short_range operator*(short_range a, short_range b) { return a.v * b.v; }
    friend short_range operator/(short_range a, short_range b) { return a.v / b.v; }
    friend short_range operator-(short_range a) { return -a.v; }
    short_range& operator+=(short_range o) { return *this = *this + o; }
    short_range& operator-=(short_range o) { return *this = *this - o; }
    short_range& operator*=(short_range o) { return *this = *this * o; }
    short_range& operator/=(short_range o) { return *this = *this / o; }
    friend bool operator==(short_range a, short_range b) { return a.v == b.v; }
    friend auto operator<=>(short_range a, short_range b) { return a.v <=> b.v; }
    friend short_range sqrt(short_range a) { return std::sqrt(a.v); }
    friend short_range abs(short_range a) { return std::abs(a.v); }
};

/// A fixed-point real like fixpnt<16,8> with saturation: every result is
/// rounded to a multiple of 2^-8 and clamped to +-(128 - 2^-8).
struct coarse_fixed {
    double v = 0.0;
    coarse_fixed() = default;
    constexpr coarse_fixed(double d) : v(round(d)) {}
    static constexpr double round(double d) {
        if (d != d) return d;   // keep a NaN a NaN, so a test can see it
        const double q = (d >= 0 ? 1.0 : -1.0) * double(static_cast<long long>((d >= 0 ? d : -d) * 256.0 + 0.5)) / 256.0;
        const double m = 128.0 - 1.0 / 256.0;
        return q > m ? m : (q < -m ? -m : q);
    }
    friend coarse_fixed operator+(coarse_fixed a, coarse_fixed b) { return a.v + b.v; }
    friend coarse_fixed operator-(coarse_fixed a, coarse_fixed b) { return a.v - b.v; }
    friend coarse_fixed operator*(coarse_fixed a, coarse_fixed b) { return a.v * b.v; }
    friend coarse_fixed operator/(coarse_fixed a, coarse_fixed b) { return a.v / b.v; }
    friend coarse_fixed operator-(coarse_fixed a) { return -a.v; }
    coarse_fixed& operator+=(coarse_fixed o) { return *this = *this + o; }
    coarse_fixed& operator-=(coarse_fixed o) { return *this = *this - o; }
    coarse_fixed& operator*=(coarse_fixed o) { return *this = *this * o; }
    coarse_fixed& operator/=(coarse_fixed o) { return *this = *this / o; }
    friend bool operator==(coarse_fixed a, coarse_fixed b) { return a.v == b.v; }
    friend auto operator<=>(coarse_fixed a, coarse_fixed b) { return a.v <=> b.v; }
    friend coarse_fixed sqrt(coarse_fixed a) { return std::sqrt(a.v); }
    friend coarse_fixed abs(coarse_fixed a) { return std::abs(a.v); }
};

/// A real with little range above 1, like an 8-bit float: every result of
/// magnitude 4 or more becomes infinite.
struct short_max {
    double v = 0.0;
    short_max() = default;
    constexpr short_max(double d)
        : v(d >= 4.0 ? std::numeric_limits<double>::infinity()
                     : (d <= -4.0 ? -std::numeric_limits<double>::infinity() : d)) {}
    friend short_max operator+(short_max a, short_max b) { return a.v + b.v; }
    friend short_max operator-(short_max a, short_max b) { return a.v - b.v; }
    friend short_max operator*(short_max a, short_max b) { return a.v * b.v; }
    friend short_max operator/(short_max a, short_max b) { return a.v / b.v; }
    friend short_max operator-(short_max a) { return -a.v; }
    short_max& operator+=(short_max o) { return *this = *this + o; }
    short_max& operator-=(short_max o) { return *this = *this - o; }
    short_max& operator*=(short_max o) { return *this = *this * o; }
    short_max& operator/=(short_max o) { return *this = *this / o; }
    friend bool operator==(short_max a, short_max b) { return a.v == b.v; }
    friend auto operator<=>(short_max a, short_max b) { return a.v <=> b.v; }
    friend short_max sqrt(short_max a) { return std::sqrt(a.v); }
    friend short_max abs(short_max a) { return std::abs(a.v); }
};

double dbl(double x) { return x; }
double dbl(short_range x) { return x.v; }
double dbl(coarse_fixed x) { return x.v; }
double dbl(short_max x) { return x.v; }

/// max_i>=1 |(H x)(i)| / ||x|| and |beta v^T v - 2|, in double.
template <typename T>
std::pair<double, double> reflect_quality(const vec::dense_vector<T>& x) {
    auto [v, beta] = householder(x);
    const std::size_t n = x.size();
    double vtx = 0, vtv = 0, nx = 0;
    for (std::size_t i = 0; i < n; ++i) {
        vtx += dbl(v(i)) * dbl(x(i));
        vtv += dbl(v(i)) * dbl(v(i));
        nx += dbl(x(i)) * dbl(x(i));
    }
    double tail = 0;
    for (std::size_t i = 1; i < n; ++i)
        tail = std::max(tail, std::abs(dbl(x(i)) - dbl(beta) * dbl(v(i)) * vtx));
    return {tail / std::sqrt(nx), std::abs(dbl(beta) * vtv - 2.0)};
}

} // namespace

TEST_CASE("householder: a small tail in a type with little range", "[operation][householder]") {
    // The tail is 2% of x(0), so v0 ~ 2.6e-4 and v0^2 ~ 6.7e-8, below the
    // type's 2^-20: beta = 2 v0^2 / (sigma + v0^2) flushed to zero and the
    // reflector became the identity. beta = -v0 / ||x|| does not square v0.
    for (const auto& x : {vec::dense_vector<short_range>{0.72265625, -0.01643372},
                          vec::dense_vector<short_range>{0.9, 0.02, -0.01},
                          vec::dense_vector<short_range>{-0.5, 0.01, 0.0, 0.004}}) {
        auto [tail, orth] = reflect_quality(x);
        CHECK(tail <= 1e-14);
        CHECK(orth <= 1e-14);
    }
}

TEST_CASE("householder: orthogonal and reflecting onto e_1 in double", "[operation][householder]") {
    std::mt19937_64 g(3);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    const double eps = std::numeric_limits<double>::epsilon();
    for (std::size_t n : {1u, 2u, 3u, 7u, 20u})
        for (double tail_scale : {1.0, 1e-4, 1e-8, 1e-150, 1e150})
            for (int rep = 0; rep < 20; ++rep) {
                vec::dense_vector<double> x(n);
                for (std::size_t i = 0; i < n; ++i) x(i) = (i == 0 ? 1.0 : tail_scale) * u(g);
                auto [tail, orth] = reflect_quality(x);
                INFO("n " << n << ", tail scale " << tail_scale);
                CHECK(tail <= 4 * n * eps);
                if (n > 1 && tail_scale >= 1e-8) CHECK(orth <= 4 * n * eps);   // v^T v ~ 1e16 here
            }
}

TEST_CASE("householder: a narrow type keeps the reflector dlarfg's sign gives it", "[operation][householder]") {
    // The first column of mtl5-python's QR fixture, in coarse_fixed. With
    // Parlett's sign (before 5.22.0) v0 was -2^-8 and beta 2^-8, one ulp
    // each, so beta v^T v = 0.5 and householder() fell back to H = I. With
    // dlarfg's sign (#672) v0 = x0 + ||x|| and every |v(i)| <= 1, which the
    // type holds to its resolution, 2^-8.
    const double col[] = {1.030471707975443, -0.0016801157504288797, 0.036875078408249884, -0.042832782216310726, -0.051224272907153735, 0.07432541712034423, 0.0678913563071895, -0.027514225122668375};
    vec::dense_vector<coarse_fixed> xf(8);
    for (std::size_t i = 0; i < 8; ++i) xf(i) = col[i];
    auto [v, beta] = householder(xf);
    REQUIRE(dbl(beta) != 0.0);
    for (std::size_t i = 1; i < 8; ++i)
        CHECK(std::abs(dbl(v(i))) <= 1.0);
    auto [tail, orth] = reflect_quality(xf);   // 0.0029 and 0.0031 measured
    INFO("tail " << tail << ", orth " << orth);
    CHECK(tail <= 0x1.0p-8);
    CHECK(orth <= 0x1.0p-8);
}

TEST_CASE("householder: a reflector the type cannot hold is the identity", "[operation][householder]") {
    // In short_max the sum of squares of x = (1, ..., 1) reaches 4 and
    // overflows, so ||x|| is infinite and beta v^T v is NaN.
    vec::dense_vector<short_max> xm(8);
    for (std::size_t i = 0; i < 8; ++i) xm(i) = 1.0;
    auto [v, beta] = householder(xm);
    REQUIRE(dbl(beta) == 0.0);
    REQUIRE(dbl(v(0)) == 1.0);
    for (std::size_t i = 1; i < xm.size(); ++i)
        REQUIRE(dbl(v(i)) == 0.0);

    // QR of the whole fixture-like matrix stays finite and Q orthogonal.
    const std::size_t n = 8;
    mat::dense2D<short_max> A(n, n);
    std::mt19937_64 g(42);
    std::normal_distribution<double> z(0.0, 0.1);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A(i, j) = (i == j ? 1.0 : 0.0) + z(g);
    vec::dense_vector<short_max> tau(n);
    qr_factor(A, tau);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = i; j < n; ++j)
            REQUIRE(std::isfinite(dbl(A(i, j))));   // R
    auto Q = qr_extract_Q(A, tau);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            double o = 0;
            for (std::size_t k = 0; k < n; ++k) o += dbl(Q(k, i)) * dbl(Q(k, j));
            REQUIRE(std::isfinite(o));
            CHECK(std::abs(o - (i == j ? 1.0 : 0.0)) <= 1e-12);
        }
}

TEST_CASE("householder: never worse than the identity in a coarse type", "[operation][householder]") {
    // Either H = I, or its orthogonality error |beta v^T v - 2| beta v^T v is
    // at most what H = I would leave, ||tail|| / ||x||. householder() decides
    // with both sides computed in the type, rounded to 2^-8 at every step;
    // recomputed here in double, they move by about that much, hence the
    // slack.
    std::mt19937_64 g(17);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    for (std::size_t n : {2u, 3u, 5u, 8u})
        for (double tail_scale : {1.0, 0.3, 0.1, 0.03})
            for (int rep = 0; rep < 50; ++rep) {
                vec::dense_vector<coarse_fixed> x(n);
                for (std::size_t i = 0; i < n; ++i) x(i) = (i == 0 ? 1.0 : tail_scale) * u(g);
                auto [v, beta] = householder(x);
                if (dbl(beta) == 0.0) continue;
                double vtv = 0, nx = 0, nt = 0;
                for (std::size_t i = 0; i < n; ++i) {
                    vtv += dbl(v(i)) * dbl(v(i));
                    nx += dbl(x(i)) * dbl(x(i));
                    if (i > 0) nt += dbl(x(i)) * dbl(x(i));
                }
                const double c = dbl(beta) * vtv;
                INFO("n " << n << ", tail scale " << tail_scale << ", c " << c);
                CHECK(std::abs(c - 2.0) * c <= 1.5 * std::sqrt(nt / nx) + 0.02);
            }
}

TEST_CASE("householder: x is reflected onto -sign(x0) ||x||, LAPACK's sign (#672)",
          "[operation][householder][672]") {
    // H x = alpha e_1 with alpha = -sign(x0) ||x|| and sign(0) = +1, as in
    // dlarfg, the textbook form and the complex path. Exact here: x = (3, 4)
    // scales to (0.75, 1), and v = (1, 0.5), beta = 1.6.
    auto onto = [](std::initializer_list<double> init) {
        vec::dense_vector<double> x(init.size());
        std::size_t i = 0;
        for (double d : init) x(i++) = d;
        auto [v, beta] = householder(x);
        double vtx = 0;
        for (std::size_t k = 0; k < x.size(); ++k) vtx += v(k) * x(k);
        return x(0) - beta * v(0) * vtx;
    };
    CHECK(onto({3.0, 4.0}) == -5.0);
    CHECK(onto({-3.0, 4.0}) == 5.0);
    CHECK(onto({0.0, 2.0}) == -2.0);
    CHECK(onto({0.0, -2.0}) == -2.0);

    // qr_factor's R carries the same sign.
    mat::dense2D<double> A(2, 1);
    A(0, 0) = 3.0;
    A(1, 0) = 4.0;
    vec::dense_vector<double> tau;
    qr_factor(A, tau);
    CHECK(std::abs(A(0, 0) + 5.0) <= 1e-15);

    // A complex pivot with Re(x0) == 0 takes -||x||, as zlarfg does.
    mat::dense2D<std::complex<double>> C(2, 1);
    C(0, 0) = {0.0, 3.0};
    C(1, 0) = {4.0, 0.0};
    vec::dense_vector<std::complex<double>> ctau;
    qr_factor(C, ctau);
    CHECK(std::abs(C(0, 0) - std::complex<double>(-5.0, 0.0)) <= 1e-14);
}

TEST_CASE("householder: a small tail keeps v bounded and H orthogonal (#672)",
          "[operation][householder][672]") {
    // x = (+-1, +-2^-j (1 + u)), j from 0 to past the type's precision: the
    // study's tail sweep. With Parlett's sign (before 5.22.0) x0 = +1 gave
    // v(1) ~ -2 / x(1), up to 2^(j+1), and in a narrow type an inaccurate H;
    // with dlarfg's sign |v(1)| <= 1 for every j.
    std::mt19937_64 g(672);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    auto sweep = [&](auto zero, int digits, double eps, bool accurate) {
        using T = decltype(zero);
        for (double s0 : {1.0, -1.0})
            for (double s1 : {1.0, -1.0})
                for (int j = 0; j <= digits + 4; ++j)
                    for (int rep = 0; rep < 4; ++rep) {
                        vec::dense_vector<T> x(2);
                        x(0) = T(s0);
                        x(1) = T(s1 * std::ldexp(1.0 + u(g), -j));
                        auto [v, beta] = householder(x);
                        INFO("x = (" << dbl(x(0)) << ", " << dbl(x(1)) << "), j " << j);
                        if (dbl(beta) == 0.0) {
                            // Only a tail the type rounds away, or one too
                            // small for the type to reflect accurately.
                            CHECK((dbl(x(1)) == 0.0 || !accurate));
                            continue;
                        }
                        CHECK(std::abs(dbl(v(1))) <= 1.0);
                        CHECK(dbl(beta) >= 1.0);
                        CHECK(dbl(beta) <= 2.0);
                        if (accurate) {
                            auto [tail, orth] = reflect_quality(x);
                            CHECK(tail <= 4 * eps);
                            CHECK(orth <= 4 * eps);
                        }
                    }
    };
    sweep(double{}, 53, std::numeric_limits<double>::epsilon(), true);
    sweep(float{}, 24, std::numeric_limits<float>::epsilon(), true);
    sweep(coarse_fixed{}, 8, 0x1.0p-8, false);
}

TEST_CASE("householder: scaling x by a power of two leaves v and beta unchanged (#672)",
          "[operation][householder][672]") {
    // x is scaled by an exact 2^-e before anything is squared, so x and
    // x 2^k give the same scaled entries, and the same v and beta bit for bit,
    // as long as x 2^k is itself exact.
    std::mt19937_64 g(6720);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    for (std::size_t n : {2u, 5u, 9u})
        for (int rep = 0; rep < 10; ++rep) {
            vec::dense_vector<double> x(n);
            for (std::size_t i = 0; i < n; ++i) x(i) = u(g);
            auto [v, beta] = householder(x);
            for (int k : {-1000, -600, -37, -1, 1, 37, 600, 1000}) {
                vec::dense_vector<double> y(n);
                for (std::size_t i = 0; i < n; ++i) y(i) = std::ldexp(x(i), k);
                auto [w, gamma] = householder(y);
                INFO("n " << n << ", k " << k);
                CHECK(gamma == beta);
                for (std::size_t i = 0; i < n; ++i) CHECK(w(i) == v(i));
            }
        }
}

TEST_CASE("householder_reflector: beta is H x, and householder() is its v and tau (#678)",
          "[operation][householder]") {
    std::mt19937_64 g(678);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    for (int t = 0; t < 200; ++t) {
        const std::size_t n = 2 + std::size_t(t % 9);
        vec::dense_vector<double> x(n);
        for (std::size_t i = 0; i < n; ++i) x(i) = std::ldexp(u(g), t % 40 - 20);
        const auto [v, tau, beta] = householder_reflector(x);
        const auto [v2, tau2] = householder(x);
        REQUIRE(tau == tau2);
        for (std::size_t i = 0; i < n; ++i) REQUIRE(v(i) == v2(i));
        double nx = 0, vtx = 0;
        for (std::size_t i = 0; i < n; ++i) {
            nx += x(i) * x(i);
            vtx += v(i) * x(i);
        }
        nx = std::sqrt(nx);
        // H x = x - tau v (v^T x): its first entry is beta, which has the
        // opposite sign of x(0) and magnitude ||x||.
        REQUIRE(std::abs(beta - (x(0) - tau * vtx)) <= 8 * std::numeric_limits<double>::epsilon() * nx);
        REQUIRE(std::abs(std::abs(beta) - nx) <= 4 * std::numeric_limits<double>::epsilon() * nx);
        REQUIRE((x(0) >= 0) == (beta < 0));
    }
    // H = I: beta is x(0).
    vec::dense_vector<double> e1(3);
    e1(0) = -3.0; e1(1) = 0.0; e1(2) = 0.0;
    const auto r = householder_reflector(e1);
    REQUIRE(r.tau == 0.0);
    REQUIRE(r.beta == -3.0);
    vec::dense_vector<double> zero(3);
    zero(0) = zero(1) = zero(2) = 0.0;
    REQUIRE(householder_reflector(zero).beta == 0.0);
}

namespace {

/// x = (0, u, ..., u) with eight entries u = 2^-8, the resolution of
/// coarse_fixed. ||x|| = 2.83u, so beta is a nonzero multiple of u, while
/// v(i) = 0.355 makes every product v(i) x(i) = 0.355u round to 0: applying H
/// to x recomputes the pivot as 0 - tau (v^T x) = 0.
vec::dense_vector<coarse_fixed> resolution_column() {
    vec::dense_vector<coarse_fixed> x(9);
    x(0) = coarse_fixed(0.0);
    for (std::size_t i = 1; i < 9; ++i) x(i) = coarse_fixed(std::ldexp(1.0, -8));
    return x;
}

} // namespace

TEST_CASE("qr, lq, hessenberg: the pivot is beta, not recomputed (#678)", "[operation][householder]") {
    const auto x = resolution_column();
    const coarse_fixed beta = householder_reflector(x).beta;
    REQUIRE(dbl(beta) != 0.0);
    REQUIRE(std::abs(dbl(beta)) >= 2 * std::ldexp(1.0, -8));

    SECTION("qr_factor: R(0,0), and qr_solve stays finite") {
        mat::dense2D<coarse_fixed> A(9, 1);
        for (std::size_t i = 0; i < 9; ++i) A(i, 0) = x(i);
        const mat::dense2D<coarse_fixed> A0(A);
        vec::dense_vector<coarse_fixed> tau(1), sol(1), b(9);
        qr_factor(A, tau);
        REQUIRE(A(0, 0) == beta);
        for (std::size_t i = 0; i < 9; ++i) b(i) = A0(i, 0);
        // Before #678 R(0,0) was 0 and the back substitution 0/0. Forming
        // Q^T b rounds the same products v(i) b(i) away, so x is 0 rather
        // than 1: storing beta removes the division by zero, not the type's
        // resolution.
        qr_solve(A, tau, sol, b);
        REQUIRE(std::isfinite(dbl(sol(0))));
    }
    SECTION("lq_factor: L(0,0)") {
        mat::dense2D<coarse_fixed> A(1, 9);
        for (std::size_t j = 0; j < 9; ++j) A(0, j) = x(j);
        vec::dense_vector<coarse_fixed> tau(1);
        lq_factor(A, tau);
        REQUIRE(A(0, 0) == beta);
    }
    SECTION("hessenberg_factor: H(1,0)") {
        mat::dense2D<coarse_fixed> A(10, 10);
        for (std::size_t i = 0; i < 10; ++i)
            for (std::size_t j = 0; j < 10; ++j) A(i, j) = coarse_fixed(0.0);
        for (std::size_t i = 0; i < 9; ++i) A(i + 1, 0) = x(i);
        vec::dense_vector<coarse_fixed> tau(9);
        hessenberg_factor(A, tau);
        REQUIRE(A(1, 0) == beta);
    }
}

TEST_CASE("symmetric eigensolvers: the tridiagonal keeps the subdiagonal (#678)",
          "[operation][householder]") {
    // A = x e_0^T + e_0 x^T over the resolution column: rank two, eigenvalues
    // +-||x|| = +-2.83u. A recomputed subdiagonal of 0 decouples the first row
    // and gives all eigenvalues 0.
    const auto x = resolution_column();
    mat::dense2D<coarse_fixed> A(10, 10);
    for (std::size_t i = 0; i < 10; ++i)
        for (std::size_t j = 0; j < 10; ++j) A(i, j) = coarse_fixed(0.0);
    for (std::size_t i = 0; i < 9; ++i) A(i + 1, 0) = A(0, i + 1) = x(i);
    const double u = std::ldexp(1.0, -8);
    auto largest = [](const auto& lambda) {
        double m = 0;
        for (std::size_t i = 0; i < lambda.size(); ++i) m = std::max(m, std::abs(dbl(lambda(i))));
        return m;
    };
    REQUIRE(largest(eigenvalue_symmetric_generic(A)) >= 2 * u);
    REQUIRE(largest(eigen_symmetric(A).eigenvalues) >= 2 * u);
}

TEST_CASE("householder_reflector: a complex beta is real, ||x||, and opposite to Re(x0) (#698)",
          "[operation][householder]") {
    using C = std::complex<double>;
    std::mt19937_64 g(698);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    auto check = [](const vec::dense_vector<C>& x) {
        const std::size_t n = x.size();
        const auto [v, tau, beta] = householder_reflector(x);
        double nx = 0;
        for (std::size_t i = 0; i < n; ++i) nx = std::hypot(nx, std::abs(x(i)));
        // H x = x - tau v (v^H x) is beta e_1.
        C vhx = 0;
        for (std::size_t i = 0; i < n; ++i) vhx += std::conj(v(i)) * x(i);
        const double eps = std::numeric_limits<double>::epsilon();
        REQUIRE(beta.imag() == 0.0);
        REQUIRE(std::abs(std::abs(beta.real()) - nx) <= 4 * eps * nx);
        REQUIRE((x(0).real() >= 0) == (beta.real() < 0));
        REQUIRE(std::abs(x(0) - tau * vhx - beta) <= 8 * eps * nx);
        for (std::size_t i = 1; i < n; ++i)
            REQUIRE(std::abs(x(i) - tau * v(i) * vhx) <= 8 * eps * nx);
    };
    // A complex x is divided by its largest |x(i)|, not by a power of two, and
    // beta multiplied back: the scales put that factor far from 1.
    for (double scale : {1.0, 3.7e-3, 1e300, 1e-300})
        for (int t = 0; t < 50; ++t) {
            const std::size_t n = 1 + std::size_t(t % 7);
            vec::dense_vector<C> x(n);
            for (std::size_t i = 0; i < n; ++i) x(i) = scale * C(u(g), u(g));
            if (t % 5 == 0) x(0) = C(0.0, x(0).imag());   // Re(x0) == 0 takes -||x||
            CAPTURE(scale, t);
            check(x);
        }
    // A zero tail with a complex x0 still reflects, to a real beta = -|x0|
    // here; a zero tail with a real x0 is H = I, and beta is x0.
    vec::dense_vector<C> x(3);
    x(0) = C(3.0, -4.0); x(1) = x(2) = C(0.0, 0.0);
    check(x);
    REQUIRE(householder_reflector(x).beta == C(-5.0, 0.0));
    REQUIRE(householder_reflector(x).tau != C(0.0, 0.0));
    x(0) = C(-2.5, 0.0);
    const auto id = householder_reflector(x);
    REQUIRE(id.tau == C(0.0, 0.0));
    REQUIRE(id.beta == C(-2.5, 0.0));
}

TEST_CASE("symmetric eigensolvers: a fixed-point QR step meeting x = z = 0 rotates by I (#698)",
          "[operation][householder]") {
    // In coarse_fixed the QR sweep of this tridiagonal rounds both x and the
    // bulge z to 0 mid-chase. givens_rotation(0, 0) is the identity; without
    // that case it divides 0 by 0, and the NaN reaches every eigenvalue and,
    // through the accumulated rotations, the eigenvectors. Double never gets
    // there: its x and z do not round to 0 together.
    const double u = std::ldexp(1.0, -8);
    const int a[3][3] = {{-5, -6, 0}, {-6, -5, -2}, {0, -2, -1}};
    mat::dense2D<coarse_fixed> A(3, 3);
    mat::dense2D<double> Ad(3, 3);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j) {
            A(i, j) = coarse_fixed(a[i][j] * u);
            Ad(i, j) = a[i][j] * u;
        }
    const auto exact = eigenvalue_symmetric_generic(Ad);
    const auto values = eigenvalue_symmetric_generic(A);
    const auto [lambda, V] = eigen_symmetric(A);
    for (std::size_t k = 0; k < 3; ++k) {
        CAPTURE(k, dbl(values(k)) / u, dbl(lambda(k)) / u, exact(k) / u);
        REQUIRE(std::isfinite(dbl(values(k))));
        REQUIRE(std::isfinite(dbl(lambda(k))));
        // A few units of coarse_fixed's resolution: the type's rounding, not NaN.
        CHECK(std::abs(dbl(values(k)) - exact(k)) <= 4 * u);
        CHECK(std::abs(dbl(lambda(k)) - exact(k)) <= 4 * u);
        for (std::size_t l = 0; l < 3; ++l) {
            double vtv = 0;
            for (std::size_t i = 0; i < 3; ++i) vtv += dbl(V(i, k)) * dbl(V(i, l));
            CAPTURE(l);
            CHECK(std::abs(vtv - (k == l ? 1.0 : 0.0)) <= 16 * u);
        }
    }
}
