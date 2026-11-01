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
