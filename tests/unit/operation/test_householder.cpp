// MTL5 -- Tests for the default Householder reflector householder(x)
//
// H = I - beta v v^T with v(0) = 1 must be orthogonal (beta v^T v == 2) and
// reflect x onto a multiple of e_1. The tests check both directly, including
// in a type with little range below 1, where forming beta from v0^2 lost it.
#include <catch2/catch_test_macros.hpp>
#include <cmath>
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

TEST_CASE("householder: a reflector the type cannot hold is the identity", "[operation][householder]") {
    // In coarse_fixed, this column (the first of mtl5-python's QR fixture)
    // has v0 = -2^-8 and beta = 2^-8, one ulp each: beta v^T v = 0.5, so the
    // computed H is far from orthogonal, which costs more than leaving the
    // tail, 11% of ||x||. In short_max the tail v = x(1:)/v0 overflows.
    const double col[] = {1.030471707975443, -0.0016801157504288797, 0.036875078408249884, -0.042832782216310726, -0.051224272907153735, 0.07432541712034423, 0.0678913563071895, -0.027514225122668375};
    vec::dense_vector<coarse_fixed> xf(8);
    for (std::size_t i = 0; i < 8; ++i) xf(i) = col[i];
    vec::dense_vector<short_max> xm{1.0, 0.25, 0.0, 0.0};
    auto check_identity = [](const auto& x) {
        auto [v, beta] = householder(x);
        REQUIRE(dbl(beta) == 0.0);
        REQUIRE(dbl(v(0)) == 1.0);
        for (std::size_t i = 1; i < x.size(); ++i)
            REQUIRE(dbl(v(i)) == 0.0);
    };
    check_identity(xf);
    check_identity(xm);

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
