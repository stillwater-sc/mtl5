// MTL5 -- Tests for the default Householder reflector householder(x)
//
// H = I - beta v v^T with v(0) = 1 must be orthogonal (beta v^T v == 2) and
// reflect x onto a multiple of e_1. The tests check both directly, including
// in a type with little range below 1, where forming beta from v0^2 lost it.
#include <catch2/catch_test_macros.hpp>
#include <cmath>
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

double dbl(double x) { return x; }
double dbl(short_range x) { return x.v; }

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
