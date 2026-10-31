// MTL5 -- exact power-of-two scaling and the scaled Givens rotation (#671).
//
// givens_rotation() must give an orthogonal rotation, a finite and correct r,
// and a zeroed second component across the whole exponent range of float and
// double: at |a| near sqrt(max) and sqrt(min), where the unscaled
// sqrt(a*a + b*b) overflows or underflows, near max itself, and in the
// subnormals. Inside the old band it must agree with the unscaled formula.
// The fallback that a user-defined type gets is exercised with a wrapper
// around double, and the solvers that use the rotation are checked on a
// problem scaled to where the unscaled form overflows.
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <compare>
#include <complex>
#include <limits>
#include <random>

#include <mtl/math/pow2_scaling.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/givens.hpp>
#include <mtl/operation/eigenvalue_symmetric.hpp>

using namespace mtl;

namespace {

// A user-defined real type: no std::ilogb, so pow2_scaling's fallback is used.
struct wrapped {
    double v;
    wrapped() = default;
    constexpr wrapped(double x) : v(x) {}
    friend wrapped operator+(wrapped a, wrapped b) { return a.v + b.v; }
    friend wrapped operator-(wrapped a, wrapped b) { return a.v - b.v; }
    friend wrapped operator*(wrapped a, wrapped b) { return a.v * b.v; }
    friend wrapped operator/(wrapped a, wrapped b) { return a.v / b.v; }
    friend wrapped operator-(wrapped a) { return -a.v; }
    friend bool operator==(wrapped a, wrapped b) { return a.v == b.v; }
    friend auto operator<=>(wrapped a, wrapped b) { return a.v <=> b.v; }
    friend wrapped abs(wrapped a) { return std::fabs(a.v); }
    friend wrapped sqrt(wrapped a) { return std::sqrt(a.v); }
};

// The rotation's errors, in eps of T: |c^2 + s^2 - 1|, the residual of the
// zeroed component relative to |r|, and |r - hypot(a, b)| relative to hypot,
// or to min() where hypot is subnormal and r holds fewer bits.
template <typename T>
struct rotation_error { double orth, zeroed, radius; };

template <typename T>
rotation_error<T> check(T a, T b) {
    const auto g = math::givens_rotation(a, b);
    const long double eps = std::numeric_limits<T>::epsilon();
    const long double c = g.c, s = g.s, r = g.r;
    const long double la = a, lb = b;
    const long double h = std::hypot(la, lb);
    return {double(std::fabs(c * c + s * s - 1) / eps),
            double(std::fabs(-s * la + c * lb) / h / eps),
            double(std::fabs(r - h) / std::fmax(h, (long double)std::numeric_limits<T>::min()) / eps)};
}

} // namespace

TEMPLATE_TEST_CASE("pow2_scaling: exponent and scale are exact", "[pow2_scaling]", float, double) {
    using T = TestType;
    using S = math::pow2_scaling<T>;
    const int emin = std::numeric_limits<T>::min_exponent - 1;     // exponent of min()
    const int emax = std::numeric_limits<T>::max_exponent - 1;     // exponent of max()
    CHECK(S::exponent(T(1)) == 0);
    CHECK(S::exponent(T(1.5)) == 0);
    CHECK(S::exponent(T(2)) == 1);
    CHECK(S::exponent(T(0.75)) == -1);
    CHECK(S::exponent(std::numeric_limits<T>::max()) == emax);
    CHECK(S::exponent(std::numeric_limits<T>::min()) == emin);
    CHECK(S::exponent(std::numeric_limits<T>::denorm_min()) == emin - std::numeric_limits<T>::digits + 1);
    CHECK(S::scale(T(1.5), emax) == std::ldexp(T(1.5), emax));
    CHECK(S::scale(std::numeric_limits<T>::denorm_min(), 60) == std::ldexp(T(1), emin - std::numeric_limits<T>::digits + 61));
}

TEST_CASE("pow2_scaling: the fallback matches ldexp and ilogb", "[pow2_scaling]") {
    using S = math::pow2_scaling<wrapped>;
    for (double x : {1.0, 1.75, 3.0, 0.3, 1e-300, 1e300, 0x1.fffffffffffffp1023, 0x1p-1022, 0x1p-1074, 0x1.8p-1070}) {
        CAPTURE(x);
        CHECK(S::exponent(x) == std::ilogb(x));
        for (int k : {-1100, -600, -1, 0, 1, 7, 600, 1100}) {
            CAPTURE(k);
            CHECK(S::scale(x, k).v == std::ldexp(x, k));
        }
    }
}

TEMPLATE_TEST_CASE("givens_rotation: orthogonal and correct across the exponent range",
                   "[pow2_scaling][givens]", float, double) {
    using T = TestType;
    const int emin = std::numeric_limits<T>::min_exponent - std::numeric_limits<T>::digits;  // denorm_min
    const int emax = std::numeric_limits<T>::max_exponent - 2;   // ||x|| up to 2^emax * sqrt(2) < max
    std::mt19937_64 rng(671);
    std::uniform_real_distribution<double> angle(0.0, 6.283185307179586);
    double worst_orth = 0, worst_zeroed = 0, worst_radius = 0;
    for (int k = emin + 2; k <= emax; ++k) {   // from the subnormals to max
        for (int i = 0; i < 8; ++i) {
            const double t = angle(rng);
            const T a = std::ldexp(T(std::cos(t)), k), b = std::ldexp(T(std::sin(t)), k);
            const auto err = check(a, b);
            worst_orth = std::fmax(worst_orth, err.orth);
            worst_zeroed = std::fmax(worst_zeroed, err.zeroed);
            worst_radius = std::fmax(worst_radius, err.radius);
        }
    }
    CHECK(worst_orth <= 4);
    CHECK(worst_zeroed <= 4);
    CHECK(worst_radius <= 4);
}

TEMPLATE_TEST_CASE("givens_rotation: where the unscaled form overflows or underflows",
                   "[pow2_scaling][givens]", float, double) {
    using T = TestType;
    const T rmax = std::sqrt(std::numeric_limits<T>::max());
    const T rmin = std::sqrt(std::numeric_limits<T>::min());
    for (T base : {rmax * T(16), rmax * T(2), rmax / T(16), rmin * T(16), rmin / T(2), rmin / T(16), rmin * T(0x1p-40),
                   std::numeric_limits<T>::max() / T(8), std::numeric_limits<T>::min() * T(3),
                   std::numeric_limits<T>::denorm_min() * T(5)}) {
        for (auto [p, q] : {std::pair{3.0, 4.0}, std::pair{-1.0, 1.0}, std::pair{1.0, 1e-3}, std::pair{0.0, -1.0}}) {
            const T a = base * T(p), b = base * T(q);
            CAPTURE(base, a, b);
            const auto g = math::givens_rotation(a, b);
            REQUIRE(std::isfinite(g.r));
            REQUIRE(g.r > T(0));
            const auto err = check(a, b);
            // denorm_min * 5 holds 3 bits, so (3, 4) and (1, 1e-3) are rounded
            // inputs and only the rotation's own orthogonality is tight there.
            CHECK(err.orth <= 4);
            if (base > std::numeric_limits<T>::min()) {
                CHECK(err.zeroed <= 4);
                CHECK(err.radius <= 4);
            }
        }
    }
    // The unscaled formula is wrong at both ends; this is what #671 fixes.
    CHECK(std::isinf(std::sqrt(rmax * T(2) * rmax * T(2) + T(0))));
    CHECK(std::sqrt((rmin * T(0x1p-40)) * (rmin * T(0x1p-40))) == T(0));
}

// Scaling by 2^-e is exact and commutes with every rounding while no value
// is subnormal, so under strict IEEE evaluation the result inside the old
// band is the unscaled one, bit for bit. It is not checked bit for bit: where
// the target has an FMA (ARM64, or x86-64 at -march=x86-64-v3 and up), GCC
// contracts a*a + b*b by default and may fuse a different product here than
// in the header, which moves r by an ulp.
TEMPLATE_TEST_CASE("givens_rotation: inside the old band it agrees with the unscaled form",
                   "[pow2_scaling][givens]", float, double) {
    using T = TestType;
    const T eps = std::numeric_limits<T>::epsilon();
    std::mt19937_64 rng(6710);
    std::uniform_real_distribution<double> u(-4.0, 4.0);
    for (int i = 0; i < 2000; ++i) {
        const T a = T(u(rng)), b = T(u(rng));
        const T r = std::sqrt(a * a + b * b);
        const auto g = math::givens_rotation(a, b);
        CHECK(std::fabs(g.r - r) <= 2 * eps * r);
        CHECK(std::fabs(g.c - a / r) <= 2 * eps);
        CHECK(std::fabs(g.s - b / r) <= 2 * eps);
    }
}

TEST_CASE("givens_rotation: special inputs", "[pow2_scaling][givens]") {
    auto g = math::givens_rotation(0.0, 0.0);
    CHECK(g.c == 1.0);
    CHECK(g.s == 0.0);
    CHECK(g.r == 0.0);
    g = math::givens_rotation(0.0, -2.0);
    CHECK(g.c == 0.0);
    CHECK(g.s == -1.0);
    CHECK(g.r == 2.0);
    g = math::givens_rotation(-3.0, 0.0);
    CHECK(g.c == -1.0);
    CHECK(g.s == 0.0);
    CHECK(g.r == 3.0);
    // Non-finite input keeps the unscaled formula's result.
    g = math::givens_rotation(std::numeric_limits<double>::infinity(), 1.0);
    CHECK(std::isinf(g.r));
    CHECK(math::scaled_hypot(3e300, 4e300) == 5e300);
}

TEST_CASE("givens_rotation: a user-defined type uses the fallback", "[pow2_scaling][givens]") {
    for (double scale : {1.0, 0x1p600, 0x1p-600, 0x1p1000, 0x1p-1050}) {
        CAPTURE(scale);
        const wrapped a = 3.0 * scale, b = -4.0 * scale;
        const auto g = math::givens_rotation(a, b);
        CHECK(g.c.v == 0.6);
        CHECK(g.s.v == -0.8);
        CHECK(g.r.v == 5.0 * scale);
    }
}

TEST_CASE("givens_rotation: complex keeps the unscaled formula", "[pow2_scaling][givens]") {
    using C = std::complex<double>;
    const C a(3, 0), b(4, 0);
    const auto g = math::givens_rotation(a, b);
    CHECK(g.r == std::sqrt(a * a + b * b));
    CHECK(g.c == a / g.r);
}

TEST_CASE("apply_givens_rotation: a Hessenberg column beyond sqrt(max)", "[pow2_scaling][givens]") {
    const double big = 0x1p600;   // big * big overflows
    mat::dense2D<double> H(2, 1);
    H(0, 0) = 3 * big;
    H(1, 0) = 4 * big;
    vec::dense_vector<double> g(2), c(1), s(1);
    g(0) = 1.0;
    g(1) = 0.0;
    apply_givens_rotation(H, g, c, s, std::size_t(0));
    CHECK(H(0, 0) == 5 * big);
    CHECK(H(1, 0) == 0.0);
    CHECK(c(0) == 0.6);
    CHECK(s(0) == 0.8);
    CHECK(g(0) == 0.6);
    CHECK(g(1) == -0.8);
}

TEST_CASE("eigenvalue_symmetric: a tridiagonal beyond sqrt(max)", "[pow2_scaling][givens]") {
    // The tridiagonal (1, 2, 1) of order 3, scaled by 2^600. The unscaled
    // rotation and Wilkinson shift square entries of 2^600 and overflow.
    const double big = 0x1p600;
    mat::dense2D<double> A(3, 3);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j) A(i, j) = 0.0;
    A(0, 0) = 2 * big; A(0, 1) = big;
    A(1, 0) = big;     A(1, 1) = 2 * big; A(1, 2) = big;
    A(2, 1) = big;     A(2, 2) = 2 * big;
    const auto ev = eigenvalue_symmetric(A);
    // 2 - sqrt(2), 2 and 2 + sqrt(2), times 2^600.
    const double expected[] = {2 - std::sqrt(2.0), 2.0, 2 + std::sqrt(2.0)};
    for (int i = 0; i < 3; ++i) {
        CAPTURE(i, ev(i) / big);
        CHECK(std::isfinite(ev(i)));
        CHECK(std::fabs(ev(i) / big - expected[i]) <= 1e-13);
    }
}
