// MTL5 -- exact power-of-two scaling and the scaled Givens rotation (#671).
//
// givens_rotation() must give an orthogonal rotation, a finite and correct r,
// and a zeroed second component across the whole exponent range of float and
// double: at |a| near sqrt(max) and sqrt(min), where the unscaled
// sqrt(a*a + b*b) overflows or underflows, near max itself, and in the
// subnormals. Inside the old band it must give the unscaled formula's bits.
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
#include <utility>

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
// is subnormal, so inside the old band the result is the unscaled one, bit
// for bit. That holds for separately rounded operations: this target is built
// with -ffp-contract=off (tests/unit/CMakeLists.txt), since contraction may
// fuse a different product here than in the header and move r by an ulp.
TEMPLATE_TEST_CASE("givens_rotation: inside the old band it is the unscaled form",
                   "[pow2_scaling][givens]", float, double) {
    using T = TestType;
    std::mt19937_64 rng(6710);
    std::uniform_real_distribution<double> u(-4.0, 4.0);
    for (int i = 0; i < 2000; ++i) {
        const T a = T(u(rng)), b = T(u(rng));
        const T r = std::sqrt(a * a + b * b);
        const auto g = math::givens_rotation(a, b);
        CHECK(g.r == r);
        CHECK(g.c == a / r);
        CHECK(g.s == b / r);
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

namespace {

/// A real with a posit-like coarse end: 10 fraction bits for magnitudes in
/// [2^-16, 2^17), and beyond that only the powers 2^(16 + Gap j), out to
/// 2^Limit and down to 2^-Limit, saturating. ToDouble gives it an explicit
/// conversion to double, as Universal's types have.
///
/// tapered (Gap 4, out to 2^40): halving 2^36 rounds back to 2^36, which the
/// binade-by-binade exponent search of 5.22.0 took for a non-finite value and
/// stopped at e = 0 (#680). coarse (Gap 40, out to 2^96) is takum<8>-like:
/// its values lie farther apart than its largest exact power-of-two rung,
/// 2^16, so 2^96 / 2^16 rounds back to 2^96.
template <int Gap, int Limit, bool ToDouble>
struct tapered_t {
    double v = 0.0;
    tapered_t() = default;
    tapered_t(double d) : v(round(d)) {}
    explicit operator double() const requires ToDouble { return v; }
    static double round(double d) {
        if (d == 0.0 || !std::isfinite(d)) return d;
        const double sign = d < 0 ? -1.0 : 1.0;
        double a = std::fabs(d);
        if (std::ilogb(a) >= -16 && std::ilogb(a) <= 16) {
            const int e = std::ilogb(a);
            a = std::ldexp(std::nearbyint(std::ldexp(a, 10 - e)), e - 10);
            if (std::ilogb(a) <= 16) return sign * a;
        }
        const double l = std::log2(a);
        const double t = (std::fabs(l) - 16) / Gap;
        int k = 16 + Gap * int(std::lround(t < 0 ? 0.0 : t));
        k = k > Limit ? Limit : k;
        return sign * std::ldexp(1.0, l < 0 ? -k : k);
    }
    friend tapered_t operator+(tapered_t a, tapered_t b) { return a.v + b.v; }
    friend tapered_t operator-(tapered_t a, tapered_t b) { return a.v - b.v; }
    friend tapered_t operator*(tapered_t a, tapered_t b) { return a.v * b.v; }
    friend tapered_t operator/(tapered_t a, tapered_t b) { return a.v / b.v; }
    friend tapered_t operator-(tapered_t a) { return -a.v; }
    friend bool operator==(tapered_t a, tapered_t b) { return a.v == b.v; }
    friend auto operator<=>(tapered_t a, tapered_t b) { return a.v <=> b.v; }
    friend tapered_t abs(tapered_t a) { return std::fabs(a.v); }
    friend tapered_t sqrt(tapered_t a) { return std::sqrt(a.v); }
};

using tapered = tapered_t<4, 40, false>;
using coarse = tapered_t<40, 96, true>;
using coarse_opaque = tapered_t<40, 96, false>;

/// A real with an exponent range far beyond double's, f * 2^e with f in
/// [1, 2) (or 0) and a long e, counting its multiplications and divisions.
struct wide {
    double f = 0.0;
    long e = 0;
    static inline long ops = 0;
    wide() = default;
    wide(double d) { set(d, 0); }
    static wide make(double f, long e) { wide w; w.set(f, e); return w; }
    void set(double d, long ex) {
        if (d == 0.0) { f = 0.0; e = 0; return; }
        int k;
        const double m = std::frexp(d, &k);   // d = m 2^k, m in [1/2, 1)
        f = 2 * m;
        e = ex + k - 1;
    }
    friend wide operator*(wide a, wide b) { ++ops; return make(a.f * b.f, a.e + b.e); }
    friend wide operator/(wide a, wide b) { ++ops; return make(a.f / b.f, a.e - b.e); }
    friend wide operator+(wide a, wide b) {
        if (a.f == 0.0) return b;
        if (b.f == 0.0) return a;
        if (a.e < b.e) std::swap(a, b);
        const long d = a.e - b.e;
        return d > 60 ? a : make(a.f + std::ldexp(b.f, int(-d)), a.e);
    }
    friend wide operator-(wide a) { a.f = -a.f; return a; }
    friend wide operator-(wide a, wide b) { return a + (-b); }
    friend bool operator==(wide a, wide b) { return a.f == b.f && (a.f == 0.0 || a.e == b.e); }
    friend std::partial_ordering operator<=>(wide a, wide b) {
        const int sa = (a.f > 0) - (a.f < 0), sb = (b.f > 0) - (b.f < 0);
        if (sa != sb) return sa <=> sb;
        if (sa == 0) return std::partial_ordering::equivalent;
        if (a.e != b.e) return sa > 0 ? (a.e <=> b.e) : (b.e <=> a.e);
        return a.f <=> b.f;
    }
};

} // namespace

TEST_CASE("pow2_scaling: exact exponent and scale at a tapered type's coarse ends (#680)",
          "[pow2_scaling]") {
    using P = math::pow2_scaling<tapered>;
    // Every representable power of two, the coarse ones included.
    for (int k = -40; k <= 40; ++k) {
        if ((k < -16 || k > 16) && k % 4 != 0) continue;
        CAPTURE(k);
        const tapered x(std::ldexp(1.0, k));
        REQUIRE(x.v == std::ldexp(1.0, k));
        CHECK(P::exponent(x) == k);
        CHECK(P::scale(tapered(1.0), k) == x);
        CHECK(P::scale(x, -k) == tapered(1.0));
    }
    // Inside the fine range, values between powers of two.
    for (int k = -16; k <= 16; ++k) {
        CAPTURE(k);
        CHECK(P::exponent(tapered(std::ldexp(1.5, k))) == k);
    }
    // The halving that stopped the old search.
    REQUIRE(tapered(std::ldexp(1.0, 35)).v == std::ldexp(1.0, 36));
}

TEST_CASE("givens_rotation: orthogonal at a tapered type's coarse ends (#680)", "[pow2_scaling][givens]") {
    for (int k : {-40, -36, -32, 0, 32, 36, 40}) {
        CAPTURE(k);
        const tapered a(std::ldexp(1.0, k));
        const auto g = math::givens_rotation(a, a);
        // c = s = 1/sqrt(2) to the type's 10 bits; 5.22.0 squared the
        // unscaled values here and returned c = s far from it.
        CHECK(std::fabs(g.c.v * g.c.v + g.s.v * g.s.v - 1.0) <= std::ldexp(1.0, -8));
        CHECK(std::fabs(g.c.v - std::sqrt(0.5)) <= std::ldexp(1.0, -9));
    }
}

TEST_CASE("pow2_scaling: logarithmic cost over a wide exponent range (#680)", "[pow2_scaling]") {
    using P = math::pow2_scaling<wide>;
    for (long k : {150000L, -150000L, 1L, -1L, 0L}) {
        CAPTURE(k);
        const wide x = wide::make(1.5, k);
        wide::ops = 0;
        CHECK(P::exponent(x) == k);
        CHECK(P::scale(x, int(-k)) == wide(1.5));
        CHECK(P::scale(wide(1.5), int(k)) == x);
        // The ladder is built once per type (about 4 operations per rung, 31
        // rungs); the searches then take a few operations per rung. 5.22.0
        // took one per binade, 150,000 here.
        CHECK(wide::ops <= 600);
    }
}

TEST_CASE("pow2_scaling: a type whose ends are coarser than its rungs (#680)", "[pow2_scaling]") {
    using P = math::pow2_scaling<coarse>;
    // The case that hung the first ladder search: 2^96 / 2^16 is 2^96 again.
    REQUIRE(coarse(0x1p96) / coarse(0x1p16) == coarse(0x1p96));
    for (int k : {-96, -56, -16, -1, 0, 1, 16, 56, 96}) {
        CAPTURE(k);
        const coarse x(std::ldexp(1.0, k));
        REQUIRE(x.v == std::ldexp(1.0, k));
        CHECK(P::exponent(x) == k);
        CHECK(P::scale(coarse(1.0), k) == x);
        CHECK(P::scale(x, -k) == coarse(1.0));
    }
    for (int k : {-96, -56, 0, 56, 96}) {
        CAPTURE(k);
        const coarse a(std::ldexp(1.0, k));
        const auto g = math::givens_rotation(a, a);
        CHECK(std::fabs(g.c.v * g.c.v + g.s.v * g.s.v - 1.0) <= std::ldexp(1.0, -8));
        CHECK(std::fabs(g.c.v - std::sqrt(0.5)) <= std::ldexp(1.0, -9));
    }
    // Without a conversion to double the steps cannot reach e there, so the
    // trait has to be specialized for such a type; the search still ends.
    using Q = math::pow2_scaling<coarse_opaque>;
    for (int k : {-96, 96}) {
        CAPTURE(k);
        const int e = Q::exponent(coarse_opaque(std::ldexp(1.0, k)));
        CHECK(e >= -96);
        CHECK(e <= 96);
    }
}
