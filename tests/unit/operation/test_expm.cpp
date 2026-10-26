// MTL5 -- Tests for the matrix exponential expm (#650)
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>
#include <mtl/mtl.hpp>

using namespace mtl;
using Catch::Matchers::WithinAbs;

namespace {

template <typename T>
mat::dense2D<T> make(std::initializer_list<std::initializer_list<T>> rows) {
    const std::size_t n = rows.size(), m = rows.begin()->size();
    mat::dense2D<T> A(n, m);
    std::size_t i = 0;
    for (const auto& r : rows) {
        std::size_t j = 0;
        for (const auto& v : r) A(i, j++) = v;
        ++i;
    }
    return A;
}

/// ||X - Y||_1 / ||Y||_1 in long double, for any X, Y with comparable entries.
template <typename MX, typename MY>
long double rel_err(const MX& X, const MY& Y) {
    long double num = 0, den = 0;
    for (std::size_t j = 0; j < X.num_cols(); ++j) {
        long double cn = 0, cd = 0;
        for (std::size_t i = 0; i < X.num_rows(); ++i) {
            using std::abs;
            cn += static_cast<long double>(abs(std::complex<long double>(X(i, j)) -
                                               std::complex<long double>(Y(i, j))));
            cd += static_cast<long double>(abs(std::complex<long double>(Y(i, j))));
        }
        num = std::max(num, cn);
        den = std::max(den, cd);
    }
    return num / den;
}

template <typename T>
mat::dense2D<T> eye(std::size_t n) {
    mat::dense2D<T> I(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) I(i, j) = (i == j) ? T(1) : T(0);
    return I;
}

template <typename T>
mat::dense2D<T> product(const mat::dense2D<T>& A, const mat::dense2D<T>& B) {
    mat::dense2D<T> C(A.num_rows(), B.num_cols());
    mult(A, B, C);
    return C;
}

/// The 4x4 nonnormal test matrix, ||A||_1 = 10, so expm scales it.
template <typename T>
mat::dense2D<T> test_matrix() {
    return make<T>({{T(1), T(2), T(0), T(0.5)},
                    {T(-3), T(0.5), T(4), T(0)},
                    {T(0), T(-1), T(-2), T(6)},
                    {T(1.5), T(0), T(-4), T(1)}});
}

// exp(test_matrix()), from mpmath.expm at 60 digits.
const long double ref_real[4][4] = {
    {0.137903158366217940831L, 1.99823036418746114716L, -0.136291208492045538922L, 2.41703424974776298703L},
    {-1.03861485022544631981L, -0.381409546961410941167L, -0.703119390782494184028L, -0.548759929846242797898L},
    {0.0538726990251998223094L, 1.15514519572349624647L, 0.504581595754856567558L, 0.559674753628119824518L},
    {-0.583820034979692642585L, 0.351602874107145458677L, 0.279793729599835250628L, 0.69744837584664266536L},
};

// exp((0.75 + 0.5i) test_matrix() + i I), from mpmath.expm at 60 digits.
const long double ref_complex[4][4][2] = {
    {{0.859817272878956169821L, 0.235905730566517688635L}, {0.170385484495122571254L, 1.81105549618534631456L}, {1.52450645948485839894L, 0.137471600602104259593L}, {-0.625390486727951221028L, 3.5272062874270776001L}},
    {{-1.00061789036524153056L, 0.408897393916639429105L}, {-0.341125109994080942566L, -0.529380338967034113633L}, {-0.816987419245286791412L, -4.58674459809688079246L}, {5.73122690890711433591L, 0.455413105369532148457L}},
    {{2.92483964169929051894L, 0.795856640903318336011L}, {-0.168272976999957138988L, 2.70942646862154964859L}, {-4.41835503725924723464L, 1.98502562992556242789L}, {-1.10319786659615950222L, -4.12667336916424154142L}},
    {{1.10398719430637703165L, -1.92030369049740280346L}, {1.58437112802730908041L, 0.975145580744652375327L}, {0.487118689856587110253L, 3.79294245884104732793L}, {-3.77101284893120038652L, 0.0639125163136402099275L}},
};

/// A class-typed real with no std::numeric_limits specialization and no exp,
/// as in test_eigen_symmetric.cpp: expm must measure its unit roundoff, and a
/// triangular matrix of it takes the general path.
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
static_assert(!detail::expm_impl::adl::has_exp_sinh<unlimited>);

} // namespace

TEST_CASE("unit_roundoff: half the machine epsilon, measured without numeric_limits",
          "[operation][expm]") {
    REQUIRE(detail::unit_roundoff<double>() == std::ldexp(1.0, -53));
    REQUIRE(detail::unit_roundoff<float>() == std::ldexp(1.0f, -24));
    REQUIRE(detail::unit_roundoff<unlimited>().v == std::ldexp(1.0, -53));
    REQUIRE(detail::machine_epsilon<unlimited>().v == std::numeric_limits<double>::epsilon());
}

TEST_CASE("expm: the theta table follows the type's unit roundoff", "[operation][expm]") {
    using detail::expm_impl::table_index;
    using detail::expm_impl::theta_table;
    REQUIRE(theta_table[table_index<float>()].t == 24);
    REQUIRE(theta_table[table_index<double>()].t == 53);
    REQUIRE(theta_table[table_index<unlimited>()].t == 53);
    // long double is x87 extended, double, or binary128 depending on the
    // platform; whichever it is, the table is the first with 2^-t <= u.
    const std::size_t k = table_index<long double>();
    const long double u = detail::unit_roundoff<long double>();
    REQUIRE(std::ldexp(1.0L, -theta_table[k].t) <= u);
    if (k > 0) REQUIRE(std::ldexp(1.0L, -theta_table[k - 1].t) > u);
    // Double: Higham (2005), Table 2.3, except theta_13 = 4.25 from Al-Mohy &
    // Higham (2009), Sec. 5.
    REQUIRE_THAT(theta_table[table_index<double>()].theta[3], WithinAbs(2.097847961257068, 1e-15));
    REQUIRE_THAT(theta_table[table_index<double>()].theta[4], WithinAbs(4.25, 1e-15));
    REQUIRE_THAT(theta_table[table_index<double>()].theta[0], WithinAbs(1.495585217958292e-2, 1e-17));
}

TEST_CASE("expm: zero, empty and 1x1 matrices", "[operation][expm]") {
    mat::dense2D<double> Z(3, 3);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j) Z(i, j) = 0.0;
    auto E = expm(Z);
    REQUIRE(rel_err(E, eye<double>(3)) == 0);

    mat::dense2D<double> empty(0, 0);
    REQUIRE(expm(empty).num_rows() == 0);

    auto one = make<double>({{-2.5}});
    REQUIRE_THAT(expm(one)(0, 0), WithinAbs(std::exp(-2.5), 1e-16));
}

TEST_CASE("expm: diagonal and triangular matrices", "[operation][expm]") {
    auto D = make<double>({{1.0, 0.0, 0.0}, {0.0, -2.0, 0.0}, {0.0, 0.0, 3.5}});
    auto E = expm(D);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            REQUIRE_THAT(E(i, j), WithinAbs(i == j ? std::exp(D(i, i)) : 0.0, 1e-14 * std::exp(3.5)));

    // Nilpotent: the series ends, exp(N) = I + N + N^2/2.
    auto N = make<double>({{0.0, 1.0, 2.0}, {0.0, 0.0, 3.0}, {0.0, 0.0, 0.0}});
    auto EN = expm(N);
    auto ref = make<double>({{1.0, 1.0, 3.5}, {0.0, 1.0, 3.0}, {0.0, 0.0, 1.0}});
    REQUIRE(rel_err(EN, ref) < 1e-15);

    // The same through the general path: unlimited has no exp.
    auto Nu = make<unlimited>({{0.0, 1.0, 2.0}, {0.0, 0.0, 3.0}, {0.0, 0.0, 0.0}});
    auto ENu = expm(Nu);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            REQUIRE_THAT(ENu(i, j).v, WithinAbs(ref(i, j), 1e-15));
}

TEST_CASE("expm: rotation generators give rotations", "[operation][expm]") {
    // exp(t [0 -1; 1 0]) = [cos t, -sin t; sin t, cos t]. Not triangular, so
    // the general path; t = 50 needs several squarings.
    for (double t : {0.3, 2.0, 50.0}) {
        auto G = make<double>({{0.0, -t}, {t, 0.0}});
        auto R = expm(G);
        // The condition number of exp at G is about ||G||, so allow t u.
        const double tol = 64 * t * 1.1e-16 + 1e-16;
        REQUIRE_THAT(R(0, 0), WithinAbs(std::cos(t), tol));
        REQUIRE_THAT(R(0, 1), WithinAbs(-std::sin(t), tol));
        REQUIRE_THAT(R(1, 0), WithinAbs(std::sin(t), tol));
        REQUIRE_THAT(R(1, 1), WithinAbs(std::cos(t), tol));
    }
}

TEST_CASE("expm: against a 60-digit reference, double, float, long double, complex",
          "[operation][expm]") {
    mat::dense2D<long double> ref(4, 4);
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j) ref(i, j) = ref_real[i][j];

    REQUIRE(rel_err(expm(test_matrix<double>()), ref) < 1e-14);
    REQUIRE(rel_err(expm(test_matrix<float>()), ref) < 2e-6);
    // Within a small multiple of long double's own epsilon, whatever it is.
    REQUIRE(rel_err(expm(test_matrix<long double>()), ref) <
            200 * std::numeric_limits<long double>::epsilon());

    auto Au = test_matrix<unlimited>();
    auto Eu = expm(Au);
    mat::dense2D<double> Ed(4, 4);
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j) Ed(i, j) = Eu(i, j).v;
    REQUIRE(rel_err(Ed, ref) < 1e-14);

    using C = std::complex<double>;
    auto A = test_matrix<double>();
    mat::dense2D<C> Ac(4, 4);
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            Ac(i, j) = C(0.75, 0.5) * A(i, j) + (i == j ? C(0.0, 1.0) : C(0.0));
    mat::dense2D<std::complex<long double>> refc(4, 4);
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            refc(i, j) = std::complex<long double>(ref_complex[i][j][0], ref_complex[i][j][1]);
    REQUIRE(rel_err(expm(Ac), refc) < 1e-14);
}

TEST_CASE("expm: exp(A) exp(-A) = I and exp(2A) = exp(A)^2", "[operation][expm]") {
    auto A = test_matrix<double>();
    mat::dense2D<double> mA(4, 4), A2(4, 4);
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j) {
            mA(i, j) = -A(i, j);
            A2(i, j) = 2 * A(i, j);
        }
    const auto E = expm(A);
    REQUIRE(rel_err(product(E, expm(mA)), eye<double>(4)) < 1e-13);
    REQUIRE(rel_err(product(E, E), expm(A2)) < 1e-13);
}

TEST_CASE("expm: no overscaling of a nonnormal matrix (Al-Mohy & Higham 2009)",
          "[operation][expm]") {
    // A = [1 0; b -1] has ||A||_1 = 1 + b but A^2 = I, so every alpha_p is 1.
    // Higham (2005) scales by 2^-s with s ~ log2(b / 5.4) and squares s times;
    // the alpha_p bounds choose s = 0. exp(A) = [e, 0; b sinh(1), 1/e].
    // A is lower triangular, so this is the general path, not the exact
    // triangular recomputation.
    for (double b : {1e4, 1e8, 1e12}) {
        auto A = make<double>({{1.0, 0.0}, {b, -1.0}});
        auto E = expm(A);
        // A few ulps, independent of b.
        REQUIRE_THAT(E(0, 0) / std::exp(1.0), WithinAbs(1.0, 2e-15));
        REQUIRE_THAT(E(1, 1) / std::exp(-1.0), WithinAbs(1.0, 2e-15));
        REQUIRE_THAT(E(0, 1), WithinAbs(0.0, 1e-15));
        REQUIRE_THAT(E(1, 0) / (b * std::sinh(1.0)), WithinAbs(1.0, 2e-15));
    }
    // Upper triangular, scaled so the squaring phase runs: the diagonal and
    // superdiagonal come out exact.
    for (double b : {1e4, 1e8}) {
        auto A = make<double>({{8.0, b}, {0.0, -8.0}});
        auto E = expm(A);
        REQUIRE_THAT(E(0, 0) / std::exp(8.0), WithinAbs(1.0, 2e-15));
        REQUIRE_THAT(E(1, 1) / std::exp(-8.0), WithinAbs(1.0, 2e-15));
        REQUIRE_THAT(E(0, 1) / (b * std::sinh(8.0) / 8.0), WithinAbs(1.0, 2e-15));
        REQUIRE(E(1, 0) == 0.0);
    }
}

TEST_CASE("expm: large norms, with and without pre-scaling", "[operation][expm]") {
    // float's narrow exponent range: ||A|| = 5000 is pre-scaled before the
    // powers are formed, and the result is still a rotation.
    const float t = 5000.0f;
    auto G = make<float>({{0.0f, -t}, {t, 0.0f}});
    auto R = expm(G);
    const double c = std::cos(5000.0), s = std::sin(5000.0);
    const double tol = 5000.0 * 6e-8 * 16;
    REQUIRE_THAT(R(0, 0), WithinAbs(c, tol));
    REQUIRE_THAT(R(1, 0), WithinAbs(s, tol));

    // A large negative-definite symmetric matrix: exp decays to tiny entries.
    auto S = make<double>({{-200.0, 100.0}, {100.0, -200.0}});
    auto ES = expm(S);
    // Eigenvalues -100 (vector [1,1]) and -300 (vector [1,-1]).
    const double e1 = std::exp(-100.0) / 2, e3 = std::exp(-300.0) / 2;
    REQUIRE_THAT(ES(0, 0) / (e1 + e3), WithinAbs(1.0, 1e-12));
    REQUIRE_THAT(ES(0, 1) / (e1 - e3), WithinAbs(1.0, 1e-12));
}

TEST_CASE("expm: errors and non-finite input", "[operation][expm]") {
    mat::dense2D<double> R(2, 3);
    REQUIRE_THROWS_AS(expm(R), std::invalid_argument);

    auto N = make<double>({{1.0, std::numeric_limits<double>::quiet_NaN()}, {0.0, 1.0}});
    auto EN = expm(N);
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 2; ++j) REQUIRE(std::isnan(EN(i, j)));

    auto I = make<double>({{1.0, std::numeric_limits<double>::infinity()}, {0.0, 1.0}});
    auto EI = expm(I);
    REQUIRE(std::isnan(EI(0, 0)));
}

TEST_CASE("expm: an explicit Accumulator gives the same result", "[operation][expm]") {
    auto A = test_matrix<float>();
    auto E = expm(A);
    auto Ed = expm<double>(A);              // products and LU accumulate in double
    REQUIRE(rel_err(Ed, E) < 2e-6);
}
