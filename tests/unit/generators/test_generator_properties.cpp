// MTL5 -- the generators' documented properties against ground truth, their
// edge sizes and a non-double scalar type (#844).
//
// Each generator's header states a property: a spectrum, a determinant, an
// inverse, definiteness or a condition number. These tests check the
// property itself, computed independently (an eigensolver, LU, the SVD),
// not just the entries the generator writes. The Kahan generator built a
// well-conditioned matrix for years because its tests checked only the
// formula it used.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/generators/generators.hpp>
#include <mtl/operation/eigenvalue.hpp>
#include <mtl/operation/eigenvalue_symmetric.hpp>
#include <mtl/operation/factorization_properties.hpp>
#include <mtl/operation/inv.hpp>
#include <mtl/operation/spectral_properties.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <cstddef>
#include <stdexcept>
#include <vector>

using namespace mtl;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

/// A dense copy of an implicit generator (hilbert, lehmer, minij, lotkin, ones).
template <typename G>
mat::dense2D<typename G::value_type> dense(const G& g) {
    mat::dense2D<typename G::value_type> A(g.num_rows(), g.num_cols());
    for (std::size_t i = 0; i < g.num_rows(); ++i)
        for (std::size_t j = 0; j < g.num_cols(); ++j) A(i, j) = g(i, j);
    return A;
}

/// The eigenvalues of a symmetric matrix, ascending, as a std::vector.
template <typename M>
std::vector<double> sym_eigs(const M& A) {
    const auto e = eigenvalue_symmetric(A);
    std::vector<double> v(e.size());
    for (std::size_t i = 0; i < e.size(); ++i) v[i] = static_cast<double>(e(i));
    std::sort(v.begin(), v.end());
    return v;
}

/// The eigenvalues of a general matrix, sorted by real then imaginary part.
template <typename M>
std::vector<std::complex<double>> eigs(const M& A) {
    const auto e = eigenvalue(A, 1e-14);
    std::vector<std::complex<double>> v(e.size());
    for (std::size_t i = 0; i < e.size(); ++i)
        v[i] = std::complex<double>(static_cast<double>(e(i).real()), static_cast<double>(e(i).imag()));
    std::sort(v.begin(), v.end(), [](auto a, auto b) {
        return a.real() < b.real() || (a.real() == b.real() && a.imag() < b.imag());
    });
    return v;
}

double norm1(const mat::dense2D<double>& A) {
    double m = 0;
    for (std::size_t j = 0; j < A.num_cols(); ++j) {
        double s = 0;
        for (std::size_t i = 0; i < A.num_rows(); ++i) s += std::abs(A(i, j));
        m = std::max(m, s);
    }
    return m;
}

} // namespace

// -- spectra -----------------------------------------------------------------

TEST_CASE("clement eigenvalues are n-1, n-3, ..., -(n-1)", "[generators][clement][ground_truth]") {
    for (std::size_t n : {2u, 5u, 6u, 7u}) {
        CAPTURE(n);
        const auto e = sym_eigs(generators::clement<double>(n));
        REQUIRE(e.size() == n);
        for (std::size_t k = 0; k < n; ++k)
            REQUIRE_THAT(e[k], WithinAbs(-double(n - 1) + 2.0 * double(k), 1e-10));
    }
}

TEST_CASE("rosser has its documented spectrum", "[generators][rosser][ground_truth]") {
    // MTL5's header listed {0, 1020, 1020, 1000, 1000, +-10 sqrt(10405),
    // 510 + 100 sqrt(26)}, which sums to 5059.9, not the trace 4040 (#844).
    const double r = 10.0 * std::sqrt(10405.0), q = 100.0 * std::sqrt(26.0);
    const std::vector<double> want = {-r, 0.0, 510.0 - q, 1000.0, 1000.0, 510.0 + q, 1020.0, r};
    const auto e = sym_eigs(generators::rosser<double>());
    REQUIRE(e.size() == 8);
    for (std::size_t k = 0; k < 8; ++k) {
        CAPTURE(k, e[k], want[k]);
        // the eigenvalues near 0 need an absolute tolerance; ||R|| ~ 1e3
        REQUIRE(std::abs(e[k] - want[k]) <= 1e-10 * 1020.0);
    }
    double sum = 0;
    for (double w : want) sum += w;
    REQUIRE_THAT(sum, WithinAbs(4040.0, 1e-9));
}

TEST_CASE("wilkinson W21+ has its nearly equal largest pair", "[generators][wilkinson][ground_truth]") {
    // Wilkinson, The Algebraic Eigenvalue Problem, p. 308: the two largest
    // eigenvalues of W21+ agree to about 15 significant digits.
    const auto e = sym_eigs(generators::wilkinson<double>(21));
    REQUIRE_THAT(e[20], WithinRel(10.746194182903393, 1e-12));
    REQUIRE_THAT(e[19], WithinRel(10.746194182903322, 1e-12));
    REQUIRE(e[20] - e[19] < 1e-12);
    REQUIRE(e[19] - e[18] > 1.0);   // the next pair, near 9.21, is well apart
}

TEST_CASE("wilkinson of even order matches MATLAB", "[generators][wilkinson][ground_truth]") {
    // MATLAB: diag(wilkinson(4)) = [1.5 0.5 0.5 1.5]
    const auto W = generators::wilkinson<double>(4);
    const double d[] = {1.5, 0.5, 0.5, 1.5};
    for (std::size_t i = 0; i < 4; ++i) REQUIRE(W(i, i) == d[i]);
    const auto e = sym_eigs(W);
    // The eigenvectors are symmetric, (a, b, b, a), or antisymmetric,
    // (a, b, -b, -a), which reduces W to [[1.5, 1], [1, 1.5]] (eigenvalues 0.5
    // and 2.5) and [[1.5, 1], [1, -0.5]] (0.5 -+ sqrt(2)).
    const std::vector<double> want = {0.5 - std::sqrt(2.0), 0.5, 0.5 + std::sqrt(2.0), 2.5};
    auto sorted = want;
    std::sort(sorted.begin(), sorted.end());
    for (std::size_t k = 0; k < 4; ++k) REQUIRE_THAT(e[k], WithinAbs(sorted[k], 1e-12));
}

TEST_CASE("frank eigenvalues are real, positive and come in reciprocal pairs",
          "[generators][frank][ground_truth]") {
    const std::size_t n = 6;
    const auto F = generators::frank<double>(n);
    REQUIRE_THAT(determinant(F), WithinAbs(1.0, 1e-10));
    const auto e = eigs(F);
    REQUIRE(e.size() == n);
    for (const auto& z : e) {
        REQUIRE(std::abs(z.imag()) < 1e-8);
        REQUIRE(z.real() > 0.0);
    }
    for (std::size_t k = 0; k < n; ++k)
        REQUIRE_THAT(e[k].real() * e[n - 1 - k].real(), WithinAbs(1.0, 1e-8));
}

TEST_CASE("forsythe eigenvalues are lambda + alpha^(1/n) times the roots of unity",
          "[generators][forsythe][ground_truth]") {
    // n = 4, alpha = 1e-4: lambda + 0.1 * {1, i, -1, -i}
    const auto F = generators::forsythe<double>(4, 1e-4, 2.0);
    const auto e = eigs(F);
    const std::vector<std::complex<double>> want = {{1.9, 0.0}, {2.0, -0.1}, {2.0, 0.1}, {2.1, 0.0}};
    REQUIRE(e.size() == 4);
    for (std::size_t k = 0; k < 4; ++k) {
        CAPTURE(k, e[k], want[k]);
        REQUIRE(std::abs(e[k] - want[k]) < 1e-10);
    }
}

TEST_CASE("companion eigenvalues are the roots of the polynomial",
          "[generators][companion][ground_truth]") {
    // (x-1)(x-2)(x-3)(x-4) = x^4 - 10x^3 + 35x^2 - 50x + 24, low to high
    const auto C = generators::companion<double>({24.0, -50.0, 35.0, -10.0});
    const auto e = eigs(C);
    REQUIRE(e.size() == 4);
    for (std::size_t k = 0; k < 4; ++k) {
        REQUIRE_THAT(e[k].real(), WithinAbs(double(k + 1), 1e-9));
        REQUIRE(std::abs(e[k].imag()) < 1e-9);
    }
}

// -- definiteness, determinants and inverses ----------------------------------

TEST_CASE("pascal has determinant 1 and is SPD", "[generators][pascal][ground_truth]") {
    for (std::size_t n : {1u, 4u, 8u}) {
        CAPTURE(n);
        const auto P = generators::pascal<double>(n);
        REQUIRE(is_spd(P));
        REQUIRE_THAT(determinant(P), WithinAbs(1.0, 1e-8));
    }
}

TEST_CASE("moler, lehmer and minij are SPD", "[generators][ground_truth]") {
    REQUIRE(is_spd(generators::moler<double>(8)));
    REQUIRE(is_spd(generators::moler<double>(8, 0.5)));
    REQUIRE(is_spd(dense(generators::lehmer<double>(8))));
    REQUIRE(is_spd(dense(generators::minij<double>(8))));
    // and the check can fail: clement is symmetric but indefinite
    REQUIRE_FALSE(is_spd(generators::clement<double>(4)));
}

TEST_CASE("minij has its known tridiagonal inverse", "[generators][minij][ground_truth]") {
    // inv(min(i,j)) is tridiagonal: 2 on the diagonal except 1 last, -1 off it
    const std::size_t n = 7;
    const auto X = inv(dense(generators::minij<double>(n)));
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            double want = 0.0;
            if (i == j) want = (i + 1 == n) ? 1.0 : 2.0;
            else if (i + 1 == j || j + 1 == i) want = -1.0;
            CAPTURE(i, j);
            REQUIRE_THAT(X(i, j), WithinAbs(want, 1e-12));
        }
}

TEST_CASE("lehmer has its known tridiagonal inverse", "[generators][lehmer][ground_truth]") {
    // With 1-based k: inv(L)(k,k) = 4k^3 / (4k^2 - 1) for k < n, n^2 / (2n - 1)
    // for k = n, and inv(L)(k,k+1) = -k(k+1) / (2k+1) (Newman and Todd 1958).
    const std::size_t n = 6;
    const auto X = inv(dense(generators::lehmer<double>(n)));
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            const double k = double(std::min(i, j) + 1);
            double want = 0.0;
            if (i == j) want = (i + 1 == n) ? k * k / (2 * k - 1) : 4 * k * k * k / (4 * k * k - 1);
            else if (i + 1 == j || j + 1 == i) want = -k * (k + 1) / (2 * k + 1);
            CAPTURE(i, j);
            REQUIRE_THAT(X(i, j), WithinAbs(want, 1e-11));
        }
}

// -- condition numbers ---------------------------------------------------------

TEST_CASE("hilbert condition numbers match the known values", "[generators][hilbert][ground_truth]") {
    // 2-norm condition numbers of the Hilbert matrix (e.g. Higham, Accuracy
    // and Stability of Numerical Algorithms, Table 28.1)
    REQUIRE_THAT(condition_number(dense(generators::hilbert<double>(4))), WithinRel(1.5514e4, 1e-4));
    REQUIRE_THAT(condition_number(dense(generators::hilbert<double>(6))), WithinRel(1.4951e7, 1e-4));
    REQUIRE_THAT(condition_number(dense(generators::hilbert<double>(8))), WithinRel(1.5258e10, 1e-4));
}

TEST_CASE("kahan condition number grows exponentially", "[generators][kahan][ground_truth]") {
    // K = diag(s^i) U with U = I - c triu(ones, 1), and inv(U)(i,j) =
    // c (1+c)^(j-i-1) above the diagonal, so inv(K) = inv(U) diag(s^-j) in
    // closed form. The old generator scaled the diagonal by 25: kappa_1 was
    // 2.14 for n = 10 and 11.7 for n = 30 (#844).
    const double theta = 1.2, s = std::sin(theta), c = std::cos(theta);
    double k10 = 0;
    for (std::size_t n : {10u, 30u}) {
        CAPTURE(n);
        const auto K = generators::kahan<double>(n, theta, 0.0);
        mat::dense2D<double> X(n, n);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j) {
                const double u = j < i ? 0.0 : j == i ? 1.0 : c * std::pow(1.0 + c, double(j - i - 1));
                X(i, j) = u * std::pow(s, -double(j));
            }
        const double kappa1 = norm1(K) * norm1(X);
        REQUIRE_THAT(norm1(K) * norm1(inv(K)), WithinRel(kappa1, 1e-10));
        if (n == 10) {
            REQUIRE_THAT(kappa1, WithinRel(92.3705, 1e-5));
            k10 = kappa1;
        } else {
            REQUIRE_THAT(kappa1, WithinRel(287937.0, 1e-5));
            REQUIRE(kappa1 / k10 > 1e3);
        }
        // the default perturbation, 25 eps (n - i) on the diagonal, barely moves it
        const auto Kp = generators::kahan<double>(n);
        REQUIRE_THAT(norm1(Kp) * norm1(inv(Kp)), WithinRel(kappa1, 1e-6));
    }
}

TEST_CASE("kahan entries follow MATLAB's gallery('kahan')", "[generators][kahan]") {
    const double theta = 0.7, pert = 25.0, s = std::sin(theta), c = std::cos(theta);
    const double eps = std::numeric_limits<double>::epsilon();
    const std::size_t n = 4;
    const auto K = generators::kahan<double>(n, theta, pert);
    double si = 1.0;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            CAPTURE(i, j);
            if (j < i) REQUIRE(K(i, j) == 0.0);
            else if (j == i) REQUIRE_THAT(K(i, j), WithinRel(si + pert * eps * double(n - i), 1e-15));
            else REQUIRE_THAT(K(i, j), WithinRel(-c * si, 1e-15));
        }
        si *= s;
    }
    REQUIRE(K(0, 0) < 1.0 + 1e-12);   // unit-scaled: the old diagonal started at 25
}

// -- edge sizes ----------------------------------------------------------------

TEST_CASE("generators of order 0 are empty, or throw", "[generators][edge]") {
    auto empty = [](const auto& A) { return A.num_rows() == 0 && A.num_cols() == 0; };
    REQUIRE(empty(generators::kahan<double>(0)));
    REQUIRE(empty(generators::forsythe<double>(0)));
    REQUIRE(empty(generators::clement<double>(0)));
    REQUIRE(empty(generators::frank<double>(0)));
    REQUIRE(empty(generators::pascal<double>(0)));
    REQUIRE(empty(generators::moler<double>(0)));
    REQUIRE(empty(generators::companion<double>({})));
    REQUIRE(empty(generators::vandermonde<double>({})));
    REQUIRE(empty(generators::hilbert<double>(0)));
    REQUIRE(empty(generators::lehmer<double>(0)));
    REQUIRE(empty(generators::minij<double>(0)));
    REQUIRE(empty(generators::lotkin<double>(0)));
    REQUIRE(empty(generators::randorth<double>(0)));
    // with a kappa, n = 0 wrote sigma[0] out of bounds (#844)
    for (int mode = 1; mode <= 5; ++mode) {
        CAPTURE(mode);
        REQUIRE(empty(generators::randsvd<double>(0, 10.0, mode)));
        REQUIRE(empty(generators::randsym<double>(0, 10.0, mode)));
        REQUIRE(empty(generators::randspd<double>(0, 10.0, mode)));
        REQUIRE(generators::detail::make_sigma(0, 10.0, mode).empty());
    }
    const auto R = generators::randsvd<double>(0, 3, 10.0, 3);
    REQUIRE((R.num_rows() == 0 && R.num_cols() == 3));
    REQUIRE_THROWS_AS(generators::wilkinson<double>(0), std::invalid_argument);
    REQUIRE_THROWS_AS(generators::magic<double>(0), std::invalid_argument);
}

TEST_CASE("generators of order 1", "[generators][edge]") {
    REQUIRE(generators::kahan<double>(1, 1.2, 0.0)(0, 0) == 1.0);
    // the corner (n-1, 0) is the diagonal: it used to overwrite lambda (#844)
    REQUIRE(generators::forsythe<double>(1, 1e-10, 3.0)(0, 0) == 3.0);
    REQUIRE(generators::clement<double>(1)(0, 0) == 0.0);
    REQUIRE(generators::frank<double>(1)(0, 0) == 1.0);
    REQUIRE(generators::pascal<double>(1)(0, 0) == 1.0);
    REQUIRE(generators::moler<double>(1)(0, 0) == 1.0);
    REQUIRE(generators::wilkinson<double>(1)(0, 0) == 0.0);
    REQUIRE(generators::companion<double>({-5.0})(0, 0) == 5.0);
    REQUIRE(generators::hilbert<double>(1)(0, 0) == 1.0);
    REQUIRE(generators::lehmer<double>(1)(0, 0) == 1.0);
    REQUIRE(generators::minij<double>(1)(0, 0) == 1.0);
    REQUIRE(generators::magic<double>(1)(0, 0) == 1.0);
    REQUIRE(std::abs(generators::randorth<double>(1)(0, 0)) == 1.0);
    for (int mode = 1; mode <= 5; ++mode) {
        CAPTURE(mode);
        REQUIRE_THAT(std::abs(generators::randsvd<double>(1, 10.0, mode)(0, 0)), WithinAbs(1.0, 1e-15));
        REQUIRE_THAT(generators::randsym<double>(1, 10.0, mode)(0, 0), WithinAbs(1.0, 1e-15));
        REQUIRE_THAT(generators::randspd<double>(1, 10.0, mode)(0, 0), WithinAbs(1.0, 1e-15));
    }
}

TEST_CASE("wilkinson of an even order needs a type that holds 1/2", "[generators][wilkinson][edge]") {
    REQUIRE_THROWS_AS(generators::wilkinson<int>(4), std::invalid_argument);
    const auto W = generators::wilkinson<int>(5);
    REQUIRE(W(0, 0) == 2);
    REQUIRE(W(2, 2) == 0);
    REQUIRE(generators::wilkinson<double>(2)(0, 0) == 0.5);
}

// -- float ---------------------------------------------------------------------

TEST_CASE("generators in float", "[generators][float]") {
    SECTION("pascal is exact") {
        const auto P = generators::pascal<float>(14);   // largest entry C(26,13) = 10400600 < 2^24
        REQUIRE(P(13, 13) == 10400600.0f);
        REQUIRE(P(4, 4) == 70.0f);
    }
    SECTION("hilbert") {
        const generators::hilbert<float> H(5);
        REQUIRE(H(2, 4) == 1.0f / 7.0f);
        REQUIRE_THAT(double(condition_number(dense(generators::hilbert<float>(4)))), WithinRel(1.5514e4, 1e-2));
    }
    SECTION("kahan") {
        const auto K = generators::kahan<float>(10);
        REQUIRE(K(3, 1) == 0.0f);
        const float s = std::sin(1.2f);
        const float eps = std::numeric_limits<float>::epsilon();
        REQUIRE_THAT(double(K(2, 2)), WithinRel(double(s * s + 25.0f * eps * 8.0f), 1e-6));
        mat::dense2D<double> Kd(10, 10);
        for (std::size_t i = 0; i < 10; ++i)
            for (std::size_t j = 0; j < 10; ++j) Kd(i, j) = K(i, j);
        REQUIRE_THAT(norm1(Kd) * norm1(inv(Kd)), WithinRel(92.3705, 1e-4));
    }
    SECTION("clement") {
        const auto e = sym_eigs(generators::clement<float>(6));
        for (std::size_t k = 0; k < 6; ++k)
            REQUIRE_THAT(e[k], WithinAbs(-5.0 + 2.0 * double(k), 1e-4));
    }
}
