#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/eigenvalue_symmetric.hpp>
#include <mtl/operation/eigenvalue.hpp>
#include <mtl/operation/hessenberg.hpp>
#include <mtl/generators/wilkinson.hpp>
#include <mtl/generators/randsym.hpp>
#include <mtl/generators/clement.hpp>
#include <mtl/generators/lehmer.hpp>
#include <mtl/generators/frank.hpp>
#include <mtl/generators/forsythe.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <numbers>
#include <vector>

using namespace mtl;

TEST_CASE("Hessenberg reduction preserves eigenvalues", "[operation][hessenberg]") {
    // 3x3 non-symmetric matrix
    mat::dense2D<double> A(3, 3);
    A(0,0) = 4; A(0,1) = 1; A(0,2) = -1;
    A(1,0) = 2; A(1,1) = 3; A(1,2) =  0;
    A(2,0) = 1; A(2,1) = 0; A(2,2) =  2;

    auto H = hessenberg(A);

    // Verify upper Hessenberg form: H(i,j) == 0 for i > j+1
    for (std::size_t i = 2; i < 3; ++i)
        for (std::size_t j = 0; j + 1 < i; ++j)
            REQUIRE_THAT(H(i, j), Catch::Matchers::WithinAbs(0.0, 1e-10));

    // Trace and determinant (sum/product of eigenvalues) should be preserved
    double trace_A = A(0,0) + A(1,1) + A(2,2);
    double trace_H = H(0,0) + H(1,1) + H(2,2);
    REQUIRE_THAT(trace_H, Catch::Matchers::WithinAbs(trace_A, 1e-10));
}

TEST_CASE("Symmetric eigenvalue: known 2x2 SPD", "[operation][eigenvalue]") {
    mat::dense2D<double> A(2, 2);
    A(0,0) = 2; A(0,1) = 1;
    A(1,0) = 1; A(1,1) = 3;

    auto eigs = eigenvalue_symmetric(A);
    REQUIRE(eigs.size() == 2);

    // Eigenvalues of {{2,1},{1,3}}: (5 +/- sqrt(5))/2
    double expected1 = (5.0 - std::sqrt(5.0)) / 2.0;
    double expected2 = (5.0 + std::sqrt(5.0)) / 2.0;
    REQUIRE_THAT(eigs(0), Catch::Matchers::WithinAbs(expected1, 1e-8));
    REQUIRE_THAT(eigs(1), Catch::Matchers::WithinAbs(expected2, 1e-8));
}

TEST_CASE("Symmetric eigenvalue: 3x3 SPD", "[operation][eigenvalue]") {
    mat::dense2D<double> A(3, 3);
    A(0,0) = 4; A(0,1) = 2; A(0,2) = 1;
    A(1,0) = 2; A(1,1) = 5; A(1,2) = 3;
    A(2,0) = 1; A(2,1) = 3; A(2,2) = 6;

    auto eigs = eigenvalue_symmetric(A);
    REQUIRE(eigs.size() == 3);

    // Sum of eigenvalues should equal trace
    double trace = A(0,0) + A(1,1) + A(2,2);
    double eig_sum = eigs(0) + eigs(1) + eigs(2);
    REQUIRE_THAT(eig_sum, Catch::Matchers::WithinAbs(trace, 1e-8));

    // All eigenvalues should be positive (SPD)
    REQUIRE(eigs(0) > 0);
    REQUIRE(eigs(1) > 0);
    REQUIRE(eigs(2) > 0);
}

TEST_CASE("Symmetric eigenvalue: diagonal matrix", "[operation][eigenvalue]") {
    mat::dense2D<double> A(3, 3);
    A(0,0) = 1; A(0,1) = 0; A(0,2) = 0;
    A(1,0) = 0; A(1,1) = 5; A(1,2) = 0;
    A(2,0) = 0; A(2,1) = 0; A(2,2) = 3;

    auto eigs = eigenvalue_symmetric(A);
    // Sorted: 1, 3, 5
    REQUIRE_THAT(eigs(0), Catch::Matchers::WithinAbs(1.0, 1e-10));
    REQUIRE_THAT(eigs(1), Catch::Matchers::WithinAbs(3.0, 1e-10));
    REQUIRE_THAT(eigs(2), Catch::Matchers::WithinAbs(5.0, 1e-10));
}

TEST_CASE("General eigenvalue: known 2x2", "[operation][eigenvalue]") {
    mat::dense2D<double> A(2, 2);
    A(0,0) = 0; A(0,1) = 1;
    A(1,0) = -2; A(1,1) = -3;

    auto eigs = eigenvalue(A);
    REQUIRE(eigs.size() == 2);

    // Eigenvalues of {{0,1},{-2,-3}} are -1 and -2
    std::vector<double> real_parts = {eigs(0).real(), eigs(1).real()};
    std::sort(real_parts.begin(), real_parts.end());

    REQUIRE_THAT(real_parts[0], Catch::Matchers::WithinAbs(-2.0, 1e-8));
    REQUIRE_THAT(real_parts[1], Catch::Matchers::WithinAbs(-1.0, 1e-8));
    REQUIRE_THAT(eigs(0).imag(), Catch::Matchers::WithinAbs(0.0, 1e-8));
    REQUIRE_THAT(eigs(1).imag(), Catch::Matchers::WithinAbs(0.0, 1e-8));
}

TEST_CASE("General eigenvalue: complex conjugate pair", "[operation][eigenvalue]") {
    // A = {{0, -1}, {1, 0}} has eigenvalues +/- i
    mat::dense2D<double> A(2, 2);
    A(0,0) = 0; A(0,1) = -1;
    A(1,0) = 1; A(1,1) =  0;

    auto eigs = eigenvalue(A);
    REQUIRE(eigs.size() == 2);

    // Should have purely imaginary eigenvalues +/- i
    std::vector<double> imag_parts = {eigs(0).imag(), eigs(1).imag()};
    std::sort(imag_parts.begin(), imag_parts.end());

    REQUIRE_THAT(eigs(0).real(), Catch::Matchers::WithinAbs(0.0, 1e-8));
    REQUIRE_THAT(eigs(1).real(), Catch::Matchers::WithinAbs(0.0, 1e-8));
    REQUIRE_THAT(imag_parts[0], Catch::Matchers::WithinAbs(-1.0, 1e-8));
    REQUIRE_THAT(imag_parts[1], Catch::Matchers::WithinAbs(1.0, 1e-8));
}

// -- Generator-based eigenvalue tests ---------------------------------

TEST_CASE("Symmetric eigenvalues of Wilkinson W7", "[operation][eigenvalue][generator]") {
    // Wilkinson has nearly-equal eigenvalue pairs -- stress test
    constexpr std::size_t n = 7;
    auto W = generators::wilkinson<double>(n);

    auto eigs = eigenvalue_symmetric(W);
    REQUIRE(eigs.size() == n);

    // Sum of eigenvalues = trace = sum of |i - m| for m=3
    // trace = 3+2+1+0+1+2+3 = 12
    double eig_sum = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        eig_sum += eigs(i);
    REQUIRE_THAT(eig_sum, Catch::Matchers::WithinAbs(12.0, 1e-8));

    // Eigenvalues are returned sorted -- verify monotone
    for (std::size_t i = 0; i + 1 < n; ++i)
        REQUIRE(eigs(i) <= eigs(i + 1) + 1e-12);
}

TEST_CASE("Symmetric eigenvalues of randsym: ground truth", "[operation][eigenvalue][generator]") {
    // Prescribed eigenvalues must be recovered
    std::vector<double> expected_eigs = {10.0, 5.0, 3.0, 2.0, 1.0};
    constexpr std::size_t n = 5;
    auto A = generators::randsym<double>(n, expected_eigs);

    auto eigs = eigenvalue_symmetric(A);
    REQUIRE(eigs.size() == n);

    // Sort both and compare pairwise
    std::sort(expected_eigs.begin(), expected_eigs.end());
    std::vector<double> computed(n);
    for (std::size_t i = 0; i < n; ++i)
        computed[i] = eigs(i);
    std::sort(computed.begin(), computed.end());

    for (std::size_t i = 0; i < n; ++i)
        REQUIRE_THAT(computed[i], Catch::Matchers::WithinAbs(expected_eigs[i], 1e-8));
}

TEST_CASE("Symmetric eigenvalues of Clement matrix", "[operation][eigenvalue][generator]") {
    // Clement n=6: eigenvalues are +/-5, +/-3, +/-1
    constexpr std::size_t n = 6;
    auto C = generators::clement<double>(n);

    auto eigs = eigenvalue_symmetric(C);
    REQUIRE(eigs.size() == n);

    std::vector<double> computed(n);
    for (std::size_t i = 0; i < n; ++i)
        computed[i] = eigs(i);
    std::sort(computed.begin(), computed.end());

    std::vector<double> expected = {-5.0, -3.0, -1.0, 1.0, 3.0, 5.0};
    for (std::size_t i = 0; i < n; ++i)
        REQUIRE_THAT(computed[i], Catch::Matchers::WithinAbs(expected[i], 1e-8));
}

TEST_CASE("Symmetric eigenvalues of Lehmer matrix", "[operation][eigenvalue][generator]") {
    // Lehmer is SPD -- all eigenvalues positive, trace = n
    constexpr std::size_t n = 5;
    generators::lehmer<double> L_gen(n);
    mat::dense2D<double> A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A(i, j) = L_gen(i, j);

    auto eigs = eigenvalue_symmetric(A);
    REQUIRE(eigs.size() == n);

    // All eigenvalues positive (SPD)
    for (std::size_t i = 0; i < n; ++i)
        REQUIRE(eigs(i) > 0.0);

    // Sum = trace = n (diagonal is all 1s)
    double eig_sum = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        eig_sum += eigs(i);
    REQUIRE_THAT(eig_sum, Catch::Matchers::WithinAbs(static_cast<double>(n), 1e-8));
}

TEST_CASE("General eigenvalues of Frank matrix", "[operation][eigenvalue][generator]") {
    // Frank matrix has known positive real eigenvalues
    constexpr std::size_t n = 5;
    auto F = generators::frank<double>(n);

    auto eigs = eigenvalue(F);
    REQUIRE(eigs.size() == n);

    // All eigenvalues should be real and positive
    for (std::size_t i = 0; i < n; ++i) {
        REQUIRE_THAT(eigs(i).imag(), Catch::Matchers::WithinAbs(0.0, 1e-6));
        REQUIRE(eigs(i).real() > 0.0);
    }

    // Sum of eigenvalues = trace
    double trace = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        trace += F(i, i);

    double eig_sum = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        eig_sum += eigs(i).real();
    REQUIRE_THAT(eig_sum, Catch::Matchers::WithinAbs(trace, 1e-8));
}

namespace {

// Greedy multiset comparison of two complex spectra within `tol`.
// Returns true iff every expected eigenvalue is matched to a distinct computed
// one. Guards against the single-shift QR regression where all eigenvalues
// collapsed to the shift value while trace/sum-based checks still passed.
bool spectra_match(std::vector<std::complex<double>> computed,
                   std::vector<std::complex<double>> expected, double tol) {
    if (computed.size() != expected.size()) return false;
    for (const auto& e : expected) {
        std::size_t best = computed.size();
        double bestd = tol;
        for (std::size_t j = 0; j < computed.size(); ++j) {
            double d = std::abs(computed[j] - e);
            if (d <= bestd) { bestd = d; best = j; }
        }
        if (best == computed.size()) return false;
        computed.erase(computed.begin() + best);
    }
    return true;
}

// Closed-form Forsythe spectrum: lambda + alpha^{1/n} * exp(2*pi*i*k/n).
std::vector<std::complex<double>> forsythe_spectrum(std::size_t n, double alpha, double lambda) {
    std::vector<std::complex<double>> s(n);
    const double r = std::pow(alpha, 1.0 / static_cast<double>(n));
    for (std::size_t k = 0; k < n; ++k) {
        double th = 2.0 * std::numbers::pi * static_cast<double>(k) / static_cast<double>(n);
        s[k] = std::complex<double>(lambda + r * std::cos(th), r * std::sin(th));
    }
    return s;
}

} // namespace

TEST_CASE("General eigenvalues of Forsythe matrix (full spectrum)", "[operation][eigenvalue][generator]") {
    // Forsythe is a strongly non-normal companion matrix whose eigenvalues lie
    // on a circle of radius alpha^{1/n} around lambda. This is the regression
    // for the Francis double-shift fix (#209): the old single-shift QR returned
    // all eigenvalues equal to the shift (e.g. {3,3,3,3,3}), which slipped the
    // previous trace/distance-only checks. Compare the whole spectrum.
    for (std::size_t n : {4u, 5u, 6u, 8u}) {
        auto F = generators::forsythe<double>(n, 0.5, 3.0);
        auto eigs = eigenvalue(F);
        REQUIRE(eigs.size() == n);

        std::vector<std::complex<double>> computed(n);
        for (std::size_t i = 0; i < n; ++i) computed[i] = eigs(i);

        auto expected = forsythe_spectrum(n, 0.5, 3.0);
        REQUIRE(spectra_match(computed, expected, 1e-8));
    }
}
