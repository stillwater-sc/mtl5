#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/generators/randsym.hpp>
#include <mtl/operation/eigenvalue_symmetric.hpp>
#include <mtl/operation/norms.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

using namespace mtl;
using Catch::Matchers::WithinAbs;

TEST_CASE("randsym dimensions", "[generators][randsym]") {
    std::vector<double> eigs = {5.0, 3.0, 1.0, 0.5};
    auto A = generators::randsym<double>(4, eigs);
    REQUIRE(A.num_rows() == 4);
    REQUIRE(A.num_cols() == 4);
}

TEST_CASE("randsym is symmetric", "[generators][randsym]") {
    std::vector<double> eigs = {4.0, 2.0, 1.0};
    auto A = generators::randsym<double>(3, eigs);

    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            REQUIRE_THAT(A(i, j), WithinAbs(A(j, i), 1e-10));
}

TEST_CASE("randsym recovers prescribed eigenvalues", "[generators][randsym]") {
    std::vector<double> prescribed = {10.0, 5.0, 2.0, 1.0};
    auto A = generators::randsym<double>(4, prescribed);

    auto computed = eigenvalue_symmetric(A, 1e-12);

    // Sort both for comparison
    std::vector<double> expected = prescribed;
    std::sort(expected.begin(), expected.end());

    REQUIRE(computed.size() == 4);
    for (std::size_t i = 0; i < 4; ++i) {
        REQUIRE_THAT(double(computed(i)), WithinAbs(expected[i], 0.1));
    }
}

TEST_CASE("randsym negative eigenvalues", "[generators][randsym]") {
    std::vector<double> prescribed = {-3.0, -1.0, 2.0, 5.0};
    auto A = generators::randsym<double>(4, prescribed);

    // Should still be symmetric
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            REQUIRE_THAT(A(i, j), WithinAbs(A(j, i), 1e-10));

    auto computed = eigenvalue_symmetric(A, 1e-12);
    std::vector<double> expected = prescribed;
    std::sort(expected.begin(), expected.end());

    for (std::size_t i = 0; i < 4; ++i) {
        REQUIRE_THAT(double(computed(i)), WithinAbs(expected[i], 0.1));
    }
}

TEST_CASE("randsym with kappa and mode", "[generators][randsym]") {
    auto A = generators::randsym<double>(5, 50.0, 3);
    REQUIRE(A.num_rows() == 5);
    REQUIRE(A.num_cols() == 5);

    // Should be symmetric
    for (std::size_t i = 0; i < 5; ++i)
        for (std::size_t j = 0; j < 5; ++j)
            REQUIRE_THAT(A(i, j), WithinAbs(A(j, i), 1e-10));
}

// Issue 961: randsym_indefinite(n, kappa, mode, num_negative) -- randsym's
// spectrum with a random subset of num_negative eigenvalues negated.
#include <mtl/operation/spectral_properties.hpp>
#include <cstdint>
#include <random>
#include <stdexcept>

namespace {
// |lambda| sorted, and the magnitudes of the negative eigenvalues sorted.
std::pair<std::vector<double>, std::vector<double>> spectrum_961(const mat::dense2D<double>& A) {
    auto ev = eigenvalue_symmetric(A, 1e-13);
    std::vector<double> mags, neg;
    for (std::size_t i = 0; i < ev.size(); ++i) {
        mags.push_back(std::abs(ev(i)));
        if (ev(i) < 0.0) neg.push_back(-ev(i));
    }
    std::sort(mags.begin(), mags.end());
    std::sort(neg.begin(), neg.end());
    return {mags, neg};
}
} // namespace

TEST_CASE("randsym_indefinite: inertia, spectrum and cond for every mode (issue 961)",
          "[generators][randsym][961]") {
    const std::size_t n = 12;
    const double kappa = 1e3;
    for (int mode = 1; mode <= 5; ++mode) {
        for (const std::size_t p : {std::size_t(0), std::size_t(1), std::size_t(5), n}) {
            CAPTURE(mode, p);
            const std::uint64_t seed = 961 + 10 * std::uint64_t(mode) + p;
            const auto A = generators::randsym_indefinite<double>(n, kappa, mode, p, seed);
            // Bitwise symmetric.
            bool sym = true;
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t j = 0; j < n; ++j) sym = sym && A(i, j) == A(j, i);
            CHECK(sym);
            // The inertia.
            const auto in = inertia(A);
            CHECK(in.positive == n - p);
            CHECK(in.negative == p);
            CHECK(in.zero == 0);
            // |lambda| are make_sigma's values for the same engine state, so
            // cond(A) = kappa. (make_sigma draws first, for mode 5.)
            std::mt19937_64 g(seed);
            auto sigma = generators::detail::make_sigma(n, kappa, mode, g);
            std::sort(sigma.begin(), sigma.end());
            const auto [mags, neg] = spectrum_961(A);
            REQUIRE(mags.size() == n);
            for (std::size_t i = 0; i < n; ++i) CHECK_THAT(mags[i], WithinAbs(sigma[i], 1e-11));
            CHECK_THAT(mags.back() / mags.front(), Catch::Matchers::WithinRel(kappa, 1e-8));
        }
    }
}

TEST_CASE("randsym_indefinite: a seed reproduces the matrix, and 0 negatives is randsym (issue 961)",
          "[generators][randsym][961]") {
    const auto A = generators::randsym_indefinite<double>(9, 50.0, 3, 4, std::uint64_t(7));
    const auto B = generators::randsym_indefinite<double>(9, 50.0, 3, 4, std::uint64_t(7));
    const auto R = generators::randsym<double>(9, 50.0, 3, std::uint64_t(7));
    const auto Z = generators::randsym_indefinite<double>(9, 50.0, 3, 0, std::uint64_t(7));
    bool same_ab = true, same_rz = true;
    for (std::size_t i = 0; i < 9; ++i)
        for (std::size_t j = 0; j < 9; ++j) {
            same_ab = same_ab && A(i, j) == B(i, j);
            same_rz = same_rz && R(i, j) == Z(i, j);
        }
    CHECK(same_ab);
    CHECK(same_rz);
    // The engine form continues the caller's stream.
    std::mt19937_64 g1(7), g2(7);
    const auto C = generators::randsym_indefinite<double>(9, 50.0, 3, 4, g1);
    const auto D = generators::randsym_indefinite<double>(9, 50.0, 3, 4, g2);
    bool same_cd = true;
    for (std::size_t i = 0; i < 9; ++i)
        for (std::size_t j = 0; j < 9; ++j) same_cd = same_cd && C(i, j) == D(i, j) && C(i, j) == A(i, j);
    CHECK(same_cd);
    CHECK(g1() == g2());
}

TEST_CASE("randsym_indefinite: which eigenvalues a seed negates is portable (issue 961)",
          "[generators][randsym][961]") {
    // Mode 3: lambda_i = kappa^(-i/(n-1)), all distinct, so the negative
    // eigenvalues' magnitudes name the negated positions. The subset is drawn
    // from the raw engine stream (random_unit), so these positions are the same
    // with every compiler and standard library; CI runs this on all platforms.
    const std::size_t n = 8;
    const double kappa = 128.0;   // lambda_i = 2^(-i)
    const auto A = generators::randsym_indefinite<double>(n, kappa, 3, 3, std::uint64_t(961));
    const auto [mags, neg] = spectrum_961(A);
    REQUIRE(neg.size() == 3);
    std::vector<int> positions;
    for (const double m : neg) positions.push_back(int(std::lround(-std::log2(m))));
    std::sort(positions.begin(), positions.end());
    INFO("negated positions: " << positions[0] << " " << positions[1] << " " << positions[2]);
    CHECK(positions == std::vector<int>{0, 4, 5});
}

TEST_CASE("randsym_indefinite: argument checks (issue 961)", "[generators][randsym][961]") {
    CHECK_THROWS_AS(generators::randsym_indefinite<double>(4, 10.0, 3, 5, std::uint64_t(1)), std::invalid_argument);
    CHECK_THROWS_AS(generators::randsym_indefinite<double>(4, 0.5, 3, 1, std::uint64_t(1)), std::invalid_argument);
    CHECK_THROWS_AS(generators::randsym_indefinite<double>(4, 10.0, 6, 1, std::uint64_t(1)), std::invalid_argument);
    CHECK_NOTHROW(generators::randsym_indefinite<double>(4, 10.0, 3, 4, std::uint64_t(1)));
    CHECK_NOTHROW(generators::randsym_indefinite<double>(0, 10.0, 3, 0, std::uint64_t(1)));
    // The non-deterministic form: still the inertia asked for.
    const auto A = generators::randsym_indefinite<double>(6, 10.0, 3, 2);
    CHECK(inertia(A).negative == 2);
}
