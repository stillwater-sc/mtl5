// MTL5 -- the seeded and engine forms of the random fills and the random
// matrix generators (#844).
//
// A seed must give the same numbers with every compiler and standard library:
// the draws come from the raw std::mt19937_64 stream, which the standard
// specifies bit for bit, not from <random>'s distributions, which it does not.
// The expected values below are that stream's: (gen() >> 11) * 2^-53 for
// [0, 1), and lo (1 - u) + hi u for [lo, hi). A port that drew through
// std::uniform_real_distribution would fail them on some library.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/generators/randorth.hpp>
#include <mtl/generators/randspd.hpp>
#include <mtl/generators/randsvd.hpp>
#include <mtl/generators/randsym.hpp>
#include <mtl/operation/random.hpp>
#include <mtl/operation/spectral_properties.hpp>
#include <mtl/operation/eigenvalue_symmetric.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

using namespace mtl;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

template <typename A>
bool same(const A& x, const A& y) {
    if (x.num_rows() != y.num_rows() || x.num_cols() != y.num_cols()) return false;
    for (std::size_t i = 0; i < x.num_rows(); ++i)
        for (std::size_t j = 0; j < x.num_cols(); ++j)
            if (!(x(i, j) == y(i, j))) return false;
    return true;
}

template <typename A>
bool bitwise_symmetric(const A& x) {
    for (std::size_t i = 0; i < x.num_rows(); ++i)
        for (std::size_t j = 0; j < i; ++j)
            if (!(x(i, j) == x(j, i))) return false;
    return true;
}

} // namespace

TEST_CASE("a seed gives the raw mt19937_64 stream's values", "[random][seed]") {
    // std::mt19937_64(42)'s first output is 13930160852258120406
    std::mt19937_64 g(42);
    REQUIRE(g() == 13930160852258120406ull);

    const auto v = random_vector<double>(4, 0.0, 1.0, std::uint64_t(42));
    REQUIRE(v(0) == 0.75515553295453897);
    REQUIRE(v(1) == 0.63903139385469743);
    REQUIRE(v(2) == 0.7521452007480266);
    REQUIRE(v(3) == 0.13627268363243705);
    REQUIRE(v(0) == double(13930160852258120406ull >> 11) * 0x1p-53);

    const auto f = random_vector<float>(3, -1.0f, 1.0f, std::uint64_t(7));
    REQUIRE(f(0) == 0.508770585f);
    REQUIRE(f(1) == 0.898602366f);
    REQUIRE(f(2) == -0.765171409f);
}

TEST_CASE("random_matrix fills row by row from the seed", "[random][seed]") {
    const auto A = random_matrix<double>(2, 2, 0.0, 1.0, std::uint64_t(42));
    REQUIRE(A(0, 0) == 0.75515553295453897);
    REQUIRE(A(0, 1) == 0.63903139385469743);
    REQUIRE(A(1, 0) == 0.7521452007480266);
    REQUIRE(A(1, 1) == 0.13627268363243705);

    // a column-major matrix gets the same values at the same (i, j)
    mat::dense2D<double, mat::parameters<tag::col_major>> C(2, 2);
    fill_random(C, 0.0, 1.0, std::uint64_t(42));
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 2; ++j) REQUIRE(C(i, j) == A(i, j));
}

TEST_CASE("the engine form continues the caller's stream", "[random][seed]") {
    std::mt19937_64 g(42);
    const auto a = random_vector<double>(2, 0.0, 1.0, g);
    const auto b = random_vector<double>(2, 0.0, 1.0, g);
    const auto all = random_vector<double>(4, 0.0, 1.0, std::uint64_t(42));
    REQUIRE(a(0) == all(0));
    REQUIRE(a(1) == all(1));
    REQUIRE(b(0) == all(2));
    REQUIRE(b(1) == all(3));
}

TEST_CASE("random values stay in [lo, hi)", "[random]") {
    SECTION("float, where a double draw can round up to hi") {
        std::mt19937_64 g(1);
        const auto v = random_vector<float>(20000, 1.0f, 1.0f + 8 * std::numeric_limits<float>::epsilon(), g);
        for (std::size_t i = 0; i < v.size(); ++i) {
            REQUIRE(v(i) >= 1.0f);
            REQUIRE(v(i) < 1.0f + 8 * std::numeric_limits<float>::epsilon());
        }
    }
    SECTION("the whole double range, where hi - lo overflows") {
        const double big = std::numeric_limits<double>::max();
        const auto v = random_vector<double>(1000, -big, big, std::uint64_t(3));
        for (std::size_t i = 0; i < v.size(); ++i) {
            REQUIRE(std::isfinite(v(i)));
            REQUIRE(v(i) >= -big);
            REQUIRE(v(i) < big);
        }
    }
    SECTION("lo == hi gives lo") {
        const auto v = random_vector<double>(3, 2.5, 2.5, std::uint64_t(3));
        for (std::size_t i = 0; i < 3; ++i) REQUIRE(v(i) == 2.5);
    }
    SECTION("hi < lo, NaN and infinite bounds throw") {
        const double inf = std::numeric_limits<double>::infinity();
        REQUIRE_THROWS_AS(random_vector<double>(3, 1.0, 0.0), std::invalid_argument);
        REQUIRE_THROWS_AS(random_vector<double>(3, std::nan(""), 1.0, std::uint64_t(1)), std::invalid_argument);
        // an infinite bound made every draw inf or NaN, so the redraw never ended
        REQUIRE_THROWS_AS(random_vector<double>(1, 0.0, inf, std::uint64_t(42)), std::invalid_argument);
        REQUIRE_THROWS_AS(random_vector<double>(1, -inf, 0.0, std::uint64_t(42)), std::invalid_argument);
        REQUIRE_THROWS_AS(random_vector<double>(1, -inf, inf, std::uint64_t(42)), std::invalid_argument);
    }
    SECTION("the draw is fma(hi, u, lo (1 - u)), whatever the contraction setting") {
        const double lo = -0.3, hi = 2.7;
        const auto v = random_vector<double>(64, lo, hi, std::uint64_t(8));
        std::mt19937_64 g(8);
        for (std::size_t i = 0; i < v.size(); ++i) {
            const double u = double(g() >> 11) * 0x1p-53;
            const double w = std::fma(hi, u, lo * (1.0 - u));
            if (w < hi) REQUIRE(v(i) == w);
            else break;   // a redraw shifts the stream; never happens for this seed
        }
    }
}

TEST_CASE("calls without a seed stay random", "[random]") {
    const auto a = random_vector<double>(8);
    const auto b = random_vector<double>(8);
    bool differ = false;
    for (std::size_t i = 0; i < 8; ++i) differ = differ || a(i) != b(i);
    REQUIRE(differ);
}

TEST_CASE("the random matrix generators are reproducible from a seed", "[generators][random][seed]") {
    using namespace generators;
    REQUIRE(same(randorth<double>(5, std::uint64_t(11)), randorth<double>(5, std::uint64_t(11))));
    REQUIRE_FALSE(same(randorth<double>(5, std::uint64_t(11)), randorth<double>(5, std::uint64_t(12))));

    const std::vector<double> sigma = {4.0, 2.0, 1.0};
    REQUIRE(same(randsvd<double>(4, 3, sigma, std::uint64_t(5)), randsvd<double>(4, 3, sigma, std::uint64_t(5))));
    for (int mode = 1; mode <= 5; ++mode) {
        CAPTURE(mode);
        REQUIRE(same(randsvd<double>(5, 5, 100.0, mode, std::uint64_t(5)),
                     randsvd<double>(5, 5, 100.0, mode, std::uint64_t(5))));
        REQUIRE(same(randsym<double>(5, 100.0, mode, std::uint64_t(5)),
                     randsym<double>(5, 100.0, mode, std::uint64_t(5))));
        REQUIRE(same(randspd<double>(5, 100.0, mode, std::uint64_t(5)),
                     randspd<double>(5, 100.0, mode, std::uint64_t(5))));
    }
    REQUIRE_FALSE(same(randspd<double>(5, 100.0, 3, std::uint64_t(5)),
                       randspd<double>(5, 100.0, 3, std::uint64_t(6))));

    // randorth's random matrix is random_matrix(n, n, -1, 1) from the same stream
    std::mt19937_64 g1(9), g2(9);
    const auto Q = randorth<double>(3, g1);
    const auto A = random_matrix<double>(3, 3, -1.0, 1.0, g2);
    REQUIRE(Q(0, 0) * A(0, 0) > 0.0);   // R(0,0) >= 0, so Q(:,0) has A(:,0)'s signs
    REQUIRE(g1() == g2());               // and it drew exactly n*n values
}

TEST_CASE("randsym and randspd are exactly symmetric", "[generators][randsym][randspd]") {
    // Q*Lambda*Q^T is not bitwise symmetric; the generators average the pairs (#844)
    for (std::uint64_t seed = 0; seed < 5; ++seed) {
        CAPTURE(seed);
        REQUIRE(bitwise_symmetric(generators::randsym<double>(7, 1e3, 3, seed)));
        REQUIRE(bitwise_symmetric(generators::randspd<double>(7, 1e3, 3, seed)));
        REQUIRE(bitwise_symmetric(generators::randsym<double>(6, std::vector<double>{-3, -1, 0, 1, 2, 5}, seed)));
    }
    // eigenvalues near the double maximum: (a + b) / 2 overflowed to -inf
    const auto B = generators::randsym<double>(2, std::vector<double>{-1e308, 1e308}, std::uint64_t(42));
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 2; ++j) REQUIRE(std::isfinite(B(i, j)));
    REQUIRE(bitwise_symmetric(B));
    REQUIRE(bitwise_symmetric(generators::randspd<double>(9, 50.0)));
}

TEST_CASE("randsvd hits kappa in every mode", "[generators][randsvd]") {
    // modes 1 and 3 were the only ones whose condition number was checked
    const double kappa = 1e4;
    for (int mode = 1; mode <= 5; ++mode) {
        CAPTURE(mode);
        const auto A = generators::randsvd<double>(6, 6, kappa, mode, std::uint64_t(mode));
        REQUIRE_THAT(condition_number(A), WithinRel(kappa, 1e-8));
        auto sv = singular_values(A);
        std::sort(sv.begin(), sv.end());
        const auto want = generators::detail::make_sigma(6, kappa, mode == 5 ? 1 : mode);
        if (mode != 5) {
            auto w = want;
            std::sort(w.begin(), w.end());
            for (std::size_t i = 0; i < 6; ++i) REQUIRE_THAT(sv[i], WithinRel(w[i], 1e-8));
        } else {
            for (double s : sv) {
                REQUIRE(s >= 1.0 / kappa * (1 - 1e-10));
                REQUIRE(s <= 1.0 + 1e-10);
            }
        }
    }
    // rectangular, with a seed: min(m, n) singular values
    const auto R = generators::randsvd<double>(7, 4, 50.0, 4, std::uint64_t(2));
    REQUIRE_THAT(condition_number(R), WithinRel(50.0, 1e-8));
}

TEST_CASE("randsym and randspd eigenvalues in modes 2, 4 and 5", "[generators][randsym][randspd]") {
    const double kappa = 1e3;
    for (int mode : {2, 4, 5}) {
        CAPTURE(mode);
        for (bool spd : {false, true}) {
            const auto A = spd ? generators::randspd<double>(6, kappa, mode, std::uint64_t(3))
                               : generators::randsym<double>(6, kappa, mode, std::uint64_t(3));
            const auto e = eigenvalue_symmetric(A);
            double lo = e(0), hi = e(0);
            for (std::size_t i = 0; i < e.size(); ++i) { lo = std::min(lo, e(i)); hi = std::max(hi, e(i)); }
            REQUIRE_THAT(hi, WithinRel(1.0, 1e-10));
            REQUIRE_THAT(lo, WithinRel(1.0 / kappa, 1e-8));
            if (mode == 2) REQUIRE_THAT(e(1), WithinRel(1.0, 1e-10));   // all but one are 1
        }
    }
}

TEST_CASE("make_sigma modes and argument checks", "[generators][randsvd]") {
    using generators::detail::make_sigma;
    const auto s2 = make_sigma(4, 10.0, 2);
    REQUIRE(s2 == std::vector<double>{1.0, 1.0, 1.0, 0.1});
    const auto s4 = make_sigma(3, 10.0, 4);
    REQUIRE_THAT(s4[1], WithinAbs(0.55, 1e-15));
    const auto s1 = make_sigma(3, 10.0, 1);
    REQUIRE(s1 == std::vector<double>{1.0, 0.1, 0.1});
    std::mt19937_64 g(4);
    const auto s5 = make_sigma(50, 1e6, 5, g);
    REQUIRE(s5.front() == 1.0);
    REQUIRE(s5.back() == 1e-6);
    for (double s : s5) { REQUIRE(s >= 1e-6); REQUIRE(s <= 1.0); }

    // invalid modes and kappa < 1 asserted, and fell back to mode 3 under NDEBUG
    REQUIRE_THROWS_AS(make_sigma(4, 10.0, 0), std::invalid_argument);
    REQUIRE_THROWS_AS(make_sigma(4, 10.0, 6), std::invalid_argument);
    REQUIRE_THROWS_AS(make_sigma(4, 0.5, 3), std::invalid_argument);
    REQUIRE_THROWS_AS(make_sigma(4, std::nan(""), 3), std::invalid_argument);
    // mode 5 draws on [log(1/kappa), 0), which never returned for kappa = inf
    REQUIRE_THROWS_AS(make_sigma(4, std::numeric_limits<double>::infinity(), 5), std::invalid_argument);
    REQUIRE_THROWS_AS(generators::randsvd<double>(4, 10.0, 7), std::invalid_argument);
    REQUIRE_THROWS_AS(generators::randspd<double>(4, 0.5, 3), std::invalid_argument);
    REQUIRE_THROWS_AS(generators::randsym<double>(4, 0.5, 3), std::invalid_argument);
}

TEST_CASE("the explicit-spectrum generators check their arguments", "[generators][randsvd][randspd]") {
    REQUIRE_THROWS_AS(generators::randsvd<double>(4, 3, std::vector<double>{1.0, 2.0}), std::invalid_argument);
    REQUIRE_THROWS_AS(generators::randsym<double>(3, std::vector<double>{1.0, 2.0}), std::invalid_argument);
    REQUIRE_THROWS_AS(generators::randspd<double>(3, std::vector<double>{1.0, 2.0}), std::invalid_argument);
    REQUIRE_THROWS_AS(generators::randspd<double>(3, std::vector<double>{1.0, 0.0, 2.0}), std::invalid_argument);
    REQUIRE_THROWS_AS(generators::randspd<double>(3, std::vector<double>{1.0, -1.0, 2.0}), std::invalid_argument);
    REQUIRE_THROWS_AS(generators::randspd<double>(2, std::vector<double>{1.0, std::nan("")}), std::invalid_argument);
}
