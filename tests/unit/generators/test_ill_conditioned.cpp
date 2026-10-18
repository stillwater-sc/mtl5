// MTL5 -- gen_dot and gen_sum, the Ogita-Rump-Oishi test-data generators (#611).
//
//   reproducible -- a seed gives the same bits (the draws are built from the raw
//                   mt19937_64 stream, which the standard pins down);
//   GenDot       -- d is x . y rounded to nearest, cond is 2 |x|.|y| / |x.y|,
//                   and the achieved condition number tracks the anticipated
//                   one the way the paper's Table 6.1 reports: typically about
//                   10x larger, with a wide spread;
//   GenSum       -- the 2m summands are the exact split of a GenDot product of
//                   length m, so their exact sum is that dot product, and the
//                   sum's condition number is half the dot product's.
//
// Measured, achieved / anticipated condition number over 500 draws, n = 100..500:
//
//               min      median   max
//   double      0.020    11.2     4.9e3     c = 1 .. 1e100
//   float       0.38     15.7     3.7e3     c = 1 .. 1e30
//   paper       1.5e-3   10.0     1.9e6     double, 50000 draws, c = 1 .. 1e100
//
// The test asserts the median is within [1, 100].
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>

#include <mtl/generators/ill_conditioned.hpp>
#include <mtl/math/exact_accumulator.hpp>

using namespace mtl;

namespace {

template <typename T>
bool same_bits(T a, T b) { return std::memcmp(&a, &b, sizeof(T)) == 0; }

template <typename T>
bool same_vector(const vec::dense_vector<T>& a, const vec::dense_vector<T>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (!same_bits(a(i), b(i))) return false;
    return true;
}

} // namespace

TEMPLATE_TEST_CASE("gen_dot and gen_sum: a seed gives the same bits", "[generators][611]", float,
                   double) {
    using T = TestType;
    const auto a = generators::gen_dot<T>(100, 1e20, std::uint64_t{7});
    const auto b = generators::gen_dot<T>(100, 1e20, std::uint64_t{7});
    const auto c = generators::gen_dot<T>(100, 1e20, std::uint64_t{8});
    CHECK(same_vector(a.x, b.x));
    CHECK(same_vector(a.y, b.y));
    CHECK(same_bits(a.d, b.d));
    CHECK(!same_vector(a.x, c.x));
    const auto s1 = generators::gen_sum<T>(200, 1e20, std::uint64_t{7});
    const auto s2 = generators::gen_sum<T>(200, 1e20, std::uint64_t{7});
    CHECK(same_vector(s1.p, s2.p));
}

TEMPLATE_TEST_CASE("gen_dot: the result rounded to nearest and the achieved condition number",
                   "[generators][611]", float, double) {
    using T = TestType;
    const double cmax = std::is_same_v<T, float> ? 1e30 : 1e100;
    std::mt19937_64 gen(1);
    std::vector<double> ratio;
    for (int t = 0; t < 500; ++t) {
        const std::size_t n = 100 + static_cast<std::size_t>(t % 5) * 100;
        const double c = std::pow(cmax, static_cast<double>(t % 50) / 49.0);
        const auto r = generators::gen_dot<T>(n, c, gen);
        REQUIRE(r.x.size() == n);
        REQUIRE(r.y.size() == n);
        math::exact_accumulator<T> s{}, a{};
        for (std::size_t i = 0; i < n; ++i) {
            REQUIRE(std::isfinite(r.x(i)));
            REQUIRE(std::isfinite(r.y(i)));
            accumulator_add_product(s, r.x(i), r.y(i));
            accumulator_add_product(a, std::fabs(r.x(i)), std::fabs(r.y(i)));
        }
        REQUIRE(same_bits(r.d, accumulator_value(s, std::type_identity<T>{})));
        const double want = 2.0 * accumulator_value(a, std::type_identity<double>{}) /
                            std::fabs(accumulator_value(s, std::type_identity<double>{}));
        REQUIRE(r.cond == want);
        ratio.push_back(r.cond / c);
    }
    std::sort(ratio.begin(), ratio.end());
    const double median = ratio[ratio.size() / 2];
    INFO("achieved / anticipated: min " << ratio.front() << ", median " << median << ", max "
         << ratio.back());
    // The paper's Table 6.1 (double, 50000 draws): min 1.5e-3, median 10.0, max 1.9e6.
    CHECK(median >= 1.0);
    CHECK(median <= 100.0);
    CHECK(ratio.front() >= 1e-6);
}

TEMPLATE_TEST_CASE("gen_sum: the exact split of a GenDot product, permuted", "[generators][611]",
                   float, double) {
    using T = TestType;
    for (std::uint64_t seed : {1u, 2u, 3u}) {
        const auto s = generators::gen_sum<T>(400, 1e25, seed);
        const auto d = generators::gen_dot<T>(200, 1e25, seed);   // the same first draws
        REQUIRE(s.p.size() == 400);
        // sum p == x . y exactly.
        math::exact_accumulator<T> diff{};
        for (std::size_t i = 0; i < 400; ++i) accumulator_add_product(diff, s.p(i), T(1));
        for (std::size_t i = 0; i < 200; ++i) accumulator_add_product(diff, -d.x(i), d.y(i));
        bool zero = diff.special == 0;
        for (auto limb : diff.w) zero = zero && limb == 0;
        REQUIRE(zero);
        REQUIRE(same_bits(s.s, d.d));
        // sum |p_i| = sum |x_i y_i| up to the rounding of the hi parts: half the
        // dot product's condition number, to about u.
        CHECK(std::fabs(s.cond / (d.cond / 2) - 1.0) < 1e-5);
    }
}
