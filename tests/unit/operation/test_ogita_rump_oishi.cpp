// MTL5 -- the theorems of Ogita, Rump and Oishi, "Accurate sum and dot
// product", SIAM J. Sci. Comput. 26(6), 2005, asserted over the paper's own
// test data (#611).
//
// Each test case is one numbered result of the paper, checked on every
// instance of a GenDot / GenSum battery (generators/ill_conditioned.hpp): n in
// {50, 200, 1000}, anticipated condition numbers 1 .. 1e100 in double and
// 1 .. 1e30 in float, three seeds each. What is asserted is the inequality,
// never an individual value. The exact quantities -- s, S, |x|.|y| and every
// error |res - s| -- come from math::exact_accumulator.
//
//   Theorem 3.4      TwoSum and TwoProduct are error-free, |y| <= eps |x|
//   (2.4)            the recursive sum, |res - s| <= gamma_{n-1} S
//   Proposition 4.5  Sum2:  |res - s| <= eps |s| + gamma_{n-1}^2 S
//   Corollary 4.7    Sum2's computable bound err: res - err <= s <= res + err
//   (4.14), Lemma 4.9  VecSum keeps the sum, and shrinks sum |p| as stated
//   Proposition 4.10 SumK, K >= 3:
//                    |res - s| <= (eps + 3 gamma_{n-1}^2) |s| + gamma_{2n-2}^K S
//   Proposition 5.5  Dot2:  |res - x.y| <= eps |x.y| + gamma_n^2 |x|.|y|
//   Corollary 5.7    Dot2, relative:  <= eps + gamma_n^2 cond / 2
//   Proposition 5.11 DotK, K >= 3:  |res - x.y| <= (eps + 2 gamma_{4n-2}^2) |x.y|
//                                                  + gamma_{4n-2}^K |x|.|y|
//   (5.11)           DotK, relative:  <= eps + 2 gamma_{4n-2}^2 + gamma_{4n-2}^K cond / 2
//
// eps is the paper's relative rounding error unit, 2^-p (u elsewhere in MTL5);
// eta is the underflow unit, the smallest subnormal. The bounds are evaluated
// in double from exactly computed quantities rounded once, so each carries a
// (1 + 1e-12) factor for that rounding; Corollary 4.7, a bound computed in
// working precision by design, is compared exactly.
//
// Sum2 and Dot2 are checked in both of MTL5's forms: sum_k<2> / dot_k<2>, and
// mtl::sum / mtl::dot with math::compensated_accumulator.
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>

#include <mtl/generators/ill_conditioned.hpp>
#include <mtl/math/accumulator_traits.hpp>
#include <mtl/math/error_free.hpp>
#include <mtl/math/exact_accumulator.hpp>
#include <mtl/operation/dot.hpp>
#include <mtl/operation/sum.hpp>
#include <mtl/operation/sum_dot_k.hpp>
#include <mtl/vec/dense_vector.hpp>

using namespace mtl;

namespace {

template <typename T>
using dvec = mtl::vec::dense_vector<T>;

template <typename T>
constexpr double eps = std::numeric_limits<T>::epsilon() / 2;   // the paper's eps

double gamma(double m, double e) { return m * e / (1.0 - m * e); }

constexpr double slack = 1.0 + 1e-12;

template <typename T>
double cond_max() { return std::is_same_v<T, float> ? 1e30 : 1e100; }

/// The battery's anticipated condition numbers: 10^0, 10^5, ... up to cond_max.
template <typename T>
std::vector<double> conds() {
    std::vector<double> c;
    for (double v = 1; v <= cond_max<T>() * 1.0001; v *= 1e5) c.push_back(v);
    return c;
}

constexpr std::size_t sizes[] = {50, 200, 1000};
constexpr int seeds = 3;

/// An exact sum (or dot product) and its sum of absolute values, rounded to double.
struct exact_pair_d {
    double s, S;
};

template <typename T>
exact_pair_d exact_sum(const dvec<T>& p) {
    math::exact_accumulator<T> s{}, a{};
    for (std::size_t i = 0; i < p.size(); ++i) {
        accumulator_add_product(s, p(i), T(1));
        accumulator_add_product(a, std::fabs(p(i)), T(1));
    }
    return {accumulator_value(s, std::type_identity<double>{}),
            accumulator_value(a, std::type_identity<double>{})};
}

template <typename T>
exact_pair_d exact_dot(const dvec<T>& x, const dvec<T>& y) {
    math::exact_accumulator<T> s{}, a{};
    for (std::size_t i = 0; i < x.size(); ++i) {
        accumulator_add_product(s, x(i), y(i));
        accumulator_add_product(a, std::fabs(x(i)), std::fabs(y(i)));
    }
    return {accumulator_value(s, std::type_identity<double>{}),
            accumulator_value(a, std::type_identity<double>{})};
}

/// |res - sum p|, exactly, rounded to double.
template <typename T>
double sum_error(const dvec<T>& p, T res) {
    math::exact_accumulator<T> d{};
    for (std::size_t i = 0; i < p.size(); ++i) accumulator_add_product(d, p(i), T(1));
    accumulator_add_product(d, -res, T(1));
    return std::fabs(accumulator_value(d, std::type_identity<double>{}));
}

/// |res - x . y|, exactly, rounded to double.
template <typename T>
double dot_error(const dvec<T>& x, const dvec<T>& y, T res) {
    math::exact_accumulator<T> d{};
    for (std::size_t i = 0; i < x.size(); ++i) accumulator_add_product(d, x(i), y(i));
    accumulator_add_product(d, -res, T(1));
    return std::fabs(accumulator_value(d, std::type_identity<double>{}));
}

/// One battery instance and where it came from.
template <typename G>
struct instance {
    G g;
    std::size_t n;
    double c;
    int seed;
};

/// The GenSum battery, generated once per type (GenDot is O(n^2) exact work).
template <typename T>
const std::vector<instance<generators::ill_conditioned_sum<T>>>& sum_instances() {
    static const auto all = [] {
        std::vector<instance<generators::ill_conditioned_sum<T>>> v;
        for (std::size_t n : sizes)
            for (double c : conds<T>())
                for (int seed = 0; seed < seeds; ++seed)
                    v.push_back({generators::gen_sum<T>(n, c, std::uint64_t(1000 * n + 10 * seed) +
                                                                  std::uint64_t(std::log10(c))),
                                 n, c, seed});
        return v;
    }();
    return all;
}

/// The GenDot battery, generated once per type.
template <typename T>
const std::vector<instance<generators::ill_conditioned_dot<T>>>& dot_instances() {
    static const auto all = [] {
        std::vector<instance<generators::ill_conditioned_dot<T>>> v;
        for (std::size_t n : sizes)
            for (double c : conds<T>())
                for (int seed = 0; seed < seeds; ++seed)
                    v.push_back({generators::gen_dot<T>(n, c, std::uint64_t(2000 * n + 10 * seed) +
                                                                  std::uint64_t(std::log10(c))),
                                 n, c, seed});
        return v;
    }();
    return all;
}

/// Run `check(p)` on every GenSum instance.
template <typename T, typename F>
void sum_battery(F&& check) {
    for (const auto& b : sum_instances<T>()) {
        INFO("GenSum n = " << b.n << ", anticipated cond = " << b.c << ", achieved " << b.g.cond
             << ", seed " << b.seed);
        check(b.g.p);
    }
}

/// Run `check(x, y)` on every GenDot instance.
template <typename T, typename F>
void dot_battery(F&& check) {
    for (const auto& b : dot_instances<T>()) {
        INFO("GenDot n = " << b.n << ", anticipated cond = " << b.c << ", achieved " << b.g.cond
             << ", seed " << b.seed);
        check(b.g.x, b.g.y);
    }
}

/// A dense_vector's entries as a std::vector, for the VecSum passes.
template <typename T>
std::vector<T> to_std(const dvec<T>& p) {
    std::vector<T> v(p.size());
    for (std::size_t i = 0; i < p.size(); ++i) v[i] = p(i);
    return v;
}

} // namespace

TEMPLATE_TEST_CASE("Theorem 3.4: TwoSum and TwoProduct are error-free, |y| <= eps |x|",
                   "[611:theorems][611]", float, double) {
    using T = TestType;
    const double e = eps<T>;
    dot_battery<T>([&](const dvec<T>& x, const dvec<T>& y) {
        for (std::size_t i = 0; i + 1 < x.size(); ++i) {
            // (3.2) on neighbouring entries, (3.3) on each product.
            const auto s = math::two_sum(x(i), x(i + 1));
            REQUIRE(s.hi == T(x(i) + x(i + 1)));
            math::exact_accumulator<T> z{};
            accumulator_add_product(z, x(i), T(1));
            accumulator_add_product(z, x(i + 1), T(1));
            accumulator_add_product(z, -s.hi, T(1));
            accumulator_add_product(z, -s.lo, T(1));
            REQUIRE(accumulator_value(z, std::type_identity<double>{}) == 0.0);
            REQUIRE(std::fabs(double(s.lo)) <= e * std::fabs(double(s.hi)));
            const auto p = math::two_product(x(i), y(i));
            const auto q = math::two_product_dekker(x(i), y(i));
            REQUIRE(p.hi == T(x(i) * y(i)));
            REQUIRE(q.hi == p.hi);
            REQUIRE(q.lo == p.lo);
            REQUIRE(std::fabs(double(p.lo)) <= e * std::fabs(double(p.hi)));
            math::exact_accumulator<T> d{};
            accumulator_add_product(d, x(i), y(i));
            accumulator_add_product(d, -p.hi, T(1));
            accumulator_add_product(d, -p.lo, T(1));
            REQUIRE(accumulator_value(d, std::type_identity<double>{}) == 0.0);
        }
    });
}

TEMPLATE_TEST_CASE("(2.4) and Proposition 4.5: the recursive sum and Sum2",
                   "[611:theorems][611]", float, double) {
    using T = TestType;
    const double e = eps<T>;
    sum_battery<T>([&](const dvec<T>& p) {
        const double n = static_cast<double>(p.size());
        const auto [s, S] = exact_sum(p);
        REQUIRE(sum_error(p, mtl::sum_k<1>(p)) <= gamma(n - 1, e) * S * slack);
        const double sum2 = (e * std::fabs(s) + std::pow(gamma(n - 1, e), 2) * S) * slack;
        REQUIRE(sum_error(p, mtl::sum_k<2>(p)) <= sum2);
        REQUIRE(sum_error(p, mtl::sum<math::compensated_accumulator<T>>(p)) <= sum2);
    });
}

TEMPLATE_TEST_CASE("Corollary 4.7: Sum2's computable error bound encloses the sum",
                   "[611:theorems][611]", float, double) {
    using T = TestType;
    const T e = T(eps<T>);
    const T eta = std::numeric_limits<T>::denorm_min();
    sum_battery<T>([&](const dvec<T>& pv) {
        auto p = to_std(pv);
        const std::size_t n = p.size();
        REQUIRE(T(2) * T(n) * e < T(1));
        // Sum2 written out (Algorithm 4.4), then the appended statements, all
        // in working precision.
        detail::vec_sum(p);
        T sum = 0, abs_sum = 0;
        for (std::size_t i = 0; i + 1 < n; ++i) {
            sum += p[i];
            abs_sum += std::fabs(p[i]);
        }
        const T res = sum + p.back();
        REQUIRE(res == mtl::sum_k<2>(pv));
        const T beta = (T(2) * T(n) * e / (T(1) - T(2) * T(n) * e)) * abs_sum;
        const T err = e * std::fabs(res) + (beta + (T(2) * e * e * std::fabs(res) + T(3) * eta));
        // res - err <= s <= res + err, decided exactly.
        math::exact_accumulator<T> lo{}, hi{};
        for (std::size_t i = 0; i < n; ++i) {
            accumulator_add_product(lo, pv(i), T(1));
            accumulator_add_product(hi, pv(i), T(1));
        }
        accumulator_add_product(lo, -res, T(1));
        accumulator_add_product(lo, err, T(1));    // s - res + err >= 0
        accumulator_add_product(hi, -res, T(1));
        accumulator_add_product(hi, -err, T(1));   // s - res - err <= 0
        REQUIRE(accumulator_value(lo, std::type_identity<double>{}) >= 0.0);
        REQUIRE(accumulator_value(hi, std::type_identity<double>{}) <= 0.0);
    });
}

TEMPLATE_TEST_CASE("(4.14) and Lemma 4.9: each VecSum pass keeps the sum and shrinks sum |p|",
                   "[611:theorems][611]", float, double) {
    using T = TestType;
    const double e = eps<T>;
    sum_battery<T>([&](const dvec<T>& pv) {
        auto p = to_std(pv);
        const double n = static_cast<double>(p.size());
        REQUIRE(4 * (n - 1) * e <= 1);
        const auto [s, S0] = exact_sum(pv);
        double S_prev = S0;
        for (int k = 1; k <= 5; ++k) {
            detail::vec_sum(p);
            dvec<T> pk(p.size());
            for (std::size_t i = 0; i < p.size(); ++i) pk(i) = p[i];
            const auto [sk, Sk] = exact_sum(pk);
            INFO("pass k = " << k);
            REQUIRE(sk == s);                                                       // i)
            REQUIRE(Sk <= (std::fabs(s) + 2 * gamma(n - 1, e) * S_prev) * slack);    // (4.14)
            REQUIRE(Sk <= (3 * std::fabs(s) + std::pow(gamma(2 * n - 2, e), k) * S0) * slack);  // iii)
            S_prev = Sk;
        }
    });
}

TEMPLATE_TEST_CASE("Proposition 4.10: SumK, K = 3..6", "[611:theorems][611]", float, double) {
    using T = TestType;
    const double e = eps<T>;
    sum_battery<T>([&](const dvec<T>& p) {
        const double n = static_cast<double>(p.size());
        REQUIRE(4 * n * e <= 1);
        const auto [s, S] = exact_sum(p);
        auto bound = [&](int K) {
            return ((e + 3 * std::pow(gamma(n - 1, e), 2)) * std::fabs(s) +
                    std::pow(gamma(2 * n - 2, e), K) * S) * slack;
        };
        REQUIRE(sum_error(p, mtl::sum_k<3>(p)) <= bound(3));
        REQUIRE(sum_error(p, mtl::sum_k<4>(p)) <= bound(4));
        REQUIRE(sum_error(p, mtl::sum_k<5>(p)) <= bound(5));
        REQUIRE(sum_error(p, mtl::sum_k<6>(p)) <= bound(6));
    });
}

TEMPLATE_TEST_CASE("Proposition 5.5 and Corollary 5.7: Dot2", "[611:theorems][611]", float,
                   double) {
    using T = TestType;
    const double e = eps<T>;
    dot_battery<T>([&](const dvec<T>& x, const dvec<T>& y) {
        const double n = static_cast<double>(x.size());
        const auto [d, A] = exact_dot(x, y);
        const double g2 = std::pow(gamma(n, e), 2);
        const double abs_bound = (e * std::fabs(d) + g2 * A) * slack;            // 5.5
        REQUIRE(d != 0.0);
        const double cond = 2 * A / std::fabs(d);
        const double rel_bound = (e + 0.5 * g2 * cond) * slack;                   // (5.8)
        for (T res : {mtl::dot_k<2>(x, y), mtl::dot<math::compensated_accumulator<T>, T>(x, y)}) {
            const double err = dot_error(x, y, res);
            REQUIRE(err <= abs_bound);
            REQUIRE(err / std::fabs(d) <= rel_bound);
        }
    });
}

TEMPLATE_TEST_CASE("Proposition 5.11 and (5.11): DotK, K = 3..6", "[611:theorems][611]", float,
                   double) {
    using T = TestType;
    const double e = eps<T>;
    dot_battery<T>([&](const dvec<T>& x, const dvec<T>& y) {
        const double n = static_cast<double>(x.size());
        REQUIRE(8 * n * e <= 1);
        const auto [d, A] = exact_dot(x, y);
        REQUIRE(d != 0.0);
        const double g = gamma(4 * n - 2, e);
        const double cond = 2 * A / std::fabs(d);
        auto check = [&](int K, T res) {
            INFO("K = " << K);
            const double err = dot_error(x, y, res);
            REQUIRE(err <= ((e + 2 * g * g) * std::fabs(d) + std::pow(g, K) * A) * slack);
            REQUIRE(err / std::fabs(d) <= (e + 2 * g * g + 0.5 * std::pow(g, K) * cond) * slack);
        };
        check(3, mtl::dot_k<3>(x, y));
        check(4, mtl::dot_k<4>(x, y));
        check(5, mtl::dot_k<5>(x, y));
        check(6, mtl::dot_k<6>(x, y));
    });
}
