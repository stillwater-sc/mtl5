// MTL5 -- sum_k and dot_k: Ogita-Rump-Oishi SumK and DotK (#611).
//
// Checked here:
//
//   identity  -- sum_k<1> is the recursive sum; sum_k<2> is Sum2, which is
//                mtl::sum<compensated_accumulator>; dot_k<2> is Dot2, which is
//                mtl::dot<compensated_accumulator>: bit for bit;
//   VecSum    -- a pass keeps the exact sum and leaves the rounded sum last;
//   bounds    -- every result within the paper's bound for its K
//                (Propositions 4.5 and 4.10 for sums, 5.5 and 5.11 for dot
//                products), the exact quantities from math::exact_accumulator;
//   K-fold    -- each extra pass buys a factor of about u: the relative error
//                stays at u up to a condition number of about u^-(K-1).
//
// The data are cancelling sums built here (a random half, its exact negation
// and a few small terms, shuffled), with the condition number set by the
// exponent range. The paper's own GenDot / GenSum battery and its theorem
// tests are a later #611 task.
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
#include <utility>
#include <vector>

#include <mtl/math/accumulator_traits.hpp>
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
bool same_bits(T a, T b) { return std::memcmp(&a, &b, sizeof(T)) == 0; }

template <typename T>
constexpr double unit_roundoff = std::numeric_limits<T>::epsilon() / 2;

double gamma(double m, double u) { return m * u / (1.0 - m * u); }

/// A portable random source (the distributions are not portable).
struct rng {
    std::mt19937_64 g;
    explicit rng(std::uint64_t seed) : g(seed) {}
    double symmetric() { return static_cast<double>(g() >> 11) * 0x1p-52 - 1.0; }   // [-1, 1)
    int integer(int lo, int hi) { return lo + static_cast<int>(g() % std::uint64_t(hi - lo + 1)); }
    template <typename A, typename B>
    void shuffle(A& a, B& b) {
        for (std::size_t i = a.size(); i > 1; --i) {
            const std::size_t j = static_cast<std::size_t>(g() % i);
            std::swap(a[i - 1], a[j]);
            std::swap(b[i - 1], b[j]);
        }
    }
};

/// An ill-conditioned dot product of length n: pairs (z, w) and (z, -w) with
/// |z w| up to 2^e_max, whose products cancel exactly, and n / 10 small
/// products that are the whole dot product. Condition number about
/// n 2^e_max. With every w = 1 it is a cancelling sum.
template <typename T>
void cancelling(std::size_t n, int e_max, rng& r, bool sum_only, std::vector<T>& x,
                std::vector<T>& y) {
    const std::size_t small = std::max<std::size_t>(1, n / 10);
    const std::size_t half = (n - small) / 2;
    x.clear();
    y.clear();
    for (std::size_t i = 0; i < half; ++i) {
        const int e = r.integer(0, e_max);
        const T z = T(std::ldexp(r.symmetric(), e - e / 2));
        const T w = sum_only ? T(1) : T(std::ldexp(r.symmetric(), e / 2));
        x.push_back(z); y.push_back(w);
        if (sum_only) { x.push_back(-z); y.push_back(T(1)); }
        else          { x.push_back(z);  y.push_back(-w); }
    }
    while (x.size() < n) {
        x.push_back(T(r.symmetric()));
        y.push_back(sum_only ? T(1) : T(r.symmetric()));
    }
    r.shuffle(x, y);
}

template <typename T>
dvec<T> to_vec(const std::vector<T>& v) {
    dvec<T> out(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) out(i) = v[i];
    return out;
}

/// The exact x . y and sum |x_i y_i|, each rounded to double.
template <typename T>
std::pair<double, double> exact_dot(const std::vector<T>& x, const std::vector<T>& y) {
    math::exact_accumulator<T> s{}, a{};
    for (std::size_t i = 0; i < x.size(); ++i) {
        accumulator_add_product(s, x[i], y[i]);
        accumulator_add_product(a, std::fabs(x[i]), std::fabs(y[i]));
    }
    return {accumulator_value(s, std::type_identity<double>{}),
            accumulator_value(a, std::type_identity<double>{})};
}

/// |res - exact|, exactly, rounded to double.
template <typename T>
double error(const std::vector<T>& x, const std::vector<T>& y, T res) {
    math::exact_accumulator<T> d{};
    for (std::size_t i = 0; i < x.size(); ++i) accumulator_add_product(d, x[i], y[i]);
    accumulator_add_product(d, -res, T(1));
    return std::fabs(accumulator_value(d, std::type_identity<double>{}));
}

/// The paper's bound for sum_k<K> over n terms (K = 1: the recursive sum, (2.4)).
double sum_bound(int K, std::size_t n, double s, double S, double u) {
    const double m = static_cast<double>(n);
    double b;
    if (K == 1)      b = gamma(m - 1, u) * S;
    else if (K == 2) b = u * std::fabs(s) + std::pow(gamma(m - 1, u), 2) * S;
    else             b = (u + 3 * std::pow(gamma(m - 1, u), 2)) * std::fabs(s) +
                         std::pow(gamma(2 * m - 2, u), K) * S;
    return b * (1.0 + 1e-12);   // the exact quantities were rounded to double
}

/// The paper's bound for dot_k<K> over n products.
double dot_bound(int K, std::size_t n, double s, double S, double u) {
    const double m = static_cast<double>(n);
    double b;
    if (K == 2) b = u * std::fabs(s) + std::pow(gamma(m, u), 2) * S;
    else        b = (u + 2 * std::pow(gamma(4 * m - 2, u), 2)) * std::fabs(s) +
                    std::pow(gamma(4 * m - 2, u), K) * S;
    return b * (1.0 + 1e-12);
}

template <int K, typename T>
T sum_k_of(const std::vector<T>& x) { return mtl::sum_k<K>(to_vec(x)); }

template <int K, typename T>
T dot_k_of(const std::vector<T>& x, const std::vector<T>& y) {
    return mtl::dot_k<K>(to_vec(x), to_vec(y));
}

} // namespace

TEMPLATE_TEST_CASE("sum_k and dot_k: K = 1 and K = 2 are the recursive sum, Sum2 and Dot2",
                   "[operation][sum_k][dot_k][611]", float, double) {
    using T = TestType;
    rng r(1);
    std::vector<T> x, y;
    for (std::size_t n : {1u, 2u, 3u, 50u, 1000u}) {
        cancelling<T>(n, 40, r, false, x, y);
        const auto vx = to_vec(x), vy = to_vec(y);
        INFO("n = " << n);
        T plain = 0;
        for (T v : x) plain += v;
        REQUIRE(same_bits(sum_k_of<1>(x), plain));
        REQUIRE(same_bits(sum_k_of<2>(x), mtl::sum<math::compensated_accumulator<T>>(vx)));
        // Below the threading grain: dot runs one serial pass.
        REQUIRE(same_bits(dot_k_of<2>(x, y), mtl::dot<math::compensated_accumulator<T>, T>(vx, vy)));
    }
}

TEMPLATE_TEST_CASE("VecSum keeps the exact sum and leaves the rounded sum last",
                   "[operation][sum_k][611]", float, double) {
    using T = TestType;
    rng r(2);
    std::vector<T> x, y;
    cancelling<T>(500, 60, r, true, x, y);
    auto p = x;
    T plain = 0;
    for (T v : x) plain += v;
    for (int pass = 0; pass < 4; ++pass) {
        detail::vec_sum(p);
        if (pass == 0) REQUIRE(same_bits(p.back(), plain));
        math::exact_accumulator<T> d{};
        for (T v : x) accumulator_add_product(d, v, T(1));
        for (T v : p) accumulator_add_product(d, -v, T(1));
        bool zero = d.special == 0;
        for (auto limb : d.w) zero = zero && limb == 0;
        REQUIRE(zero);
    }
}

TEMPLATE_TEST_CASE("sum_k and dot_k: every result within the paper's bound",
                   "[operation][sum_k][dot_k][611]", float, double) {
    using T = TestType;
    const double u = unit_roundoff<T>;
    constexpr int range = std::numeric_limits<T>::digits * 4;   // cond up to ~2^(4p)
    rng r(3);
    std::vector<T> x, y;
    for (int e = 0; e <= range; e += std::numeric_limits<T>::digits / 4) {
        for (int t = 0; t < 10; ++t) {
            const std::size_t n = 50 + 50 * static_cast<std::size_t>(t);
            INFO("e_max = " << e << ", n = " << n);
            cancelling<T>(n, e, r, true, x, y);
            auto [s, S] = exact_dot(x, y);
            CHECK(error(x, y, sum_k_of<1>(x)) <= sum_bound(1, n, s, S, u));
            CHECK(error(x, y, sum_k_of<2>(x)) <= sum_bound(2, n, s, S, u));
            CHECK(error(x, y, sum_k_of<3>(x)) <= sum_bound(3, n, s, S, u));
            CHECK(error(x, y, sum_k_of<4>(x)) <= sum_bound(4, n, s, S, u));
            CHECK(error(x, y, sum_k_of<5>(x)) <= sum_bound(5, n, s, S, u));

            cancelling<T>(n, e, r, false, x, y);
            std::tie(s, S) = exact_dot(x, y);
            CHECK(error(x, y, dot_k_of<2>(x, y)) <= dot_bound(2, n, s, S, u));
            CHECK(error(x, y, dot_k_of<3>(x, y)) <= dot_bound(3, n, s, S, u));
            CHECK(error(x, y, dot_k_of<4>(x, y)) <= dot_bound(4, n, s, S, u));
            CHECK(error(x, y, dot_k_of<5>(x, y)) <= dot_bound(5, n, s, S, u));
        }
    }
}

TEMPLATE_TEST_CASE("dot_k: each pass buys a factor of u", "[operation][dot_k][611]", float,
                   double) {
    using T = TestType;
    const double u = unit_roundoff<T>;
    constexpr int p = std::numeric_limits<T>::digits;
    rng r(4);
    std::vector<T> x, y;
    // At cond ~ 2^(j p): DotK with K >= j + 2 is still at u, K = j + 1 is
    // about there (u^(j+1) cond = u), and K = j has lost its digits.
    for (int j = 1; j <= 3; ++j) {
        double rel[6] = {0, 0, 0, 0, 0, 0};
        constexpr int runs = 20;
        for (int t = 0; t < runs; ++t) {
            cancelling<T>(200, j * p, r, false, x, y);
            const double s = std::fabs(exact_dot(x, y).first);
            rel[2] = std::max(rel[2], error(x, y, dot_k_of<2>(x, y)) / s);
            rel[3] = std::max(rel[3], error(x, y, dot_k_of<3>(x, y)) / s);
            rel[4] = std::max(rel[4], error(x, y, dot_k_of<4>(x, y)) / s);
            rel[5] = std::max(rel[5], error(x, y, dot_k_of<5>(x, y)) / s);
        }
        INFO("cond ~ 2^" << j * p << ": worst relative error K=2 " << rel[2] << ", K=3 " << rel[3]
             << ", K=4 " << rel[4] << ", K=5 " << rel[5]);
        for (int K = j + 2; K <= 5; ++K) CHECK(rel[K] <= 2 * u);
        if (j >= 2) CHECK(rel[j] > 1e-3);   // K = j passes: the cancellation wins
    }
}

TEMPLATE_TEST_CASE("sum_k and dot_k: empty, one term, non-finite", "[operation][sum_k][dot_k][611]",
                   float, double) {
    using T = TestType;
    const T inf = std::numeric_limits<T>::infinity();
    dvec<T> e(0), one(1, T(3)), two(2);
    CHECK(mtl::sum_k<3>(e) == T(0));
    CHECK(mtl::dot_k<3>(e, e) == T(0));
    CHECK(mtl::dot_k<2>(e, e) == T(0));
    CHECK(mtl::sum_k<3>(one) == T(3));
    CHECK(mtl::dot_k<3>(one, one) == T(9));
    two(0) = std::numeric_limits<T>::max(); two(1) = std::numeric_limits<T>::max();
    CHECK(mtl::sum_k<3>(two) == inf);
    CHECK(mtl::dot_k<3>(two, two) == inf);
    two(1) = -inf;
    CHECK(mtl::sum_k<3>(two) == -inf);
    two(0) = inf;
    CHECK(std::isnan(mtl::sum_k<3>(two)));
    CHECK(mtl::dot_k<4>(two, two) == inf);   // inf*inf + (-inf)*(-inf) = +inf
}
