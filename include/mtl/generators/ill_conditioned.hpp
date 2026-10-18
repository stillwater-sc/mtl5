#pragma once
// MTL5 -- extremely ill-conditioned dot products and sums (#611)
//
// The test-data generators of Ogita, Rump and Oishi, "Accurate sum and dot
// product", SIAM J. Sci. Comput. 26(6), 2005, Section 6:
//
//   gen_dot<T>(n, c, gen)   Algorithm 6.1 (GenDot): x, y of length n whose dot
//                           product has a condition number of about c
//   gen_sum<T>(n, c, gen)   a sum of length n built the paper's way: GenDot of
//                           length n / 2, each product split by two_product
//                           into its rounded value and its error, the 2 (n / 2)
//                           summands randomly permuted
//
// Both return the result rounded to nearest and the condition number actually
// achieved, which is what the paper plots against (its Table 6.1: the achieved
// condition number is typically 10x the anticipated one, with wide spread):
//
//   cond(x.y)    = 2 |x|.|y| / |x.y|        (5.7)
//   cond(sum p)  = sum |p_i| / |sum p_i|    (4.9)
//
// DotExact. GenDot needs "a floating point number nearest to x'y" for each
// partial dot product. math::exact_accumulator is exactly that -- the exact sum,
// rounded once to nearest -- so the generator follows the paper with no
// substitute, and is limited to the types the exact accumulator covers: binary32
// and binary64.
//
// Portable randomness. Each draw is built from the raw std::mt19937_64 stream,
// which the standard specifies bit for bit, not from <random>'s distributions,
// which differ between standard libraries. So a seed gives the same vectors on
// every platform. The paper's rand is uniform on (0, 1); here it is a 53-bit
// uniform on [0, 1).
//
// Range: the entries reach 2^(b/2 + 1) and the products 2^(b + 2), b = log2 c,
// so c must stay below about 2^(max_exponent - 4): 2e37 in float, 1e307 in double.

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <type_traits>
#include <utility>

#include <mtl/math/error_free.hpp>
#include <mtl/math/exact_accumulator.hpp>
#include <mtl/vec/dense_vector.hpp>

namespace mtl::generators {

/// An ill-conditioned dot product: x . y rounded to nearest is `d`, and its
/// achieved condition number 2 |x|.|y| / |x.y| is `cond` (infinite when x . y = 0).
template <typename T>
struct ill_conditioned_dot {
    vec::dense_vector<T> x, y;
    T d;
    double cond;
};

/// An ill-conditioned sum: sum p rounded to nearest is `s`, and its achieved
/// condition number sum |p_i| / |sum p_i| is `cond`.
template <typename T>
struct ill_conditioned_sum {
    vec::dense_vector<T> p;
    T s;
    double cond;
};

namespace ill_conditioned_detail {

/// Uniform on [0, 1): the top 53 bits of one 64-bit draw.
inline double rand01(std::mt19937_64& gen) {
    return static_cast<double>(gen() >> 11) * 0x1p-53;
}

/// Matlab's round: half away from zero.
inline int mround(double v) { return static_cast<int>(std::lround(v)); }

/// Matlab's randperm applied to a pair of vectors: Fisher-Yates on the raw stream.
template <typename A, typename B>
void permute(A& a, B& b, std::mt19937_64& gen) {
    for (std::size_t i = a.size(); i > 1; --i) {
        const auto j = static_cast<std::size_t>(gen() % static_cast<std::uint64_t>(i));
        std::swap(a(i - 1), a(j));
        std::swap(b(i - 1), b(j));
    }
}

/// x(0..m) . y(0..m), exactly, rounded to nearest: the paper's DotExact.
template <typename T>
T dot_exact(const vec::dense_vector<T>& x, const vec::dense_vector<T>& y, std::size_t m) {
    math::exact_accumulator<T> a{};
    for (std::size_t i = 0; i < m; ++i) accumulator_add_product(a, x(i), y(i));
    return accumulator_value(a, std::type_identity<T>{});
}

} // namespace ill_conditioned_detail

/// GenDot (Algorithm 6.1): vectors x, y of length n >= 6 whose dot product has
/// an anticipated condition number c >= 1. The first half is random over the
/// exponent range [0, b/2], b = log2 c; each entry of the second half has its
/// y_i chosen to cancel the exact running dot product down to 2^e_i, with e_i
/// falling linearly from b/2 to 0; then both are permuted alike.
template <typename T = double>
    requires math::exact_accumulable<T>
ill_conditioned_dot<T> gen_dot(std::size_t n, double c, std::mt19937_64& gen) {
    using namespace ill_conditioned_detail;
    assert(n >= 6);
    assert(c >= 1.0);
    const double b = std::log2(c);
    assert(b + 4 <= std::numeric_limits<T>::max_exponent);
    auto sym = [&] { return 2.0 * rand01(gen) - 1.0; };   // 2*rand - 1

    const std::size_t n2 = static_cast<std::size_t>(mround(static_cast<double>(n) / 2));
    ill_conditioned_dot<T> r{vec::dense_vector<T>(n, T(0)), vec::dense_vector<T>(n, T(0)), T(0), 0.0};
    auto& x = r.x;
    auto& y = r.y;

    // e: exponents between 0 and b/2, with b/2 + 1 and 0 made sure to occur.
    for (std::size_t i = 0; i < n2; ++i) {
        int e = mround(rand01(gen) * b / 2);
        if (i == 0) e = mround(b / 2) + 1;
        if (i + 1 == n2) e = 0;
        x(i) = static_cast<T>(sym() * std::ldexp(1.0, e));
        y(i) = static_cast<T>(sym() * std::ldexp(1.0, e));
    }
    // Second half: exponents round(linspace(b/2, 0, n - n2)), and y_i chosen
    // so that x(1:i) . y(1:i) ~ 2^e_i.
    const std::size_t m = n - n2;
    for (std::size_t k = 0; k < m; ++k) {
        const std::size_t i = n2 + k;
        const double t = m > 1 ? static_cast<double>(k) / static_cast<double>(m - 1) : 0.0;
        const int e = mround(b / 2 - (b / 2) * t);
        T xi;
        do xi = static_cast<T>(sym() * std::ldexp(1.0, e)); while (xi == T(0));
        x(i) = xi;
        const T target = static_cast<T>(sym() * std::ldexp(1.0, e));
        y(i) = (target - dot_exact(x, y, i)) / xi;
    }
    permute(x, y, gen);

    math::exact_accumulator<T> s{}, a{};
    for (std::size_t i = 0; i < n; ++i) {
        accumulator_add_product(s, x(i), y(i));
        accumulator_add_product(a, std::fabs(x(i)), std::fabs(y(i)));
    }
    r.d = accumulator_value(s, std::type_identity<T>{});
    const double dd = std::fabs(accumulator_value(s, std::type_identity<double>{}));
    const double aa = accumulator_value(a, std::type_identity<double>{});
    r.cond = dd > 0 ? 2.0 * aa / dd : std::numeric_limits<double>::infinity();
    return r;
}

/// GenDot from a seed.
template <typename T = double>
    requires math::exact_accumulable<T>
ill_conditioned_dot<T> gen_dot(std::size_t n, double c, std::uint64_t seed) {
    std::mt19937_64 gen(seed);
    return gen_dot<T>(n, c, gen);
}

/// An ill-conditioned sum of even length n >= 12, the paper's way: GenDot of
/// length n / 2 at condition number c, each product x_i y_i split exactly by
/// two_product into hi + lo, the n summands permuted. The sum is x . y, so its
/// condition number sum |p_i| / |sum p_i| is about half the dot product's.
template <typename T = double>
    requires math::exact_accumulable<T>
ill_conditioned_sum<T> gen_sum(std::size_t n, double c, std::mt19937_64& gen) {
    assert(n >= 12 && n % 2 == 0);
    const auto dot = gen_dot<T>(n / 2, c, gen);
    ill_conditioned_sum<T> r{vec::dense_vector<T>(n), T(0), 0.0};
    vec::dense_vector<T> other(n, T(0));   // a partner for the shared permutation
    for (std::size_t i = 0; i < n / 2; ++i) {
        const auto t = math::two_product(dot.x(i), dot.y(i));
        r.p(2 * i) = t.hi;
        r.p(2 * i + 1) = t.lo;
    }
    ill_conditioned_detail::permute(r.p, other, gen);

    math::exact_accumulator<T> s{}, a{};
    for (std::size_t i = 0; i < n; ++i) {
        accumulator_add_product(s, r.p(i), T(1));
        accumulator_add_product(a, std::fabs(r.p(i)), T(1));
    }
    r.s = accumulator_value(s, std::type_identity<T>{});
    const double ss = std::fabs(accumulator_value(s, std::type_identity<double>{}));
    const double aa = accumulator_value(a, std::type_identity<double>{});
    r.cond = ss > 0 ? aa / ss : std::numeric_limits<double>::infinity();
    return r;
}

/// GenSum from a seed.
template <typename T = double>
    requires math::exact_accumulable<T>
ill_conditioned_sum<T> gen_sum(std::size_t n, double c, std::uint64_t seed) {
    std::mt19937_64 gen(seed);
    return gen_sum<T>(n, c, gen);
}

} // namespace mtl::generators
