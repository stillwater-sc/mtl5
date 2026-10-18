#pragma once
// MTL5 -- SumK and DotK: sum and dot product as if in K-fold precision (#611)
//
// Ogita, Rump and Oishi, "Accurate sum and dot product", SIAM J. Sci. Comput.
// 26(6), 2005, Algorithms 4.3 (VecSum), 4.8 (SumK), 5.3 (Dot2) and 5.10
// (DotK), built on the error-free transformations of math/error_free.hpp.
//
//   sum_k<K>(x)      the sum of x, as if computed in K-fold working precision
//                    and rounded once
//   dot_k<K>(x, y)   the dot product of x and y, likewise
//
// VecSum runs two_sum down a vector: the vector's exact sum is unchanged, its
// last entry becomes the rounded sum, and the others become the rounding
// errors. The errors are small exactly when the sum cancelled, so each pass
// improves the condition number of what is left by a factor of about eps.
// SumK makes K - 1 passes and then sums plainly. The bounds (eps = u, the
// unit roundoff; gamma_m = m u / (1 - m u); s the exact sum, S = sum |p_i|):
//
//   sum_k<K>, K >= 3  |res - s| <= (u + 3 gamma_{n-1}^2) |s| + gamma_{2n-2}^K S
//                     (Proposition 4.10, assuming 4 n u <= 1)
//   sum_k<2> = Sum2   |res - s| <= u |s| + gamma_{n-1}^2 S      (Proposition 4.5)
//   dot_k<K>, K >= 3  |res - x.y| <= (u + 2 gamma_{4n-2}^2) |x.y|
//                                    + gamma_{4n-2}^K |x|.|y|
//                     (Proposition 5.11, assuming 8 n u <= 1, no underflow)
//   dot_k<2> = Dot2   |res - x.y| <= u |x.y| + gamma_n^2 |x|.|y|  (Proposition 5.5)
//
// So the relative error is u plus about (phi u)^K times the condition number:
// each extra pass buys another factor of u.
//
// NOT AN ACCUMULATOR. SumK and DotK keep every error term -- n words for a sum,
// 2n for a dot product -- and make K - 1 passes over them, so they work on whole
// vectors, not through accumulator_traits. K = 2 needs no storage and IS an
// accumulator: math::compensated_accumulator, and dot_k<2> computes the same
// loop (so mtl::dot<compensated_accumulator<T>, T> and dot_k<2> agree bit for
// bit where the dot is not threaded). sum_k<2> is Sum2, which is also
// mtl::sum<compensated_accumulator<T>>.
//
// COST. sum_k<K>: (6K - 5)(n - 1) flops (Proposition 4.10). dot_k<K>, K >= 3:
// 8n for the products and the running sum, then SumK over 2n terms with K - 1
// passes, about (12K - 14) n in all with the FMA two_product (the paper counts
// 6K(2n - 1) + n + 5 with Dekker's). One allocation of n (or 2n) T.
//
// Built-in binary IEEE element types only, and the strict IEEE semantics of
// error_free.hpp: no -ffast-math. The products use the FMA two_product; Dekker's
// form (the paper's) gives the same pairs.

#include <cassert>
#include <cstddef>
#include <type_traits>
#include <vector>

#include <mtl/concepts/collection.hpp>
#include <mtl/concepts/vector.hpp>
#include <mtl/math/error_free.hpp>

namespace mtl {

namespace detail {

/// VecSum (Algorithm 4.3): p_i, p_{i-1} = TwoSum(p_i, p_{i-1}) for i = 2..n.
template <typename T>
void vec_sum(std::vector<T>& p) {
    for (std::size_t i = 1; i < p.size(); ++i) {
        const auto s = math::two_sum(p[i], p[i - 1]);
        p[i] = s.hi;
        p[i - 1] = s.lo;
    }
}

/// SumK (Algorithm 4.8) on a vector it may overwrite: K - 1 VecSum passes,
/// then fl(sum_{i<n} p_i + p_n).
template <typename T>
T sum_k_inplace(std::vector<T>& p, int K) {
    if (p.empty()) return T(0);
    for (int k = 1; k < K; ++k) vec_sum(p);
    T res = T(0);
    for (std::size_t i = 0; i + 1 < p.size(); ++i) res += p[i];
    return res + p.back();
}

} // namespace detail

/// The sum of `x` as if computed in K-fold working precision and rounded
/// once: Ogita-Rump-Oishi SumK (K >= 2), and the plain recursive sum for
/// K = 1. See the header for the bound.
template <int K, Collection C>
    requires(K >= 1) && math::binary_floating_point<typename C::value_type>
auto sum_k(const C& x) {
    using T = typename C::value_type;
    std::vector<T> p;
    for (auto it = x.begin(); it != x.end(); ++it) p.push_back(*it);
    return detail::sum_k_inplace(p, K);
}

/// The dot product of `x` and `y` as if computed in K-fold working precision
/// and rounded once: Dot2 (Algorithm 5.3) for K = 2, DotK (Algorithm 5.10) for
/// K >= 3. Real vectors of one binary IEEE type. See the header for the bound.
template <int K, Vector V1, Vector V2>
    requires(K >= 2) && std::is_same_v<typename V1::value_type, typename V2::value_type> &&
            math::binary_floating_point<typename V1::value_type>
auto dot_k(const V1& x, const V2& y) {
    using T = typename V1::value_type;
    assert(x.size() == y.size());
    const std::size_t n = x.size();
    if (n == 0) return T(0);
    if constexpr (K == 2) {
        // Dot2: the errors go straight into s, in the order s + (q + r).
        const auto first = math::two_product(x(0), y(0));
        T p = first.hi, s = first.lo;
        for (std::size_t i = 1; i < n; ++i) {
            const auto h = math::two_product(x(i), y(i));
            const auto q = math::two_sum(p, h.hi);
            p = q.hi;
            s += q.lo + h.lo;
        }
        return p + s;
    } else {
        // DotK: r_1..r_n the product errors, r_{n+1}..r_{2n-1} the sum
        // errors, r_{2n} the rounded dot; then SumK(r, K - 1).
        std::vector<T> r(2 * n);
        const auto first = math::two_product(x(0), y(0));
        T p = first.hi;
        r[0] = first.lo;
        for (std::size_t i = 1; i < n; ++i) {
            const auto h = math::two_product(x(i), y(i));
            r[i] = h.lo;
            const auto q = math::two_sum(p, h.hi);
            p = q.hi;
            r[n + i - 1] = q.lo;
        }
        r[2 * n - 1] = p;
        return detail::sum_k_inplace(r, K - 1);
    }
}

} // namespace mtl
