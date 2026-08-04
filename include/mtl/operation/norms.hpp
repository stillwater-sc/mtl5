#pragma once
// MTL5 -- Vector/matrix norms: one_norm, two_norm, infinity_norm, frobenius_norm
// Optional BLAS dispatch when MTL5_HAS_BLAS is defined and types qualify.
#include <algorithm>
#include <cmath>
#include <cassert>
#include <cstddef>
#include <limits>
#include <mtl/concepts/vector.hpp>
#include <mtl/concepts/matrix.hpp>
#include <mtl/concepts/magnitude.hpp>
#include <mtl/math/identity.hpp>
#include <mtl/math/accumulator_traits.hpp>
#include <mtl/detail/thread_pool.hpp>
#include <mtl/interface/dispatch_traits.hpp>
#include <type_traits>
#include <mtl/simd/algorithm.hpp>
#ifdef MTL5_HAS_BLAS
#include <mtl/interface/blas.hpp>
#endif

namespace mtl {

// -- Vector norms --------------------------------------------------------

/// one_norm(v) = sum(|v[i]|)
template <Vector V>
auto one_norm(const V& v) {
    using mag_t = magnitude_t<typename V::value_type>;
    auto acc = math::zero<mag_t>();
    for (typename V::size_type i = 0; i < v.size(); ++i) {
        using std::abs;
        acc += abs(v(i));
    }
    return acc;
}

/// two_norm(v) = sqrt(sum(|v[i]|^2)).
///
/// Mixed precision: pass an explicit `Accumulator` to sum the squares in a wider
/// precision than the element magnitude (e.g. `two_norm<double>(v)` over a
/// bfloat16/float vector), guarding the long reduction against overflow and
/// precision loss. Default `Accumulator = void` keeps the BLAS / SIMD / loop
/// dispatch unchanged.
///
/// `Result` is the delivery type, as in `dot` (#379). It defaults to void,
/// meaning the element magnitude type -- today's behaviour, byte for byte.
///
/// Note what `Result` governs: NOT just the final cast. It also selects the
/// type the accumulator is rounded out to before the square root. That
/// distinction is the whole feature. `accumulator_round_type_t<Acc, Mag>` maps a
/// non-arithmetic accumulator (a quire, a compensated sum) to `Mag`, so leaving
/// it at the magnitude type would round an exact accumulation down to the
/// element's precision BEFORE the sqrt, and a wider return type could not
/// recover it -- `two_norm<quire, double>` would then be no better than
/// `two_norm<>`, and strictly worse than `two_norm<double, double>`. Measured
/// on a 20000-element float vector: 2.372e-08 relative error that way, against
/// 0.0 when `Result` feeds the round-out as it does here.
template <typename Accumulator = void, typename Result = void, Vector V>
auto two_norm(const V& v) {
    using mag_t = magnitude_t<typename V::value_type>;
    static_assert(!std::is_void_v<Accumulator> || std::is_void_v<Result>,
        "two_norm: Result without an Accumulator would be silently ignored -- the "
        "default path dispatches to BLAS/SIMD and has no accumulator to round out. "
        "Pass an Accumulator too, e.g. two_norm<double, double>(v) (#379).");
    if constexpr (!std::is_void_v<Accumulator>) {
        using out_t = std::conditional_t<std::is_void_v<Result>, mag_t, Result>;
        using AT = math::accumulator_traits<Accumulator, mag_t>;
        Accumulator acc{};
        AT::clear(acc);
        for (typename V::size_type i = 0; i < v.size(); ++i) {
            using std::abs;
            mag_t a = abs(v(i));
            AT::add_product(acc, a, a);               // acc += a*a in Accumulator
        }
        using std::sqrt;
        // Round out to the accumulator's own arithmetic precision, not to the
        // accumulator TYPE: the latter is a no-op for a plain arithmetic
        // accumulator but yields an fma_accumulator or a quire otherwise, and
        // neither has a sqrt (#324).
        //
        // The second argument is `out_t`, not `mag_t` (#379): for a custom
        // accumulator this is what decides the precision the exact sum is
        // rounded to before the sqrt, and hence whether the accumulator is
        // observable at all.
        using round_t = math::accumulator_round_type_t<Accumulator, out_t>;
        return static_cast<out_t>(sqrt(AT::template value<round_t>(acc)));
    } else {
#ifdef MTL5_HAS_BLAS
    // BLAS takes int; fall back to the loop for vectors larger than INT_MAX.
    if constexpr (interface::BlasDenseVector<V>) {
        if (v.size() <= static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            return interface::blas::nrm2(static_cast<int>(v.size()), v.data(), 1);
        }
    }
#endif
    // Native SIMD path for contiguous real float/double (abs is the identity).
    if constexpr (interface::BlasDenseVector<V>) {
        using std::sqrt;
        // Parallel reduction of the sum of squares (deterministic per thread
        // count, not serial-bit-identical). Serial by default.
        using T = typename V::value_type;
        const T* d = v.data();
        const T ss = detail::thread_pool::instance().parallel_reduce<T>(
            v.size(), /*grain=*/std::size_t{65536},
            [&](std::size_t lo, std::size_t hi) { return simd::reduce_sum_squares<T>(d + lo, hi - lo); });
        return sqrt(ss);
    } else {
        auto acc = math::zero<mag_t>();
        for (typename V::size_type i = 0; i < v.size(); ++i) {
            using std::abs;
            auto a = abs(v(i));
            acc += a * a;
        }
        using std::sqrt;
        return sqrt(acc);
    }
    }
}

/// infinity_norm(v) = max(|v[i]|)
template <Vector V>
auto infinity_norm(const V& v) {
    using mag_t = magnitude_t<typename V::value_type>;
    auto result = math::zero<mag_t>();
    for (typename V::size_type i = 0; i < v.size(); ++i) {
        using std::abs;
        auto a = abs(v(i));
        if (a > result) result = a;
    }
    return result;
}

// -- Matrix norms --------------------------------------------------------

/// frobenius_norm(m) = sqrt(sum(|m[i,j]|^2)).
///
/// Mixed precision: pass an explicit `Accumulator` to sum the squares in a wider
/// precision than the element magnitude. Default `Accumulator = void` is
/// unchanged. `Result` is the delivery type and defaults to the element
/// magnitude type; it also selects the round-out type ahead of the square root
/// -- see the note on two_norm (#379).
template <typename Accumulator = void, typename Result = void, Matrix M>
auto frobenius_norm(const M& m) {
    using mag_t = magnitude_t<typename M::value_type>;
    static_assert(!std::is_void_v<Accumulator> || std::is_void_v<Result>,
        "frobenius_norm: Result without an Accumulator would be silently ignored. "
        "Pass an Accumulator too, e.g. frobenius_norm<double, double>(m) (#379).");
    if constexpr (!std::is_void_v<Accumulator>) {
        using out_t = std::conditional_t<std::is_void_v<Result>, mag_t, Result>;
        using AT = math::accumulator_traits<Accumulator, mag_t>;
        Accumulator acc{};
        AT::clear(acc);
        for (typename M::size_type r = 0; r < m.num_rows(); ++r) {
            for (typename M::size_type c = 0; c < m.num_cols(); ++c) {
                using std::abs;
                mag_t a = abs(m(r, c));
                AT::add_product(acc, a, a);
            }
        }
        using std::sqrt;
        // Round out to the accumulator's own arithmetic precision, not to the
        // accumulator TYPE: the latter is a no-op for a plain arithmetic
        // accumulator but yields an fma_accumulator or a quire otherwise, and
        // neither has a sqrt (#324).
        using round_t = math::accumulator_round_type_t<Accumulator, out_t>;
        return static_cast<out_t>(sqrt(AT::template value<round_t>(acc)));
    } else {
        auto acc = math::zero<mag_t>();
        for (typename M::size_type r = 0; r < m.num_rows(); ++r) {
            for (typename M::size_type c = 0; c < m.num_cols(); ++c) {
                using std::abs;
                auto a = abs(m(r, c));
                acc += a * a;
            }
        }
        using std::sqrt;
        return sqrt(acc);
    }
}

/// one_norm(m) = max column sum of |m[i,j]|
template <Matrix M>
auto one_norm(const M& m) {
    using mag_t = magnitude_t<typename M::value_type>;
    auto result = math::zero<mag_t>();
    for (typename M::size_type c = 0; c < m.num_cols(); ++c) {
        auto col_sum = math::zero<mag_t>();
        for (typename M::size_type r = 0; r < m.num_rows(); ++r) {
            using std::abs;
            col_sum += abs(m(r, c));
        }
        if (col_sum > result) result = col_sum;
    }
    return result;
}

/// infinity_norm(m) = max row sum of |m[i,j]|
template <Matrix M>
auto infinity_norm(const M& m) {
    using mag_t = magnitude_t<typename M::value_type>;
    auto result = math::zero<mag_t>();
    for (typename M::size_type r = 0; r < m.num_rows(); ++r) {
        auto row_sum = math::zero<mag_t>();
        for (typename M::size_type c = 0; c < m.num_cols(); ++c) {
            using std::abs;
            row_sum += abs(m(r, c));
        }
        if (row_sum > result) result = row_sum;
    }
    return result;
}

} // namespace mtl
