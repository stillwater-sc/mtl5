#pragma once
// MTL5 -- Vector/matrix norms: one_norm, two_norm, infinity_norm, frobenius_norm
//
// two_norm on dense float/double takes MTL5's SIMD loop, not BLAS ?nrm2:
// Level 1 never dispatches to BLAS, even when MTL5_HAS_BLAS is defined (#754).
// It is bandwidth-bound, so a vendor has nothing to win over MTL5's own SIMD
// loop, and OpenBLAS threads ?dot and ?axpy above a few thousand elements with
// a process-wide team: on a machine with busy cores, each call then waits ~3-6
// ms to wake it -- 1,000x the work at n = 16,384, enough to make a BLAS-build
// Krylov solve 30x slower. MTL5's loop threads through MTL5's own pool
// (MTL5_NUM_THREADS) instead. interface::blas:: still has the wrappers for a
// direct call.
#include <algorithm>
#include <cmath>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <limits>
#include <mtl/concepts/vector.hpp>
#include <mtl/detail/wrapping_arithmetic.hpp>
#include <mtl/concepts/matrix.hpp>
#include <mtl/concepts/magnitude.hpp>
#include <mtl/math/identity.hpp>
#include <mtl/math/accumulator_traits.hpp>
#include <mtl/detail/accumulate.hpp>
#include <mtl/detail/thread_pool.hpp>
#include <mtl/interface/dispatch_traits.hpp>
#include <type_traits>
#include <mtl/simd/algorithm.hpp>

namespace mtl {

namespace detail {

/// The default two_norm and frobenius_norm sum the squares unscaled, which
/// overflows for elements above about sqrt(max) and loses everything below
/// about sqrt(min) (#690). Rather than scale every element, as LAPACK's dnrm2
/// does, they keep the plain sum and call this afterwards: when the sum has
/// reached max() (an infinity, or a saturated posit), or is too close to the
/// underflow threshold for the small squares to have been accumulated
/// accurately, it is recomputed in a second pass scaled by a power of two near
/// max|a|. An ordinary input never rescans, so its result is bit-identical to
/// the unscaled sum.
///
/// The rescan needs frexp, ldexp and sqrt, found by ADL so that a number
/// library's own (lns, fixpnt, posit, takum, ...) are used, with the std
/// versions brought in by using-declarations for the built-in types; and a
/// std::numeric_limits specialization for max(), min() and epsilon(). A type
/// without them, and any integer type, keeps the plain sum, so integers keep
/// their wrapping semantics.
namespace norms_adl {
using std::frexp;
using std::ldexp;
using std::sqrt;

template <typename Mag>
concept rescalable_sum_squares =
    std::numeric_limits<Mag>::is_specialized && !std::numeric_limits<Mag>::is_integer &&
    requires(Mag a, int* e, int k) {
        frexp(a, e);
        { ldexp(a, k) } -> std::convertible_to<Mag>;
        { sqrt(a) } -> std::convertible_to<Mag>;
        { a < a } -> std::convertible_to<bool>;
        { a > a } -> std::convertible_to<bool>;
    };

/// sqrt(sum a^2) over the magnitudes that `for_each_abs(f)` passes to `f`,
/// computed with a power-of-two scale so that neither the squares nor their
/// sum overflow or underflow. In a binary floating-point type the scaling is
/// exact; in a tapered type it also keeps the squares near 1, where the type
/// is most precise.
template <typename Mag, typename ForEachAbs>
Mag rescaled_two_norm(ForEachAbs&& for_each_abs) {
    Mag amax(0);
    bool has_nan = false;
    Mag nan_value(0);
    for_each_abs([&](const Mag& a) {
        if constexpr (requires { { a != a } -> std::convertible_to<bool>; }) {
            if (a != a) { if (!has_nan) nan_value = a; has_nan = true; return; }
        }
        if (a > amax) amax = a;
    });
    if (has_nan) return nan_value;
    // All zero, or an infinity: the norm is amax itself.
    if (!(amax > Mag(0)) || amax > std::numeric_limits<Mag>::max()) return amax;
    int e = 0;
    frexp(amax, &e);                   // amax = f * 2^e, f in [0.5, 1)
    Mag ss(0);
    for_each_abs([&](const Mag& a) {
        const Mag s = ldexp(a, -e);    // in [0, 1): s*s cannot overflow
        ss = ss + s * s;
    });
    return ldexp(Mag(sqrt(ss)), e);
}
} // namespace norms_adl

template <typename Mag>
inline constexpr bool rescalable_sum_squares_v = norms_adl::rescalable_sum_squares<Mag>;

using norms_adl::rescaled_two_norm;

template <typename Mag>
bool sum_squares_needs_rescan(const Mag& ss) {
    // !(ss < max()) catches an infinity and a type that saturates at max()
    // instead; it is also true for NaN, which the rescan returns unchanged.
    return !(ss < std::numeric_limits<Mag>::max())
        || ss < std::numeric_limits<Mag>::min() / std::numeric_limits<Mag>::epsilon();
}

/// result = max(result, a), but a NaN `a` makes the result NaN and keeps it
/// there (#690). A plain `a > result` skips a NaN, so infinity_norm({1, NaN, 2})
/// gave 2 while one_norm gave NaN. For an integer `a != a` is always false. A
/// magnitude type without `!=` keeps the plain maximum.
template <typename Mag>
void max_propagating_nan(Mag& result, const Mag& a) {
    if constexpr (requires { { a != a } -> std::convertible_to<bool>; }) {
        if (result != result) return;          // already NaN: keep it
        if (a != a || a > result) result = a;
    } else {
        if (a > result) result = a;
    }
}

} // namespace detail

// -- Vector norms --------------------------------------------------------

/// one_norm(v) = sum(|v[i]|)
template <Vector V>
auto one_norm(const V& v) {
    using mag_t = magnitude_t<typename V::value_type>;
    auto acc = math::zero<mag_t>();
    // Wrapping accumulate, and a wrapping abs (#461): for an integral element
    // type `acc += abs(v(i))` is UB twice over -- at the sum, and at abs() of the
    // type's minimum, which has no positive counterpart.
    for (typename V::size_type i = 0; i < v.size(); ++i) {
        acc = detail::generic_add<mag_t>(acc, detail::generic_abs(v(i)));
    }
    return acc;
}

/// two_norm(v) = sqrt(sum(|v[i]|^2)).
///
/// Mixed precision: pass an explicit `Accumulator` to sum the squares in a wider
/// precision than the element magnitude (e.g. `two_norm<double>(v)` over a
/// bfloat16/float vector), guarding the long reduction against overflow and
/// precision loss. Default `Accumulator = void` keeps the SIMD / loop dispatch
/// unchanged.
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
        "default path dispatches to SIMD and has no accumulator to round out. "
        "Pass an Accumulator too, e.g. two_norm<double, double>(v) (#379).");
    if constexpr (!std::is_void_v<Accumulator>) {
        using out_t = std::conditional_t<std::is_void_v<Result>, mag_t, Result>;
        using AT = math::accumulator_traits<Accumulator, mag_t>;
        // Split across the thread pool when v is plain dense storage and the
        // accumulator can merge (#582); serial otherwise, and at one thread
        // exactly the loop it always was.
        const Accumulator acc = detail::accumulate<Accumulator, mag_t,
                                                   interface::ThreadableDenseVector<V>>(
            v.size(), detail::accumulate_chunk_units<Accumulator, mag_t>,
            [&](Accumulator& part, std::size_t lo, std::size_t hi) {
                for (std::size_t i = lo; i < hi; ++i) {
                    mag_t a = detail::generic_abs(v(i));
                    AT::add_product(part, a, a);       // part += a*a in Accumulator
                }
            });
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
    // Native SIMD path for contiguous real float/double (abs is the identity).
    if constexpr (interface::BlasDenseVector<V>) {
        using std::sqrt;
        // Parallel reduction of the sum of squares (deterministic per thread
        // count, not serial-bit-identical). Serial by default.
        using T = typename V::value_type;
        const T* d = v.data();
        const T ss = ::mtl::parallel::pool().parallel_reduce<T>(
            v.size(), /*grain=*/std::size_t{65536},
            [&](std::size_t lo, std::size_t hi) { return simd::reduce_sum_squares<T>(d + lo, hi - lo); });
        if (detail::sum_squares_needs_rescan(ss)) {   // #690
            return detail::rescaled_two_norm<T>([&](auto&& f) {
                for (std::size_t i = 0; i < v.size(); ++i) f(T(std::abs(d[i])));
            });
        }
        return sqrt(ss);
    } else {
        auto acc = math::zero<mag_t>();
        for (typename V::size_type i = 0; i < v.size(); ++i) {
            auto a = detail::generic_abs(v(i));
            acc = detail::generic_fma<mag_t>(acc, a, a);
        }
        if constexpr (detail::rescalable_sum_squares_v<mag_t>) {
            if (detail::sum_squares_needs_rescan(acc)) {   // #690
                return detail::rescaled_two_norm<mag_t>([&](auto&& f) {
                    for (typename V::size_type i = 0; i < v.size(); ++i)
                        f(static_cast<mag_t>(detail::generic_abs(v(i))));
                });
            }
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
        // Wrapping abs (#461): std::abs is UB at the minimum of a signed type.
        // NOTE what that means for an integer infinity_norm -- the wrapped |min|
        // is negative, so it never wins the max. An integer norm is DEFINED, and
        // reduced mod 2^N, but it is only a norm when nothing overflowed.
        auto a = detail::generic_abs(v(i));
        detail::max_propagating_nan(result, a);   // #690
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
        // Split by rows across the thread pool when m is plain dense storage and
        // the accumulator can merge (#582); serial otherwise. Each chunk walks
        // its rows in the serial order, so one thread gives the original loop.
        const std::size_t ncols = m.num_cols();
        const std::size_t row_grain = ncols
            ? std::max(std::size_t{1}, detail::accumulate_chunk_units<Accumulator, mag_t> / ncols)
            : std::size_t{1};
        const Accumulator acc = detail::accumulate<Accumulator, mag_t,
                                                   interface::ThreadableDenseMatrix<M>>(
            m.num_rows(), row_grain,
            [&](Accumulator& part, std::size_t lo, std::size_t hi) {
                for (std::size_t r = lo; r < hi; ++r) {
                    for (std::size_t c = 0; c < ncols; ++c) {
                        mag_t a = detail::generic_abs(m(r, c));
                        AT::add_product(part, a, a);
                    }
                }
            });
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
                auto a = detail::generic_abs(m(r, c));
                acc = detail::generic_fma<mag_t>(acc, a, a);
            }
        }
        if constexpr (detail::rescalable_sum_squares_v<mag_t>) {
            if (detail::sum_squares_needs_rescan(acc)) {   // #690
                return detail::rescaled_two_norm<mag_t>([&](auto&& f) {
                    for (typename M::size_type r = 0; r < m.num_rows(); ++r)
                        for (typename M::size_type c = 0; c < m.num_cols(); ++c)
                            f(static_cast<mag_t>(detail::generic_abs(m(r, c))));
                });
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
            col_sum = detail::generic_add<mag_t>(col_sum, detail::generic_abs(m(r, c)));
        }
        detail::max_propagating_nan(result, col_sum);   // #690
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
            row_sum = detail::generic_add<mag_t>(row_sum, detail::generic_abs(m(r, c)));
        }
        detail::max_propagating_nan(result, row_sum);   // #690
    }
    return result;
}

} // namespace mtl
