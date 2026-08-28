#pragma once
// MTL5 -- Scale collection by a scalar factor
#include <cstddef>
#include <limits>
#include <type_traits>

#include <mtl/concepts/scalar.hpp>
#include <mtl/math/accumulator_traits.hpp>
#include <mtl/detail/wrapping_arithmetic.hpp>
#include <mtl/concepts/collection.hpp>
#include <mtl/detail/thread_pool.hpp>
#include <mtl/interface/dispatch_traits.hpp>
#include <mtl/simd/algorithm.hpp>
#ifdef MTL5_HAS_BLAS
#include <mtl/interface/blas.hpp>
#endif

namespace mtl {

/// In-place scale: c[i] *= alpha
///
/// Mixed precision: pass an explicit `Accumulator` to form each product in a
/// precision distinct from the element type; the result is rounded out to the
/// element type on store (#261, Part C). Default `Accumulator = void` keeps
/// today's BLAS / SIMD / generic dispatch byte for byte.
///
/// WHAT THIS BUYS, STATED PRECISELY, because it is less than `axpy`'s and it
/// would be easy to imply otherwise. A scale sums nothing: each element is a
/// SINGLE product, seeded from zero. So the configurations do not separate the
/// way they do in a reduction --
///
///   * config 2 buys NOTHING here. `fma(m, v, 0)` is `m * v` rounded once, and
///     so is a plain multiply. There is no product rounding to fuse away.
///   * config 3 buys NOTHING here either, for the same reason: an IEEE multiply
///     is ALREADY correctly rounded, so a quire has no rounding to remove.
///
/// What a non-default accumulator does buy is WIDENING: the product is formed in
/// `Acc` and rounded once to the element type, where the default forms it in the
/// element type. Scaling a bfloat16 vector by a float with `Acc = float` is the
/// case that matters, and it is a real difference -- but it is a precision
/// difference, not an exactness one.
///
/// It is threaded anyway because Part C's goal is that every BLAS routine be
/// specializable by the same policy, and a caller writing `scale<Acc>` alongside
/// `axpy<Acc>` and `dot<Acc>` should not have to know which of them the policy
/// can actually help. The honest answer to "what does a quire do for scale" is
/// "nothing", and it belongs in the code rather than in the reader's surprise.
template <typename Accumulator = void, Scalar S, MutableCollection C>
void scale(const S& alpha, C& c) {
    if constexpr (!interface::accumulator_allows_blas_v<Accumulator>) {
        using Result = typename C::value_type;
        // `S` is part of the operand type -- see axpy for why. Scaling a
        // bfloat16 vector by a float is the case named above, and casting alpha
        // to the element type first would have thrown the float away, making
        // that paragraph false. Caught in review of #511.
        using Value  = std::common_type_t<S, typename C::value_type>;
        using AT = math::accumulator_traits<Accumulator, Value>;
        const Value a = static_cast<Value>(alpha);
        for (auto it = c.begin(); it != c.end(); ++it) {
            Accumulator acc{};
            AT::clear(acc);                              // zero seed: a bare product
            AT::add_product(acc, static_cast<Value>(*it), a);
            *it = AT::template value<Result>(acc);
        }
        return;
    }

    // Native SIMD / BLAS path for contiguous real float/double vectors;
    // generic iterator loop for everything else (matrices, strided, complex).
    if constexpr (interface::BlasDenseVector<C>) {
        using T = typename C::value_type;
        const std::size_t n = c.size();
#ifdef MTL5_HAS_BLAS
        if (n <= static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            interface::blas::scal(static_cast<int>(n), static_cast<T>(alpha), c.data(), 1);
            return;
        }
#endif
        // Element-wise: chunk the range across the pool (bit-identical).
        const T a = static_cast<T>(alpha);
        T* cp = c.data();
        detail::thread_pool::instance().parallel_for(n, /*grain=*/std::size_t{65536},
            [&](std::size_t b, std::size_t e) { simd::scal<T>(a, cp + b, e - b); });
    } else {
        // Wrapping when the element type is integral (#461). `*it *= alpha` is
        // `*it = static_cast<T>(*it * alpha)` by definition, so `generic_mul`
        // returning T preserves the rounding rather than adding one.
        using T = typename C::value_type;
        for (auto it = c.begin(); it != c.end(); ++it) {
            *it = detail::generic_mul<T>(*it, alpha);
        }
    }
}

/// Returns a scaled copy of a vector. Forwards the accumulator policy, so
/// `scaled<Acc>` and `scale<Acc>` agree.
template <typename Accumulator = void, Scalar S, Collection C>
auto scaled(const S& alpha, const C& c) {
    auto result = c;  // copy
    scale<Accumulator>(alpha, result);
    return result;
}

} // namespace mtl
