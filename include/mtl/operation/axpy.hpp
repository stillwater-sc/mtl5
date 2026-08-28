#pragma once
// MTL5 -- axpy: y <- alpha*x + y  (BLAS level-1)
// Native SIMD path for contiguous, same-type real float/double vectors
// (mtl::simd::axpy, #86); BLAS dispatch when available; generic scalar
// fallback otherwise.
#include <cassert>
#include <cstddef>
#include <limits>
#include <type_traits>

#include <mtl/concepts/scalar.hpp>
#include <mtl/math/accumulator_traits.hpp>
#include <mtl/detail/wrapping_arithmetic.hpp>
#include <mtl/concepts/vector.hpp>
#include <mtl/detail/thread_pool.hpp>
#include <mtl/interface/dispatch_traits.hpp>
#include <mtl/simd/algorithm.hpp>
#ifdef MTL5_HAS_BLAS
#include <mtl/interface/blas.hpp>
#endif

namespace mtl {

/// y[i] += alpha * x[i]
///
/// Mixed precision: pass an explicit `Accumulator` to form each element's update
/// in a precision distinct from the operand type; the result is rounded out to
/// y's element type on store (#261, Part C). The default `Accumulator = void`
/// keeps today's BLAS / SIMD / generic dispatch byte for byte.
///
/// WHY AN AXPY HAS AN ACCUMULATOR AT ALL, when it sums nothing. Each element is
/// `y(i) + alpha * x(i)` -- a one-term sum of products, seeded with y(i) rather
/// than with zero. That is exactly the shape `accumulator_traits` describes, and
/// the configurations do what they do everywhere else:
///
///   config 1  the product is formed in `Acc` and rounded, then added, and the
///             sum is rounded out to y's type: an `Acc` wider than the element
///             type buys precision, a narrower one deliberately spends it.
///   config 2  `fma_accumulator<T>` fuses the product into the sum -- one
///             rounding per element rather than two.
///   config 3  a quire holds `y(i)` and the product exactly; the single rounding
///             is the store.
///
/// THE DEFAULT'S ROUNDING IS DISPATCH-DEPENDENT, and that is the strongest
/// argument for the policy. `Accumulator = void` may reach any of four paths,
/// which do not agree with each other:
///
///   external BLAS      whatever the vendor's ?axpy does -- unspecified
///   simd::axpy body    `fma(alpha, x, y)`, fused, one rounding
///   simd::axpy tail    `wrap_add(y, wrap_mul(alpha, x))`, NOT fused, two
///   generic loop       `y + alpha*x`, not fused, two
///
/// So a vector shorter than one SIMD lane group is rounded differently from a
/// long one, and a BLAS build differently again. An explicit `Accumulator` is
/// how a caller PINS the rounding across all of them -- `axpy<fma_accumulator<T>>`
/// is fused on every path, by construction.
///
/// This was originally documented here as "the default is already fused", from a
/// build whose `batch<float>::size` is 1 so the tail never runs and no BLAS is
/// linked. The Highway and LAPACK CI lanes disagreed. The body/tail split
/// predates this change and is not touched here; it is recorded on #261.
///
/// This is the first caller to exercise the contract's `assign`. The reduction
/// kernels all start from zero and use `clear`; an axpy starts from y(i), which
/// is what `assign` is for, so a quire specialization that stubbed it out would
/// be found here rather than in a peer-repo integration.
template <typename Accumulator = void, Scalar S, Vector VX, Vector VY>
void axpy(const S& alpha, const VX& x, VY& y) {
    assert(x.size() == y.size());

    if constexpr (!interface::accumulator_allows_blas_v<Accumulator>) {
        // A custom accumulator cannot be honored by external BLAS or by the SIMD
        // kernel, both of which accumulate in a hardware-fixed precision -- the
        // same rule `mult` and `dot` follow via accumulator_allows_blas_v.
        using Result = typename VY::value_type;
        // `S` IS PART OF THE OPERAND TYPE. Casting alpha to the element common
        // type first would round a wider scalar away before the accumulator saw
        // it -- which is exactly the mixed-precision case the policy exists for
        // (narrow elements, a wider scalar). Caught in review of #511.
        using Value  = std::common_type_t<S, typename VX::value_type,
                                             typename VY::value_type>;
        using AT = math::accumulator_traits<Accumulator, Value>;
        const Value a = static_cast<Value>(alpha);
        for (typename VY::size_type i = 0; i < y.size(); ++i) {
            Accumulator acc{};
            AT::assign(acc, static_cast<Value>(y(i)));   // seed, not clear
            AT::add_product(acc, a, static_cast<Value>(x(i)));
            y(i) = AT::template value<Result>(acc);      // fused convert on store
        }
        return;
    }

    if constexpr (interface::BlasDenseVector<VX> && interface::BlasDenseVector<VY> &&
                  std::is_same_v<typename VX::value_type, typename VY::value_type>) {
        using T = typename VY::value_type;
        const std::size_t n = y.size();
#ifdef MTL5_HAS_BLAS
        if (n <= static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            interface::blas::axpy(static_cast<int>(n), static_cast<T>(alpha),
                                  x.data(), 1, y.data(), 1);
            return;
        }
#endif
        // Element-wise: chunk the range across the pool (bit-identical, each
        // element in exactly one chunk). Serial by default (MTL5_NUM_THREADS=1).
        const T a = static_cast<T>(alpha);
        const T* xp = x.data();
        T* yp = y.data();
        detail::thread_pool::instance().parallel_for(n, /*grain=*/std::size_t{65536},
            [&](std::size_t b, std::size_t e) { simd::axpy<T>(a, xp + b, yp + b, e - b); });
    } else {
        // Wrapping when the element type is integral, the plain expression
        // otherwise (#461): a signed `y(i) += alpha * x(i)` is UB exactly where
        // the SIMD path it completes wraps and defines the answer.
        using T = typename VY::value_type;
        for (typename VY::size_type i = 0; i < y.size(); ++i) {
            y(i) = detail::generic_fma<T>(y(i), alpha, x(i));
        }
    }
}

} // namespace mtl
