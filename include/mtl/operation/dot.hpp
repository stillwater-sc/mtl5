#pragma once
// MTL5 -- Dot product (inner product) of two vectors
// Optional BLAS dispatch when MTL5_HAS_BLAS is defined and types qualify
// (real float/double dense vectors), mirroring two_norm.
#include <cassert>
#include <complex>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <mtl/concepts/vector.hpp>
#include <mtl/concepts/scalar.hpp>
#include <mtl/math/identity.hpp>
#include <mtl/math/accumulator_traits.hpp>
#include <mtl/functor/scalar/conj.hpp>
#include <mtl/detail/thread_pool.hpp>
#include <mtl/detail/wrapping_arithmetic.hpp>
#include <mtl/interface/dispatch_traits.hpp>
#include <mtl/simd/algorithm.hpp>
#ifdef MTL5_HAS_BLAS
#include <mtl/interface/blas.hpp>
#endif

namespace mtl {

/// Does `dot<Accumulator, Result>(v1, v2)` name the widening integer kernel?
///
/// Both vectors contiguous over the SAME 16-bit type, and the requested
/// accumulator exactly that type's widened counterpart (int16 -> int32,
/// uint16 -> uint32) with no separate delivery type. Anything else -- a wider
/// accumulator, a mixed pair, a strided view -- falls to the generic
/// accumulator_traits loop, which is correct for all of them.
template <typename Accumulator, typename Result, typename V1, typename V2>
concept widening_int_dot =
    interface::SimdNarrowVector<V1> && interface::SimdNarrowVector<V2> &&
    std::is_same_v<typename V1::value_type, typename V2::value_type> &&
    std::is_same_v<Accumulator, simd::widen_accumulator_t<typename V1::value_type>> &&
    std::is_same_v<Result, Accumulator>;

/// Hermitian dot product: sum(conj(v1[i]) * v2[i]).
///
/// Mixed precision: pass an explicit `Accumulator` to sum the products in a
/// precision distinct from the element type (e.g. `dot<float>(a, b)` over
/// bfloat16 vectors accumulates in fp32), and an optional `Result` to round the
/// sum out to a delivery type (default = the accumulator type). With the default
/// `Accumulator = void`, behavior is unchanged (BLAS/SIMD fast path or the
/// common-type loop). The mixed path is scalar; a SIMD variant is a follow-up.
template <typename Accumulator = void, typename Result = Accumulator,
          Vector V1, Vector V2>
auto dot(const V1& v1, const V2& v2) {
    assert(v1.size() == v2.size());
    if constexpr (!interface::accumulator_allows_blas_v<Accumulator>) {
        // SIMD widening fast path: real float elements, fp64 accumulate/result.
        // (float is real, so conj is the identity and reduce_dot_widen matches
        // the Hermitian product.) The default accumulator_traits adds the same
        // promoted products, so the result matches the scalar policy path.
        if constexpr (interface::BlasDenseVector<V1> && interface::BlasDenseVector<V2> &&
                      std::is_same_v<typename V1::value_type, float> &&
                      std::is_same_v<typename V2::value_type, float> &&
                      std::is_same_v<Accumulator, double> && std::is_same_v<Result, double>) {
            return simd::reduce_dot_widen<double, float>(v1.data(), v2.data(), v1.size());
        } else if constexpr (widening_int_dot<Accumulator, Result, V1, V2>) {
            // Widening INTEGER fast path (#451 phase 2): 16-bit operands into a
            // 32-bit accumulator, via the hardware's widening multiply-add. conj
            // is the identity on the integers, so this is the Hermitian product.
            // The sum wraps mod 2^32 and does so within a few terms at full
            // range -- see reduce_dot_widen for the contract.
            return simd::reduce_dot_widen<Accumulator, typename V1::value_type>(
                v1.data(), v2.data(), v1.size());
        } else {
            using Value = std::common_type_t<typename V1::value_type, typename V2::value_type>;
            using AT = math::accumulator_traits<Accumulator, Value>;
            Accumulator acc{};
            AT::clear(acc);
            for (typename V1::size_type i = 0; i < v1.size(); ++i)
                AT::add_product(acc,
                    static_cast<Value>(functor::scalar::conj<typename V1::value_type>::apply(v1(i))),
                    static_cast<Value>(v2(i)));
            return AT::template value<Result>(acc);
        }
    } else {
#ifdef MTL5_HAS_BLAS
    // BlasDenseVector is real float/double, where conj is the identity, so
    // BLAS ?dot matches the Hermitian product on these types. Guard the int
    // length cast: BLAS takes int, so fall back to the loop for huge vectors.
    if constexpr (interface::BlasDenseVector<V1> && interface::BlasDenseVector<V2>) {
        if (v1.size() <= static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            return interface::blas::dot(static_cast<int>(v1.size()),
                                        v1.data(), 1, v2.data(), 1);
        }
    }
#endif
    // Native SIMD path for contiguous, same-type real float/double vectors
    // (conj is the identity there, so reduce_dot matches the Hermitian product).
    if constexpr (interface::BlasDenseVector<V1> && interface::BlasDenseVector<V2> &&
                  std::is_same_v<typename V1::value_type, typename V2::value_type>) {
        // Parallel reduction over chunks (deterministic per thread count, but not
        // serial-bit-identical -- summation grouping differs). Serial by default.
        using T = typename V1::value_type;
        const T* a = v1.data();
        const T* b = v2.data();
        return detail::thread_pool::instance().parallel_reduce<T>(
            v1.size(), /*grain=*/std::size_t{65536},
            [&](std::size_t lo, std::size_t hi) { return simd::reduce_dot<T>(a + lo, b + lo, hi - lo); });
    } else if constexpr (interface::SimdDenseVector<V1> && interface::SimdDenseVector<V2> &&
                         std::is_same_v<typename V1::value_type, typename V2::value_type>) {
        // Integer lanes (#451 phase 0). conj is the identity on the integers, so
        // reduce_dot is the Hermitian product here too. The sum is exact mod
        // 2^32 -- see simd/batch.hpp for the contract.
        //
        // SERIAL, unlike the float branch above. parallel_reduce combines its
        // partials with `acc = acc + partials[t]`, and on int32 that plain `+`
        // is signed-overflow UB. Wrapping addition IS associative, so a parallel
        // integer reduction would be bit-identical and is worth having; making
        // the combine wrap-safe means touching parallel_reduce's documented
        // plus-only contract for every T, which is not a phase-0 change.
        using T = typename V1::value_type;
        return simd::reduce_dot<T>(v1.data(), v2.data(), v1.size());
    } else {
        using result_type = std::common_type_t<typename V1::value_type, typename V2::value_type>;
        auto acc = math::zero<result_type>();
        for (typename V1::size_type i = 0; i < v1.size(); ++i) {
            // generic_fma wraps on integral result types, so a non-contiguous
            // integer vector (which the concepts route here) gets the same
            // mod-2^32 answer as the SIMD path rather than signed-overflow UB.
            acc = detail::generic_fma<result_type>(
                acc, functor::scalar::conj<typename V1::value_type>::apply(v1(i)), v2(i));
        }
        return acc;
    }
    }
}

/// Real dot product: sum(v1[i] * v2[i]) -- no conjugation.
///
/// Mixed precision: see `dot` -- pass `Accumulator` (and optional `Result`) to
/// accumulate in a precision distinct from the element type.
template <typename Accumulator = void, typename Result = Accumulator,
          Vector V1, Vector V2>
auto dot_real(const V1& v1, const V2& v2) {
    assert(v1.size() == v2.size());
    if constexpr (!interface::accumulator_allows_blas_v<Accumulator>) {
        // SIMD widening fast path: real float elements, fp64 accumulate/result.
        if constexpr (interface::BlasDenseVector<V1> && interface::BlasDenseVector<V2> &&
                      std::is_same_v<typename V1::value_type, float> &&
                      std::is_same_v<typename V2::value_type, float> &&
                      std::is_same_v<Accumulator, double> && std::is_same_v<Result, double>) {
            return simd::reduce_dot_widen<double, float>(v1.data(), v2.data(), v1.size());
        } else if constexpr (widening_int_dot<Accumulator, Result, V1, V2>) {
            return simd::reduce_dot_widen<Accumulator, typename V1::value_type>(
                v1.data(), v2.data(), v1.size());
        } else {
            using Value = std::common_type_t<typename V1::value_type, typename V2::value_type>;
            using AT = math::accumulator_traits<Accumulator, Value>;
            Accumulator acc{};
            AT::clear(acc);
            for (typename V1::size_type i = 0; i < v1.size(); ++i)
                AT::add_product(acc, static_cast<Value>(v1(i)), static_cast<Value>(v2(i)));
            return AT::template value<Result>(acc);
        }
    } else {
#ifdef MTL5_HAS_BLAS
    if constexpr (interface::BlasDenseVector<V1> && interface::BlasDenseVector<V2>) {
        if (v1.size() <= static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            return interface::blas::dot(static_cast<int>(v1.size()),
                                        v1.data(), 1, v2.data(), 1);
        }
    }
#endif
    // SimdDenseVector is BlasDenseVector widened to every mtl::simd lane type,
    // so this also picks up the integer lanes (#451 phase 0); on those the sum
    // is exact mod 2^32 rather than order-dependent.
    if constexpr (interface::SimdDenseVector<V1> && interface::SimdDenseVector<V2> &&
                  std::is_same_v<typename V1::value_type, typename V2::value_type>) {
        return simd::reduce_dot<typename V1::value_type>(v1.data(), v2.data(), v1.size());
    } else {
        using result_type = std::common_type_t<typename V1::value_type, typename V2::value_type>;
        auto acc = math::zero<result_type>();
        for (typename V1::size_type i = 0; i < v1.size(); ++i) {
            acc = detail::generic_fma<result_type>(acc, v1(i), v2(i));   // wraps on integers
        }
        return acc;
    }
    }
}

} // namespace mtl
