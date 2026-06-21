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
#include <mtl/interface/dispatch_traits.hpp>
#include <mtl/simd/algorithm.hpp>
#ifdef MTL5_HAS_BLAS
#include <mtl/interface/blas.hpp>
#endif

namespace mtl {

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
        using Value = std::common_type_t<typename V1::value_type, typename V2::value_type>;
        using AT = math::accumulator_traits<Accumulator, Value>;
        Accumulator acc{};
        AT::clear(acc);
        for (typename V1::size_type i = 0; i < v1.size(); ++i)
            AT::add_product(acc,
                static_cast<Value>(functor::scalar::conj<typename V1::value_type>::apply(v1(i))),
                static_cast<Value>(v2(i)));
        return AT::template value<Result>(acc);
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
        return simd::reduce_dot<typename V1::value_type>(v1.data(), v2.data(), v1.size());
    } else {
        using result_type = std::common_type_t<typename V1::value_type, typename V2::value_type>;
        auto acc = math::zero<result_type>();
        for (typename V1::size_type i = 0; i < v1.size(); ++i) {
            acc += functor::scalar::conj<typename V1::value_type>::apply(v1(i)) * v2(i);
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
        using Value = std::common_type_t<typename V1::value_type, typename V2::value_type>;
        using AT = math::accumulator_traits<Accumulator, Value>;
        Accumulator acc{};
        AT::clear(acc);
        for (typename V1::size_type i = 0; i < v1.size(); ++i)
            AT::add_product(acc, static_cast<Value>(v1(i)), static_cast<Value>(v2(i)));
        return AT::template value<Result>(acc);
    } else {
#ifdef MTL5_HAS_BLAS
    if constexpr (interface::BlasDenseVector<V1> && interface::BlasDenseVector<V2>) {
        if (v1.size() <= static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            return interface::blas::dot(static_cast<int>(v1.size()),
                                        v1.data(), 1, v2.data(), 1);
        }
    }
#endif
    if constexpr (interface::BlasDenseVector<V1> && interface::BlasDenseVector<V2> &&
                  std::is_same_v<typename V1::value_type, typename V2::value_type>) {
        return simd::reduce_dot<typename V1::value_type>(v1.data(), v2.data(), v1.size());
    } else {
        using result_type = std::common_type_t<typename V1::value_type, typename V2::value_type>;
        auto acc = math::zero<result_type>();
        for (typename V1::size_type i = 0; i < v1.size(); ++i) {
            acc += v1(i) * v2(i);
        }
        return acc;
    }
    }
}

} // namespace mtl
