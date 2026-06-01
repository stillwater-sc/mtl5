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
#include <mtl/functor/scalar/conj.hpp>
#include <mtl/interface/dispatch_traits.hpp>
#ifdef MTL5_HAS_BLAS
#include <mtl/interface/blas.hpp>
#endif

namespace mtl {

/// Hermitian dot product: sum(conj(v1[i]) * v2[i])
template <Vector V1, Vector V2>
auto dot(const V1& v1, const V2& v2) {
    assert(v1.size() == v2.size());
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
    using result_type = std::common_type_t<typename V1::value_type, typename V2::value_type>;
    auto acc = math::zero<result_type>();
    for (typename V1::size_type i = 0; i < v1.size(); ++i) {
        acc += functor::scalar::conj<typename V1::value_type>::apply(v1(i)) * v2(i);
    }
    return acc;
}

/// Real dot product: sum(v1[i] * v2[i]) -- no conjugation
template <Vector V1, Vector V2>
auto dot_real(const V1& v1, const V2& v2) {
    assert(v1.size() == v2.size());
#ifdef MTL5_HAS_BLAS
    if constexpr (interface::BlasDenseVector<V1> && interface::BlasDenseVector<V2>) {
        if (v1.size() <= static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            return interface::blas::dot(static_cast<int>(v1.size()),
                                        v1.data(), 1, v2.data(), 1);
        }
    }
#endif
    using result_type = std::common_type_t<typename V1::value_type, typename V2::value_type>;
    auto acc = math::zero<result_type>();
    for (typename V1::size_type i = 0; i < v1.size(); ++i) {
        acc += v1(i) * v2(i);
    }
    return acc;
}

} // namespace mtl
