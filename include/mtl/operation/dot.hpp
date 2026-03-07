#pragma once
// MTL5 -- Dot product (inner product) of two vectors
#include <cassert>
#include <complex>
#include <type_traits>
#include <mtl/concepts/vector.hpp>
#include <mtl/concepts/scalar.hpp>
#include <mtl/math/identity.hpp>
#include <mtl/functor/scalar/conj.hpp>

namespace mtl {

/// Hermitian dot product: sum(conj(v1[i]) * v2[i])
template <Vector V1, Vector V2>
auto dot(const V1& v1, const V2& v2) {
    assert(v1.size() == v2.size());
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
    using result_type = std::common_type_t<typename V1::value_type, typename V2::value_type>;
    auto acc = math::zero<result_type>();
    for (typename V1::size_type i = 0; i < v1.size(); ++i) {
        acc += v1(i) * v2(i);
    }
    return acc;
}

} // namespace mtl
