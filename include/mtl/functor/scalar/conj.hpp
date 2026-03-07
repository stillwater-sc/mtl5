#pragma once
// MTL5 -- Scalar conjugation functor
#include <complex>
#include <type_traits>
#include <mtl/concepts/scalar.hpp>

namespace mtl::functor::scalar {

template <typename T>
struct conj {
    using result_type = T;

    static constexpr T apply(const T& v) {
        if constexpr (is_complex_v<T>) {
            return std::conj(v);
        } else {
            return v;  // identity for real types
        }
    }
    constexpr T operator()(const T& v) const { return apply(v); }
};

} // namespace mtl::functor::scalar
