#pragma once
// MTL5 — Scalar imaginary part extraction functor
#include <complex>
#include <type_traits>
#include <mtl/concepts/scalar.hpp>
#include <mtl/concepts/magnitude.hpp>

namespace mtl::functor::scalar {

template <typename T>
struct imag {
    using result_type = magnitude_t<T>;

    static constexpr result_type apply(const T& v) {
        if constexpr (is_complex_v<T>) {
            return v.imag();
        } else {
            return result_type{0};
        }
    }
    constexpr result_type operator()(const T& v) const { return apply(v); }
};

} // namespace mtl::functor::scalar
