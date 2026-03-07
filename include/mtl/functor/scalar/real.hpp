#pragma once
// MTL5 — Scalar real part extraction functor
#include <complex>
#include <type_traits>
#include <mtl/concepts/scalar.hpp>
#include <mtl/concepts/magnitude.hpp>

namespace mtl::functor::scalar {

template <typename T>
struct real {
    using result_type = magnitude_t<T>;

    static constexpr result_type apply(const T& v) {
        if constexpr (is_complex_v<T>) {
            return v.real();
        } else {
            return v;
        }
    }
    constexpr result_type operator()(const T& v) const { return apply(v); }
};

} // namespace mtl::functor::scalar
