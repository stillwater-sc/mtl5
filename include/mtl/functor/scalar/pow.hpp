#pragma once
// MTL5 — Scalar power functor (binary: base^exponent)
#include <cmath>

namespace mtl::functor::scalar {

template <typename T>
struct pow {
    using result_type = T;

    template <typename S>
    static result_type apply(const T& base, const S& exponent) {
        using std::pow;
        return pow(base, exponent);
    }
    template <typename S>
    result_type operator()(const T& base, const S& exponent) const { return apply(base, exponent); }
};

} // namespace mtl::functor::scalar
