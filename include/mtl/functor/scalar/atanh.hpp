#pragma once
// MTL5 — Scalar inverse hyperbolic tangent functor
#include <cmath>

namespace mtl::functor::scalar {

template <typename T>
struct atanh {
    using result_type = T;

    static result_type apply(const T& v) {
        using std::atanh;
        return atanh(v);
    }
    result_type operator()(const T& v) const { return apply(v); }
};

} // namespace mtl::functor::scalar
