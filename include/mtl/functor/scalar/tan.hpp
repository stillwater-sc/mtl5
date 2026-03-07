#pragma once
// MTL5 — Scalar tangent functor
#include <cmath>

namespace mtl::functor::scalar {

template <typename T>
struct tan {
    using result_type = T;

    static result_type apply(const T& v) {
        using std::tan;
        return tan(v);
    }
    result_type operator()(const T& v) const { return apply(v); }
};

} // namespace mtl::functor::scalar
