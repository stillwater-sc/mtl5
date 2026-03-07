#pragma once
// MTL5 — Scalar hyperbolic cosine functor
#include <cmath>

namespace mtl::functor::scalar {

template <typename T>
struct cosh {
    using result_type = T;

    static result_type apply(const T& v) {
        using std::cosh;
        return cosh(v);
    }
    result_type operator()(const T& v) const { return apply(v); }
};

} // namespace mtl::functor::scalar
