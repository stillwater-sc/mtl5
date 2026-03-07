#pragma once
// MTL5 — Scalar hyperbolic sine functor
#include <cmath>

namespace mtl::functor::scalar {

template <typename T>
struct sinh {
    using result_type = T;

    static result_type apply(const T& v) {
        using std::sinh;
        return sinh(v);
    }
    result_type operator()(const T& v) const { return apply(v); }
};

} // namespace mtl::functor::scalar
