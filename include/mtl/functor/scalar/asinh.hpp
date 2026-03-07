#pragma once
// MTL5 — Scalar inverse hyperbolic sine functor
#include <cmath>

namespace mtl::functor::scalar {

template <typename T>
struct asinh {
    using result_type = T;

    static result_type apply(const T& v) {
        using std::asinh;
        return asinh(v);
    }
    result_type operator()(const T& v) const { return apply(v); }
};

} // namespace mtl::functor::scalar
