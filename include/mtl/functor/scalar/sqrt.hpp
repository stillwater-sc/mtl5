#pragma once
// MTL5 — Scalar square root functor
#include <cmath>

namespace mtl::functor::scalar {

template <typename T>
struct sqrt {
    using result_type = T;

    static result_type apply(const T& v) {
        using std::sqrt;
        return sqrt(v);
    }
    result_type operator()(const T& v) const { return apply(v); }
};

} // namespace mtl::functor::scalar
