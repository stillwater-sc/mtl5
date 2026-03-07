#pragma once
// MTL5 — Scalar cube root functor
#include <cmath>

namespace mtl::functor::scalar {

template <typename T>
struct cbrt {
    using result_type = T;

    static result_type apply(const T& v) {
        using std::cbrt;
        return cbrt(v);
    }
    result_type operator()(const T& v) const { return apply(v); }
};

} // namespace mtl::functor::scalar
