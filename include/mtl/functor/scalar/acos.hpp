#pragma once
// MTL5 — Scalar arc cosine functor
#include <cmath>

namespace mtl::functor::scalar {

template <typename T>
struct acos {
    using result_type = T;

    static result_type apply(const T& v) {
        using std::acos;
        return acos(v);
    }
    result_type operator()(const T& v) const { return apply(v); }
};

} // namespace mtl::functor::scalar
