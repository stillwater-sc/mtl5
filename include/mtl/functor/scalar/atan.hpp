#pragma once
// MTL5 -- Scalar arc tangent functor
#include <cmath>

namespace mtl::functor::scalar {

template <typename T>
struct atan {
    using result_type = T;

    static result_type apply(const T& v) {
        using std::atan;
        return atan(v);
    }
    result_type operator()(const T& v) const { return apply(v); }
};

} // namespace mtl::functor::scalar
