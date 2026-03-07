#pragma once
// MTL5 -- Scalar cosine functor
#include <cmath>

namespace mtl::functor::scalar {

template <typename T>
struct cos {
    using result_type = T;

    static result_type apply(const T& v) {
        using std::cos;
        return cos(v);
    }
    result_type operator()(const T& v) const { return apply(v); }
};

} // namespace mtl::functor::scalar
