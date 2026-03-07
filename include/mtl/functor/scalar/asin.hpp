#pragma once
// MTL5 -- Scalar arc sine functor
#include <cmath>

namespace mtl::functor::scalar {

template <typename T>
struct asin {
    using result_type = T;

    static result_type apply(const T& v) {
        using std::asin;
        return asin(v);
    }
    result_type operator()(const T& v) const { return apply(v); }
};

} // namespace mtl::functor::scalar
