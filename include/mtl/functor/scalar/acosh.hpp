#pragma once
// MTL5 -- Scalar inverse hyperbolic cosine functor
#include <cmath>

namespace mtl::functor::scalar {

template <typename T>
struct acosh {
    using result_type = T;

    static result_type apply(const T& v) {
        using std::acosh;
        return acosh(v);
    }
    result_type operator()(const T& v) const { return apply(v); }
};

} // namespace mtl::functor::scalar
