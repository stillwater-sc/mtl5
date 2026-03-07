#pragma once
// MTL5 -- Scalar natural logarithm functor
#include <cmath>

namespace mtl::functor::scalar {

template <typename T>
struct log {
    using result_type = T;

    static result_type apply(const T& v) {
        using std::log;
        return log(v);
    }
    result_type operator()(const T& v) const { return apply(v); }
};

} // namespace mtl::functor::scalar
