#pragma once
// MTL5 -- Scalar hyperbolic tangent functor
#include <cmath>

namespace mtl::functor::scalar {

template <typename T>
struct tanh {
    using result_type = T;

    static result_type apply(const T& v) {
        using std::tanh;
        return tanh(v);
    }
    result_type operator()(const T& v) const { return apply(v); }
};

} // namespace mtl::functor::scalar
