#pragma once
// MTL5 -- Scalar exponential functor
#include <cmath>

namespace mtl::functor::scalar {

template <typename T>
struct exp {
    using result_type = T;

    static result_type apply(const T& v) {
        using std::exp;
        return exp(v);
    }
    result_type operator()(const T& v) const { return apply(v); }
};

} // namespace mtl::functor::scalar
