#pragma once
// MTL5 -- Scalar base-10 logarithm functor
#include <cmath>

namespace mtl::functor::scalar {

template <typename T>
struct log10 {
    using result_type = T;

    static result_type apply(const T& v) {
        using std::log10;
        return log10(v);
    }
    result_type operator()(const T& v) const { return apply(v); }
};

} // namespace mtl::functor::scalar
