#pragma once
// MTL5 -- Scalar floor functor
#include <cmath>

namespace mtl::functor::scalar {

template <typename T>
struct floor {
    using result_type = T;

    static result_type apply(const T& v) {
        using std::floor;
        return floor(v);
    }
    result_type operator()(const T& v) const { return apply(v); }
};

} // namespace mtl::functor::scalar
