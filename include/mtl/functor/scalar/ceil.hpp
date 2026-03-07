#pragma once
// MTL5 -- Scalar ceiling functor
#include <cmath>

namespace mtl::functor::scalar {

template <typename T>
struct ceil {
    using result_type = T;

    static result_type apply(const T& v) {
        using std::ceil;
        return ceil(v);
    }
    result_type operator()(const T& v) const { return apply(v); }
};

} // namespace mtl::functor::scalar
