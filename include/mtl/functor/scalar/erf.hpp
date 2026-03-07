#pragma once
// MTL5 -- Scalar error function functor
#include <cmath>

namespace mtl::functor::scalar {

template <typename T>
struct erf {
    using result_type = T;

    static result_type apply(const T& v) {
        using std::erf;
        return erf(v);
    }
    result_type operator()(const T& v) const { return apply(v); }
};

} // namespace mtl::functor::scalar
