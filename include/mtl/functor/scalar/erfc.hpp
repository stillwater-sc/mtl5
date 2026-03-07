#pragma once
// MTL5 — Scalar complementary error function functor
#include <cmath>

namespace mtl::functor::scalar {

template <typename T>
struct erfc {
    using result_type = T;

    static result_type apply(const T& v) {
        using std::erfc;
        return erfc(v);
    }
    result_type operator()(const T& v) const { return apply(v); }
};

} // namespace mtl::functor::scalar
