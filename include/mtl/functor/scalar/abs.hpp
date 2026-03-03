#pragma once
// MTL5 — Scalar absolute value functor
#include <cmath>
#include <cstdlib>
#include <complex>
#include <mtl/concepts/magnitude.hpp>

namespace mtl::functor::scalar {

template <typename T>
struct abs {
    using result_type = magnitude_t<T>;

    static constexpr result_type apply(const T& v) {
        using std::abs;
        return abs(v);
    }
    constexpr result_type operator()(const T& v) const { return apply(v); }
};

} // namespace mtl::functor::scalar
