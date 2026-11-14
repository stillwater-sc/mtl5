#pragma once
// MTL5 -- Scalar absolute value functor
#include <cmath>
#include <cstdlib>
#include <complex>
#include <type_traits>
#include <mtl/concepts/magnitude.hpp>

namespace mtl::functor::scalar {

template <typename T>
struct abs {
    using result_type = magnitude_t<T>;

    static constexpr result_type apply(const T& v) {
        if constexpr (std::is_unsigned_v<T>) {
            return v;                    // std::abs(unsigned) is ambiguous (#838)
        } else {
            using std::abs;
            return abs(v);
        }
    }
    constexpr result_type operator()(const T& v) const { return apply(v); }
};

} // namespace mtl::functor::scalar
