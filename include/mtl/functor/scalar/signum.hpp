#pragma once
// MTL5 -- Scalar signum (sign function) functor
#include <type_traits>

namespace mtl::functor::scalar {

template <typename T>
struct signum {
    using result_type = T;

    /// 1, -1 or 0. An unsigned value is never negative, and T{-1} would not
    /// even compile for it. A NaN is returned as is, as NumPy's sign does,
    /// rather than reported as 0 (#838).
    static constexpr result_type apply(const T& v) {
        if constexpr (std::is_unsigned_v<T>) {
            return (v > T{0}) ? T{1} : T{0};
        } else {
            if (!(v == v)) return v;     // NaN
            return (v > T{0}) ? T{1} : (v < T{0}) ? T{-1} : T{0};
        }
    }
    constexpr result_type operator()(const T& v) const { return apply(v); }
};

} // namespace mtl::functor::scalar
