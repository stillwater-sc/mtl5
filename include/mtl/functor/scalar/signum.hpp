#pragma once
// MTL5 — Scalar signum (sign function) functor

namespace mtl::functor::scalar {

template <typename T>
struct signum {
    using result_type = T;

    static constexpr result_type apply(const T& v) {
        return (v > T{0}) ? T{1} : (v < T{0}) ? T{-1} : T{0};
    }
    constexpr result_type operator()(const T& v) const { return apply(v); }
};

} // namespace mtl::functor::scalar
