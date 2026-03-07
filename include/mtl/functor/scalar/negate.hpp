#pragma once
// MTL5 -- Scalar negation functor

namespace mtl::functor::scalar {

template <typename T>
struct negate {
    using result_type = T;

    static constexpr T apply(const T& v) { return -v; }
    constexpr T operator()(const T& v) const { return apply(v); }
};

} // namespace mtl::functor::scalar
