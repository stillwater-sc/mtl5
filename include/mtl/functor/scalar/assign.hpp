#pragma once
// MTL5 — Scalar assignment functor
#include <type_traits>

namespace mtl::functor::scalar {

template <typename T1, typename T2 = T1>
struct assign {
    using result_type = T1&;

    static constexpr T1& apply(T1& a, const T2& b) { return a = b; }
    constexpr T1& operator()(T1& a, const T2& b) const { return apply(a, b); }
};

} // namespace mtl::functor::scalar
