#pragma once
// MTL5 — Scalar subtraction functor
#include <type_traits>

namespace mtl::functor::scalar {

template <typename T1, typename T2 = T1>
struct minus {
    using result_type = std::common_type_t<T1, T2>;

    static constexpr result_type apply(const T1& a, const T2& b) { return a - b; }
    constexpr result_type operator()(const T1& a, const T2& b) const { return apply(a, b); }
};

} // namespace mtl::functor::scalar
