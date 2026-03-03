#pragma once
// MTL5 — Left-scale functor: alpha * x
#include <type_traits>
#include <mtl/concepts/scalar.hpp>

namespace mtl::functor::typed {

template <typename S>
    requires Scalar<S>
struct scale {
    explicit constexpr scale(const S& alpha) : alpha_(alpha) {}

    template <typename T>
    constexpr auto operator()(const T& x) const { return alpha_ * x; }

private:
    S alpha_;
};

} // namespace mtl::functor::typed
