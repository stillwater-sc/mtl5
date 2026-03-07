#pragma once
// MTL5 -- Division functor: x / alpha
#include <type_traits>
#include <mtl/concepts/scalar.hpp>

namespace mtl::functor::typed {

template <typename S>
    requires Field<S>
struct divide_by {
    explicit constexpr divide_by(const S& alpha) : alpha_(alpha) {}

    template <typename T>
    constexpr auto operator()(const T& x) const { return x / alpha_; }

private:
    S alpha_;
};

} // namespace mtl::functor::typed
