#pragma once
// MTL5 -- Algebraic shape classification (replaces MTL4 ashape.hpp)
// Used as secondary dispatch axis alongside category tags

namespace mtl::ashape {

struct universe {};
struct scal : universe {};
struct nonscal : universe {};

template <typename Value> struct rvec : nonscal {};
template <typename Value> struct cvec : nonscal {};
template <typename Value> struct mat  : nonscal {};

/// Primary template: assume scalar
template <typename T>
struct ashape {
    using type = scal;
};

} // namespace mtl::ashape
