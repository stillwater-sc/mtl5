#pragma once
// MTL5 -- Transposed orientation metafunction
#include <mtl/tag/orientation.hpp>

namespace mtl::traits {

/// Flip row_major <-> col_major
template <typename Orientation>
struct transposed_orientation;

template <>
struct transposed_orientation<tag::row_major> {
    using type = tag::col_major;
};

template <>
struct transposed_orientation<tag::col_major> {
    using type = tag::row_major;
};

template <typename Orientation>
using transposed_orientation_t = typename transposed_orientation<Orientation>::type;

} // namespace mtl::traits
