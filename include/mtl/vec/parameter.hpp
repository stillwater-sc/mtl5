#pragma once
// MTL5 — Vector parameter bundle
#include <cstddef>
#include <mtl/tag/orientation.hpp>
#include <mtl/tag/storage.hpp>
#include <mtl/vec/dimension.hpp>

namespace mtl::vec {

/// Parameter bundle for vector types
template <
    typename Orientation = tag::col_major,
    typename Dimensions  = non_fixed::dimension,
    typename Storage     = tag::on_heap,
    typename SizeType    = std::size_t
>
struct parameters {
    using orientation = Orientation;
    using dimensions_type = Dimensions;
    using storage     = Storage;
    using size_type   = SizeType;
};

} // namespace mtl::vec
