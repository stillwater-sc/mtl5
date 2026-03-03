#pragma once
// MTL5 — Matrix parameter bundle (replaces MTL4 matrix::parameters)
// Collects orientation, index style, dimensions, storage, and size_type
#include <cstddef>
#include <mtl/tag/orientation.hpp>
#include <mtl/tag/storage.hpp>
#include <mtl/detail/index.hpp>
#include <mtl/mat/dimension.hpp>

namespace mtl::mat {

/// Parameter bundle for matrix types
template <
    typename Orientation = tag::row_major,
    typename Index       = detail::c_index,
    typename Dimensions  = non_fixed::dimensions,
    typename Storage     = tag::on_heap,
    typename SizeType    = std::size_t
>
struct parameters {
    using orientation = Orientation;
    using index_type  = Index;
    using dimensions_type = Dimensions;
    using storage     = Storage;
    using size_type   = SizeType;

    static constexpr bool is_fixed = Dimensions::is_fixed;
    static_assert(!std::is_same_v<Storage, tag::on_stack> || is_fixed,
        "Stack storage requires fixed-size dimensions");

    static_assert(
        std::is_same_v<Orientation, tag::row_major> ||
        std::is_same_v<Orientation, tag::col_major>,
        "Orientation must be tag::row_major or tag::col_major"
    );
};

} // namespace mtl::mat
