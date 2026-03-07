#pragma once
// MTL5 -- Orientation tags (replaces MTL4 tag::row_major/col_major)
namespace mtl::tag {

struct row_major {};
struct col_major {};

// MTL4 compatibility alias
using column_major = col_major;

} // namespace mtl::tag
