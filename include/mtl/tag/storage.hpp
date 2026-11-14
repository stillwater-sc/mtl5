#pragma once
// MTL5 -- Storage tags (on_stack/on_heap selection)
namespace mtl::tag {

struct on_stack {};
struct on_heap {};

/// A read-only view of memory the caller owns: a dense_vector or dense2D with
/// this storage is built over a `const Value*` and offers only const access.
/// It never allocates, never frees and never writes; writing through it is a
/// compile error. See vec::const_view and mat::const_view.
struct read_only {};

} // namespace mtl::tag
