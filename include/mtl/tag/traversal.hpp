#pragma once
// MTL5 — Traversal tags (replaces MTL4 glas_tag + tag:: traversal tags)
namespace mtl::tag {

struct nz {};     // non-zero elements only
struct all {};    // all elements (including structural zeros)
struct row {};    // row-wise traversal
struct col {};    // column-wise traversal
struct major {};  // major-order traversal
struct minor {};  // minor-order traversal

namespace iter {
    struct nz {};   // non-zero iterator
    struct all {};  // all-element iterator
} // namespace iter

namespace const_iter {
    struct nz {};
    struct all {};
} // namespace const_iter

} // namespace mtl::tag
