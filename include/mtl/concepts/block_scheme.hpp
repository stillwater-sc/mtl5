#pragma once
// MTL5 -- BlockScheme: the geometry and reference decoder of a block-quantized
// storage format (#629).
//
// Block-quantized data stores groups of `block_size` values in `block_bytes`
// bytes, typically a shared scale plus narrow integers or minifloats: GGML's
// Q4_0/Q8_0/Q4_K, the OCP microscaling formats, NVIDIA-style 16-wide blocks.
// MTL5 defines the concept; a library supplies the schemes. The OCP MX formats
// ship in-tree (mat/mx_schemes.hpp) because they are a vendor-neutral standard.
// GGML's stay with their consumer, because their correctness is defined by a
// llama.cpp version, not by a specification.
//
// A scheme works on RAW BYTES. Block storage typically comes from a file or a
// memory map and carries no C++ objects, so a scheme reads it the way the
// format specifies, with memcpy where it needs multi-byte fields, not through
// a reinterpret_cast to a struct.

#include <concepts>
#include <cstddef>

namespace mtl {

/// A block-quantized storage format:
///
///   * `S::block_size`  -- elements per block;
///   * `S::block_bytes` -- bytes per block, as stored;
///   * `S::decode(src, dst)` -- the REFERENCE decode of the block at `src` into
///     `block_size` floats at `dst`. Fused kernels are optimizations measured
///     against this.
///
/// Optionally, `S::decode_element(src, j)` decodes element `j` of the block
/// alone. block_matrix_view uses it for element access when it exists, and
/// decodes the whole block otherwise.
template <typename S>
concept BlockScheme = requires(const std::byte* src, float* dst) {
    { S::block_size }  -> std::convertible_to<std::size_t>;
    { S::block_bytes } -> std::convertible_to<std::size_t>;
    S::decode(src, dst);
} && (S::block_size > 0) && (S::block_bytes > 0);

/// Does the scheme decode one element without decoding its block?
template <typename S>
concept ElementDecodableScheme = BlockScheme<S> && requires(const std::byte* src, std::size_t j) {
    { S::decode_element(src, j) } -> std::convertible_to<float>;
};

} // namespace mtl
