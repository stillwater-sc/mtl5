#pragma once
// MTL5 — Compile-time configuration knobs
#include <cstddef>

namespace mtl {

/// Default size type for indices and dimensions
using default_size_type = std::size_t;

/// Threshold below which matrices are stored on the stack
inline constexpr std::size_t stack_size_limit = 256;

/// Default block size for recursive operations
inline constexpr std::size_t default_block_size = 64;

/// Enable/disable bounds checking (debug builds)
#ifdef NDEBUG
inline constexpr bool bounds_checking = false;
#else
inline constexpr bool bounds_checking = true;
#endif

} // namespace mtl
