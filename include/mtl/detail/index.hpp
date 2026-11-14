#pragma once
// MTL5 -- Index types (replaces MTL4 c_index / f_index)
// c_index: 0-based (C-style), f_index: 1-based (Fortran-style)
#include <cstddef>

namespace mtl::detail {

/// 0-based index (C/C++ convention) -- default
struct c_index {
    static constexpr std::size_t base = 0;
    static constexpr std::size_t to_internal(std::size_t i) { return i; }
    static constexpr std::size_t to_external(std::size_t i) { return i; }
};

/// 1-based index (Fortran convention). Index 0 is out of range: to_internal(0)
/// wraps to SIZE_MAX, which every bounds check rejects (dense2D's element
/// access checks ri >= num_rows() when bounds_checking is on), and which is
/// undefined to use without one, like any out-of-range index (#838).
struct f_index {
    static constexpr std::size_t base = 1;
    static constexpr std::size_t to_internal(std::size_t i) { return i - 1; }
    static constexpr std::size_t to_external(std::size_t i) { return i + 1; }
};

} // namespace mtl::detail
