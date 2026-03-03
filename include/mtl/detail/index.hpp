#pragma once
// MTL5 — Index types (replaces MTL4 c_index / f_index)
// c_index: 0-based (C-style), f_index: 1-based (Fortran-style)
#include <cstddef>

namespace mtl::detail {

/// 0-based index (C/C++ convention) — default
struct c_index {
    static constexpr std::size_t base = 0;
    static constexpr std::size_t to_internal(std::size_t i) { return i; }
    static constexpr std::size_t to_external(std::size_t i) { return i; }
};

/// 1-based index (Fortran convention)
struct f_index {
    static constexpr std::size_t base = 1;
    static constexpr std::size_t to_internal(std::size_t i) { return i - 1; }
    static constexpr std::size_t to_external(std::size_t i) { return i + 1; }
};

} // namespace mtl::detail
