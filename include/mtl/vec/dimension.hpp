#pragma once
// MTL5 -- Vector dimension types (replaces MTL4 vector dimensions)
// fixed::dimension<N> for compile-time size, non_fixed::dimension for runtime
#include <cstddef>

namespace mtl::vec {

namespace fixed {

/// Compile-time fixed vector dimension
template <std::size_t N>
struct dimension {
    static constexpr std::size_t value = N;
    static constexpr bool is_fixed = true;

    constexpr std::size_t size() const { return N; }
};

} // namespace fixed

namespace non_fixed {

/// Runtime vector dimension
struct dimension {
    static constexpr std::size_t value = 0;  // signals dynamic size
    static constexpr bool is_fixed = false;

    constexpr dimension() : size_(0) {}
    constexpr explicit dimension(std::size_t n) : size_(n) {}

    constexpr std::size_t size() const { return size_; }

    constexpr void set_size(std::size_t n) { size_ = n; }

private:
    std::size_t size_;
};

} // namespace non_fixed

} // namespace mtl::vec
