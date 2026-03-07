#pragma once
// MTL5 -- Matrix dimension types (replaces MTL4 matrix::dimensions)
// fixed::dimensions<R,C> for compile-time sizes, non_fixed::dimensions for runtime
#include <cstddef>

namespace mtl::mat {

namespace fixed {

/// Compile-time fixed matrix dimensions
template <std::size_t Rows, std::size_t Cols>
struct dimensions {
    static constexpr std::size_t rows = Rows;
    static constexpr std::size_t cols = Cols;
    static constexpr bool is_fixed = true;

    constexpr std::size_t num_rows() const { return Rows; }
    constexpr std::size_t num_cols() const { return Cols; }
    constexpr std::size_t size() const { return Rows * Cols; }
};

} // namespace fixed

namespace non_fixed {

/// Runtime matrix dimensions
struct dimensions {
    static constexpr bool is_fixed = false;

    constexpr dimensions() : rows_(0), cols_(0) {}
    constexpr dimensions(std::size_t r, std::size_t c) : rows_(r), cols_(c) {}

    constexpr std::size_t num_rows() const { return rows_; }
    constexpr std::size_t num_cols() const { return cols_; }
    constexpr std::size_t size() const { return rows_ * cols_; }

    constexpr void set_dimensions(std::size_t r, std::size_t c) {
        rows_ = r;
        cols_ = c;
    }

private:
    std::size_t rows_;
    std::size_t cols_;
};

} // namespace non_fixed

} // namespace mtl::mat
