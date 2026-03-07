#pragma once
// MTL5 -- Block-recursive matrix traversal (recursator)
// Port from MTL4: boost/numeric/mtl/recursion/matrix_recursator.hpp
// Key changes: value semantics (no shared_ptr), C++20, constexpr utilities

#include <cstddef>
#include <cassert>
#include <bit>

namespace mtl::recursion {

// -- Utility functions --------------------------------------------------

/// True if n is a power of 2
constexpr bool is_power_of_2(std::size_t n) {
    return n > 0 && (n & (n - 1)) == 0;
}

/// Largest power of 2 that is <= n (for n >= 1)
constexpr std::size_t first_part(std::size_t n) {
    if (n == 0) return 0;
    // bit_floor returns the largest power of 2 <= n
    return std::bit_floor(n);
}

/// Smallest power of 2 that is >= n
constexpr std::size_t outer_bound(std::size_t n) {
    if (n == 0) return 0;
    return std::bit_ceil(n);
}

// -- Recursator ---------------------------------------------------------

/// Wraps a matrix and provides quad-tree subdivision for block-recursive
/// algorithms. Lightweight value type -- copy freely.
template <typename Matrix>
class recursator {
public:
    using size_type  = typename Matrix::size_type;
    using value_type = typename Matrix::value_type;

    /// Wrap the entire matrix
    explicit recursator(Matrix& m)
        : mat_(&m),
          row_offset_(0), col_offset_(0),
          nrows_(m.num_rows()), ncols_(m.num_cols()) {}

    // -- Quadrant subdivision -------------------------------------------

    /// North-west quadrant (top-left)
    recursator north_west() const {
        auto rs = row_split();
        auto cs = col_split();
        return recursator(*mat_, row_offset_, col_offset_, rs, cs);
    }

    /// North-east quadrant (top-right)
    recursator north_east() const {
        auto rs = row_split();
        auto cs = col_split();
        return recursator(*mat_, row_offset_, col_offset_ + cs,
                          rs, ncols_ - cs);
    }

    /// South-west quadrant (bottom-left)
    recursator south_west() const {
        auto rs = row_split();
        auto cs = col_split();
        return recursator(*mat_, row_offset_ + rs, col_offset_,
                          nrows_ - rs, cs);
    }

    /// South-east quadrant (bottom-right)
    recursator south_east() const {
        auto rs = row_split();
        auto cs = col_split();
        return recursator(*mat_, row_offset_ + rs, col_offset_ + cs,
                          nrows_ - rs, ncols_ - cs);
    }

    // -- Access ---------------------------------------------------------

    /// Access element (r, c) relative to this sub-region
    decltype(auto) operator()(size_type r, size_type c) const {
        return (*mat_)(row_offset_ + r, col_offset_ + c);
    }

    decltype(auto) operator()(size_type r, size_type c) {
        return (*mat_)(row_offset_ + r, col_offset_ + c);
    }

    /// Get the underlying matrix reference
    Matrix& get_matrix() { return *mat_; }
    const Matrix& get_matrix() const { return *mat_; }

    // -- Dimensions -----------------------------------------------------

    size_type num_rows() const { return nrows_; }
    size_type num_cols() const { return ncols_; }
    size_type size()     const { return nrows_ * ncols_; }
    bool      is_empty() const { return nrows_ == 0 || ncols_ == 0; }

    size_type row_offset() const { return row_offset_; }
    size_type col_offset() const { return col_offset_; }

private:
    /// Private constructor for sub-regions
    recursator(Matrix& m, size_type r_off, size_type c_off,
               size_type nr, size_type nc)
        : mat_(&m), row_offset_(r_off), col_offset_(c_off),
          nrows_(nr), ncols_(nc) {}

    /// Split point for rows: largest power of 2 <= nrows (but < nrows if nrows > 1)
    size_type row_split() const {
        if (nrows_ <= 1) return nrows_;
        auto fp = first_part(nrows_);
        return (fp == nrows_) ? fp / 2 : fp;
    }

    /// Split point for columns
    size_type col_split() const {
        if (ncols_ <= 1) return ncols_;
        auto fp = first_part(ncols_);
        return (fp == ncols_) ? fp / 2 : fp;
    }

    Matrix*   mat_;
    size_type row_offset_;
    size_type col_offset_;
    size_type nrows_;
    size_type ncols_;
};

// -- Recursive traversal ------------------------------------------------

/// Recursively apply function to all base-case sub-regions of the recursator.
template <typename Matrix, typename Function, typename BaseCaseTest>
void for_each(recursator<Matrix> rec, Function&& fn, const BaseCaseTest& is_base) {
    if (rec.is_empty()) return;

    if (is_base(rec)) {
        fn(rec);
        return;
    }

    for_each(rec.north_west(), fn, is_base);
    for_each(rec.north_east(), fn, is_base);
    for_each(rec.south_west(), fn, is_base);
    for_each(rec.south_east(), fn, is_base);
}

} // namespace mtl::recursion
