#pragma once
// MTL5 -- Block diagonal matrix: stores a sequence of dense2D blocks on the diagonal.
// Off-diagonal blocks are implicitly zero. Efficient matvec: apply each block independently.
#include <algorithm>
#include <cstddef>
#include <vector>
#include <cassert>
#include <initializer_list>
#include <mtl/math/identity.hpp>
#include <mtl/tag/sparsity.hpp>
#include <mtl/traits/category.hpp>
#include <mtl/traits/ashape.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>

namespace mtl::mat {

/// Block diagonal matrix -- stores vector of dense2D blocks.
/// Total dimension is sum of block dimensions.
template <typename Value = double>
class block_diagonal2D {
public:
    using value_type      = Value;
    using size_type       = std::size_t;
    using const_reference = Value;
    using reference       = Value;

    block_diagonal2D() = default;

    /// Construct from a vector of dense blocks.
    explicit block_diagonal2D(std::vector<dense2D<Value>> blocks)
        : blocks_(std::move(blocks)) {
        recompute_offsets();
    }

    /// Construct from initializer list of dense blocks.
    block_diagonal2D(std::initializer_list<dense2D<Value>> blocks)
        : blocks_(blocks) {
        recompute_offsets();
    }

    /// Add a block to the diagonal. O(1): it appends the new block's
    /// offsets, where it recomputed all of them, O(n) per call (#826).
    void add_block(dense2D<Value> block) {
        row_offsets_.push_back(total_rows_);
        col_offsets_.push_back(total_cols_);
        total_rows_ += block.num_rows();
        total_cols_ += block.num_cols();
        blocks_.push_back(std::move(block));
    }

    /// Element (r, c). The block holding row r is found by binary search over
    /// the row offsets: O(log blocks), where it scanned every block (#826).
    /// A block with no rows holds no element, so the last block starting at
    /// or before r is the only candidate.
    value_type operator()(size_type r, size_type c) const {
        if (blocks_.empty()) return math::zero<Value>();
        const auto it = std::upper_bound(row_offsets_.begin(), row_offsets_.end(), r);
        if (it == row_offsets_.begin()) return math::zero<Value>();
        const auto b = static_cast<size_type>(it - row_offsets_.begin()) - 1;
        const size_type r0 = row_offsets_[b];
        const size_type c0 = col_offsets_[b];
        if (r < r0 + blocks_[b].num_rows() && c >= c0 && c < c0 + blocks_[b].num_cols())
            return blocks_[b](r - r0, c - c0);
        return math::zero<Value>();
    }

    size_type num_rows() const { return total_rows_; }
    size_type num_cols() const { return total_cols_; }
    size_type size()     const { return total_rows_ * total_cols_; }

    /// Number of diagonal blocks.
    size_type num_blocks() const { return blocks_.size(); }

    /// Access the k-th block.
    const dense2D<Value>& block(size_type k) const { return blocks_[k]; }

    /// Write access to the k-th block's elements: a dense2D over the block's
    /// storage. Writing its elements writes the block. It used to return the
    /// block itself, and change_dim on that left the offsets and the total
    /// shape stale (#826); resizing this one detaches it from the block,
    /// which keeps its shape. Replace a block's shape with set_block.
    dense2D<Value> block(size_type k) {
        assert(k < blocks_.size());
        return dense2D<Value>(blocks_[k].num_rows(), blocks_[k].num_cols(), blocks_[k].data());
    }

    /// Replace the k-th block, of any shape, and update the offsets.
    void set_block(size_type k, dense2D<Value> block) {
        assert(k < blocks_.size());
        blocks_[k] = std::move(block);
        recompute_offsets();
    }

private:
    std::vector<dense2D<Value>> blocks_;
    std::vector<size_type> row_offsets_;
    std::vector<size_type> col_offsets_;
    size_type total_rows_ = 0;
    size_type total_cols_ = 0;

    void recompute_offsets() {
        row_offsets_.resize(blocks_.size());
        col_offsets_.resize(blocks_.size());
        total_rows_ = 0;
        total_cols_ = 0;
        for (size_type b = 0; b < blocks_.size(); ++b) {
            row_offsets_[b] = total_rows_;
            col_offsets_[b] = total_cols_;
            total_rows_ += blocks_[b].num_rows();
            total_cols_ += blocks_[b].num_cols();
        }
    }
};

// -- Efficient block-diagonal matvec ------------------------------------
// Apply each block independently: O(sum of block_rows * block_cols)

template <typename Value, typename VV, typename VP>
auto operator*(const block_diagonal2D<Value>& BD,
               const vec::dense_vector<VV, VP>& x) {
    assert(BD.num_cols() == x.size());
    using result_t = std::common_type_t<Value, VV>;
    vec::dense_vector<result_t> y(BD.num_rows(), math::zero<result_t>());

    size_t row_off = 0;
    size_t col_off = 0;
    for (size_t b = 0; b < BD.num_blocks(); ++b) {
        const auto& blk = BD.block(b);
        for (size_t r = 0; r < blk.num_rows(); ++r) {
            auto acc = math::zero<result_t>();
            for (size_t c = 0; c < blk.num_cols(); ++c)
                acc += static_cast<result_t>(blk(r, c))
                     * static_cast<result_t>(x(col_off + c));
            y(row_off + r) = acc;
        }
        row_off += blk.num_rows();
        col_off += blk.num_cols();
    }
    return y;
}

} // namespace mtl::mat

namespace mtl::traits {
template <typename Value>
struct category<mat::block_diagonal2D<Value>> { using type = tag::sparse; };
} // namespace mtl::traits

namespace mtl::ashape {
template <typename Value>
struct ashape<::mtl::mat::block_diagonal2D<Value>> { using type = mat<Value>; };
} // namespace mtl::ashape

namespace mtl { using mat::block_diagonal2D; }
