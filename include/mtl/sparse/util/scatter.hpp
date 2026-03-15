#pragma once
// MTL5 -- Sparse accumulator (scatter/gather) for sparse direct solvers
// The sparse accumulator is the workhorse of sparse column assembly.
// It provides O(1) amortized insert/lookup by maintaining a dense work
// vector with a timestamp-based validity check, avoiding explicit clearing.

#include <cassert>
#include <cstddef>
#include <limits>
#include <vector>

namespace mtl::sparse::util {

/// Sparse accumulator for assembling a sparse column.
/// Uses a dense workspace with generation-based marking to avoid clearing.
/// This is the "scatter" data structure from CSparse (cs_scatter).
template <typename Value, typename SizeType = std::size_t>
class sparse_accumulator {
public:
    explicit sparse_accumulator(SizeType n)
        : n_(n), x_(n, Value{0}), mark_(n, SizeType{0}), generation_(1) {}

    /// Reset for a new column assembly. O(1) — just bumps the generation counter.
    void clear() {
        ++generation_;
        // Handle wraparound (extremely unlikely in practice)
        if (generation_ == 0) {
            std::fill(mark_.begin(), mark_.end(), SizeType{0});
            generation_ = 1;
        }
        indices_.clear();
    }

    /// Scatter value into position i: x[i] += val.
    /// If i hasn't been touched in the current generation, initializes to val.
    void scatter(SizeType i, const Value& val) {
        assert(i < n_);
        if (mark_[i] != generation_) {
            mark_[i] = generation_;
            x_[i] = val;
            indices_.push_back(i);
        } else {
            x_[i] += val;
        }
    }

    /// Store value at position i (overwrite, don't accumulate).
    void store(SizeType i, const Value& val) {
        assert(i < n_);
        if (mark_[i] != generation_) {
            mark_[i] = generation_;
            indices_.push_back(i);
        }
        x_[i] = val;
    }

    /// Check if position i has been touched in the current generation.
    bool is_set(SizeType i) const {
        assert(i < n_);
        return mark_[i] == generation_;
    }

    /// Read value at position i. Returns 0 if not set.
    Value operator()(SizeType i) const {
        assert(i < n_);
        return (mark_[i] == generation_) ? x_[i] : Value{0};
    }

    /// Mutable access to value at position i.
    Value& operator[](SizeType i) {
        assert(i < n_);
        if (mark_[i] != generation_) {
            mark_[i] = generation_;
            x_[i] = Value{0};
            indices_.push_back(i);
        }
        return x_[i];
    }

    /// Return the indices that have been touched in the current generation.
    const std::vector<SizeType>& indices() const { return indices_; }

    /// Number of nonzeros currently accumulated.
    SizeType nnz() const { return static_cast<SizeType>(indices_.size()); }

    /// Dimension of the accumulator.
    SizeType size() const { return n_; }

private:
    SizeType n_;
    std::vector<Value>    x_;           // dense workspace
    std::vector<SizeType> mark_;        // generation stamps
    SizeType              generation_;  // current generation
    std::vector<SizeType> indices_;     // touched indices in current generation
};

} // namespace mtl::sparse::util
