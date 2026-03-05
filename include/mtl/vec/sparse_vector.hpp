#pragma once
// MTL5 — Sparse vector using dual sorted arrays (indices + values)
// Port from MTL4: boost/numeric/mtl/vector/sparse_vector.hpp
// Key changes: C++20, no Boost dependencies, std::lower_bound for O(log n) lookup

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <stdexcept>
#include <utility>
#include <vector>

#include <mtl/config.hpp>
#include <mtl/tag/orientation.hpp>
#include <mtl/tag/sparsity.hpp>
#include <mtl/traits/category.hpp>
#include <mtl/traits/ashape.hpp>
#include <mtl/concepts/vector.hpp>
#include <mtl/vec/parameter.hpp>

namespace mtl::vec {

/// Sparse vector stored as dual sorted arrays (indices + values).
/// Indices are maintained in sorted order for O(log n) lookup.
template <typename Value, typename Parameters = parameters<>>
class sparse_vector {
public:
    using param_type      = Parameters;
    using orientation     = typename Parameters::orientation;
    using size_type       = typename Parameters::size_type;
    using value_type      = Value;
    using reference       = Value&;
    using const_reference = const Value&;

    // ── Iterator over (index, value) pairs ─────────────────────────────
    class const_iterator {
    public:
        using difference_type   = std::ptrdiff_t;
        using value_type        = std::pair<size_type, Value>;
        using pointer           = const value_type*;
        using reference         = value_type;
        using iterator_category = std::input_iterator_tag;

        const_iterator(const sparse_vector* vec, size_type pos)
            : vec_(vec), pos_(pos) {}

        reference operator*() const {
            return {vec_->indices_[pos_], vec_->values_[pos_]};
        }

        const_iterator& operator++() { ++pos_; return *this; }
        const_iterator operator++(int) { auto tmp = *this; ++pos_; return tmp; }

        bool operator==(const const_iterator& o) const { return pos_ == o.pos_; }
        bool operator!=(const const_iterator& o) const { return pos_ != o.pos_; }

    private:
        const sparse_vector* vec_;
        size_type pos_;
    };

    // ── Constructors ───────────────────────────────────────────────────
    sparse_vector() : dim_(0) {}

    explicit sparse_vector(size_type n) : dim_(n) {}

    // ── Size / shape ───────────────────────────────────────────────────
    size_type size() const { return dim_; }
    size_type nnz()  const { return static_cast<size_type>(indices_.size()); }
    bool      empty() const { return dim_ == 0; }

    /// Orientation-aware dimensions (same pattern as dense_vector)
    size_type num_rows() const {
        if constexpr (std::is_same_v<orientation, tag::col_major>) return size();
        else return 1;
    }

    size_type num_cols() const {
        if constexpr (std::is_same_v<orientation, tag::col_major>) return 1;
        else return size();
    }

    // ── Read access — returns zero if absent ───────────────────────────
    value_type operator()(size_type i) const {
        if constexpr (bounds_checking) {
            if (i >= dim_) throw std::out_of_range("sparse_vector: index out of range");
        }
        auto it = std::lower_bound(indices_.begin(), indices_.end(), i);
        auto pos = static_cast<size_type>(it - indices_.begin());
        if (it != indices_.end() && *it == i)
            return values_[pos];
        return value_type{};
    }

    // ── Read/write access — inserts if absent ──────────────────────────
    reference operator[](size_type i) {
        if constexpr (bounds_checking) {
            if (i >= dim_) throw std::out_of_range("sparse_vector: index out of range");
        }
        auto it = std::lower_bound(indices_.begin(), indices_.end(), i);
        auto pos = static_cast<size_type>(it - indices_.begin());
        if (it != indices_.end() && *it == i)
            return values_[pos];
        // Insert maintaining sorted order
        indices_.insert(indices_.begin() + pos, i);
        values_.insert(values_.begin() + pos, value_type{});
        return values_[pos];
    }

    // ── Existence check ────────────────────────────────────────────────
    bool exists(size_type i) const {
        auto it = std::lower_bound(indices_.begin(), indices_.end(), i);
        return it != indices_.end() && *it == i;
    }

    // ── Insert maintaining sorted order ────────────────────────────────
    void insert(size_type i, const Value& v) {
        assert(i < dim_);
        auto it = std::lower_bound(indices_.begin(), indices_.end(), i);
        auto pos = static_cast<size_type>(it - indices_.begin());
        if (it != indices_.end() && *it == i) {
            values_[pos] = v; // overwrite existing
        } else {
            indices_.insert(indices_.begin() + pos, i);
            values_.insert(values_.begin() + pos, v);
        }
    }

    // ── Clear all entries ──────────────────────────────────────────────
    void clear() {
        indices_.clear();
        values_.clear();
    }

    // ── Drop entries below threshold ───────────────────────────────────
    void crop(const Value& threshold) {
        std::vector<size_type> new_idx;
        std::vector<Value> new_val;
        for (size_type k = 0; k < nnz(); ++k) {
            using std::abs;
            if (abs(values_[k]) >= threshold) {
                new_idx.push_back(indices_[k]);
                new_val.push_back(values_[k]);
            }
        }
        indices_ = std::move(new_idx);
        values_  = std::move(new_val);
    }

    // ── Raw access to internal arrays ──────────────────────────────────
    const std::vector<size_type>& indices() const { return indices_; }
    const std::vector<Value>&     values()  const { return values_; }

    // ── Iterators over (index, value) pairs ────────────────────────────
    const_iterator begin() const { return const_iterator(this, 0); }
    const_iterator end()   const { return const_iterator(this, nnz()); }

private:
    size_type dim_;
    std::vector<size_type> indices_;
    std::vector<Value>     values_;
};

// ── Free functions ──────────────────────────────────────────────────────

template <typename Value, typename Parameters>
auto size(const sparse_vector<Value, Parameters>& v) { return v.size(); }

template <typename Value, typename Parameters>
auto num_rows(const sparse_vector<Value, Parameters>& v) { return v.num_rows(); }

template <typename Value, typename Parameters>
auto num_cols(const sparse_vector<Value, Parameters>& v) { return v.num_cols(); }

} // namespace mtl::vec

// ── Traits specializations ──────────────────────────────────────────────

namespace mtl::traits {

template <typename Value, typename Parameters>
struct category<vec::sparse_vector<Value, Parameters>> {
    using type = tag::sparse;
};

} // namespace mtl::traits

namespace mtl::ashape {

template <typename Value, typename Parameters>
struct ashape<vec::sparse_vector<Value, Parameters>> {
    using type = std::conditional_t<
        std::is_same_v<typename Parameters::orientation, tag::col_major>,
        cvec<Value>,
        rvec<Value>
    >;
};

} // namespace mtl::ashape

// ── Convenience alias ───────────────────────────────────────────────────
namespace mtl { using vec::sparse_vector; }
