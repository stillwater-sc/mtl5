#pragma once
// MTL5 -- Sparse vector using dual sorted arrays (indices + values)
// Port from MTL4: boost/numeric/mtl/vector/sparse_vector.hpp
// Key changes: C++20, no Boost dependencies, std::lower_bound for O(log n) lookup

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <mtl/config.hpp>
#include <mtl/concepts/magnitude.hpp>
#include <mtl/detail/exceptions.hpp>
#include <mtl/tag/orientation.hpp>
#include <mtl/tag/sparsity.hpp>
#include <mtl/traits/category.hpp>
#include <mtl/traits/ashape.hpp>
#include <mtl/concepts/vector.hpp>
#include <mtl/vec/parameter.hpp>

namespace mtl::vec {

/// Sparse vector stored as dual sorted arrays (indices + values).
/// Indices are maintained in sorted order for O(log n) lookup.
///
/// Index checks (#830), by the "check once" policy of #825/#826:
///   - insert() and operator[], which store an index, check it in every
///     build and throw std::out_of_range (through detail::throw_or_abort, so
///     a -fno-exceptions build aborts). There is no finalize step to check
///     once at, so each insertion checks: one comparison next to the
///     O(log nnz) search.
///   - operator(), a read, checks only when bounds_checking is on, as the
///     other containers' element access does.
///   - exists() returns false for an index outside the vector.
/// A signed size_type rejects negative indices too.
template <typename Value, typename Parameters = parameters<>>
class sparse_vector {
public:
    using param_type      = Parameters;
    using orientation     = typename Parameters::orientation;
    using size_type       = typename Parameters::size_type;
    using value_type      = Value;
    using reference       = Value&;
    using const_reference = const Value&;

    // -- Iterator over (index, value) pairs -----------------------------
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

    // -- Constructors ---------------------------------------------------
    sparse_vector() : dim_(0) {}

    explicit sparse_vector(size_type n) : dim_(n) {}

    // -- Size / shape ---------------------------------------------------
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

    // -- Read access -- returns zero if absent ---------------------------
    value_type operator()(size_type i) const {
        if constexpr (bounds_checking) {
            if (!in_range(i))
                ::mtl::detail::throw_or_abort(std::out_of_range("sparse_vector: index out of range"));
        }
        auto it = std::lower_bound(indices_.begin(), indices_.end(), i);
        auto pos = static_cast<size_type>(it - indices_.begin());
        if (it != indices_.end() && *it == i)
            return values_[pos];
        return value_type{};
    }

    // -- Read/write access -- inserts if absent --------------------------
    reference operator[](size_type i) {
        check_insert_index(i);   // it inserts, so it checks in every build
        auto it = std::lower_bound(indices_.begin(), indices_.end(), i);
        auto pos = static_cast<size_type>(it - indices_.begin());
        if (it != indices_.end() && *it == i)
            return values_[pos];
        // Insert maintaining sorted order
        indices_.insert(indices_.begin() + pos, i);
        values_.insert(values_.begin() + pos, value_type{});
        return values_[pos];
    }

    // -- Existence check ------------------------------------------------
    bool exists(size_type i) const {
        if (!in_range(i)) return false;
        auto it = std::lower_bound(indices_.begin(), indices_.end(), i);
        return it != indices_.end() && *it == i;
    }

    // -- Insert maintaining sorted order --------------------------------
    void insert(size_type i, const Value& v) {
        check_insert_index(i);
        auto it = std::lower_bound(indices_.begin(), indices_.end(), i);
        auto pos = static_cast<size_type>(it - indices_.begin());
        if (it != indices_.end() && *it == i) {
            values_[pos] = v; // overwrite existing
        } else {
            indices_.insert(indices_.begin() + pos, i);
            values_.insert(values_.begin() + pos, v);
        }
    }

    // -- Clear all entries ----------------------------------------------
    void clear() {
        indices_.clear();
        values_.clear();
    }

    /// Reserve storage for nnz entries, so that many inserts don't
    /// reallocate. Out-of-order insertion still moves the later entries.
    void reserve(size_type nnz) {
        indices_.reserve(static_cast<std::size_t>(nnz));
        values_.reserve(static_cast<std::size_t>(nnz));
    }

    // -- Drop entries below threshold -----------------------------------
    /// Drops the entries whose magnitude is below threshold, in place. The
    /// threshold is a magnitude (the real type, for a complex Value). A NaN
    /// entry is kept: it is not below any threshold, and dropping it would
    /// hide it (#830).
    void crop(const magnitude_t<Value>& threshold) {
        using std::abs;
        std::size_t out = 0;
        for (std::size_t k = 0; k < indices_.size(); ++k) {
            if (!(abs(values_[k]) < threshold)) {
                if (out != k) {
                    indices_[out] = indices_[k];
                    values_[out]  = std::move(values_[k]);
                }
                ++out;
            }
        }
        indices_.erase(indices_.begin() + static_cast<std::ptrdiff_t>(out), indices_.end());
        values_.erase(values_.begin() + static_cast<std::ptrdiff_t>(out), values_.end());
    }

    // -- Raw access to internal arrays ----------------------------------
    const std::vector<size_type>& indices() const { return indices_; }
    const std::vector<Value>&     values()  const { return values_; }

    // -- Iterators over (index, value) pairs ----------------------------
    const_iterator begin() const { return const_iterator(this, 0); }
    const_iterator end()   const { return const_iterator(this, nnz()); }

private:
    bool in_range(size_type i) const {
        return std::cmp_greater_equal(i, 0) && std::cmp_less(i, dim_);
    }

    void check_insert_index(size_type i) const {
        if (!in_range(i))
            ::mtl::detail::throw_or_abort(std::out_of_range("sparse_vector: index out of range"));
    }

    size_type dim_;
    std::vector<size_type> indices_;
    std::vector<Value>     values_;
};

// -- Free functions ------------------------------------------------------

template <typename Value, typename Parameters>
auto size(const sparse_vector<Value, Parameters>& v) { return v.size(); }

template <typename Value, typename Parameters>
auto num_rows(const sparse_vector<Value, Parameters>& v) { return v.num_rows(); }

template <typename Value, typename Parameters>
auto num_cols(const sparse_vector<Value, Parameters>& v) { return v.num_cols(); }

} // namespace mtl::vec

// -- Traits specializations ----------------------------------------------

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

// -- Convenience alias ---------------------------------------------------
namespace mtl { using vec::sparse_vector; }
