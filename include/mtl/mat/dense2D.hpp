#pragma once
// MTL5 — Dense 2D matrix (replaces MTL4 dense2D)
// Composition-based design with contiguous_memory_block, C++20 concepts
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <mtl/config.hpp>
#include <mtl/tag/orientation.hpp>
#include <mtl/tag/storage.hpp>
#include <mtl/tag/sparsity.hpp>
#include <mtl/traits/category.hpp>
#include <mtl/traits/ashape.hpp>
#include <mtl/detail/contiguous_memory_block.hpp>
#include <mtl/detail/index.hpp>
#include <mtl/mat/dimension.hpp>
#include <mtl/mat/parameter.hpp>

namespace mtl::mat {

/// Dense row/column-major matrix with configurable dimensions and storage.
template <typename Value, typename Parameters = parameters<>>
class dense2D {
public:
    // ── Parameter extraction ────────────────────────────────────────────
    using param_type   = Parameters;
    using orientation  = typename Parameters::orientation;
    using index_type   = typename Parameters::index_type;
    using dim_type     = typename Parameters::dimensions_type;
    using size_type    = typename Parameters::size_type;

    // Derive static size for memory block
    static constexpr bool is_fixed = dim_type::is_fixed;
    static constexpr std::size_t static_size = []() constexpr {
        if constexpr (is_fixed) return dim_type::rows * dim_type::cols;
        else return std::size_t{0};
    }();

    using actual_storage = typename Parameters::storage;
    using memory_type = detail::contiguous_memory_block<Value, actual_storage, static_size>;

    // ── Standard type aliases ───────────────────────────────────────────
    using value_type      = Value;
    using reference       = Value&;
    using const_reference = const Value&;
    using pointer         = Value*;
    using const_pointer   = const Value*;

    // ── Offset computation ──────────────────────────────────────────────
private:
    static constexpr size_type compute_offset(size_type r, size_type c, size_type ldim) {
        if constexpr (std::is_same_v<orientation, tag::row_major>)
            return r * ldim + c;
        else
            return c * ldim + r;
    }

    void set_ldim() {
        if constexpr (std::is_same_v<orientation, tag::row_major>)
            ldim_ = num_cols();
        else
            ldim_ = num_rows();
    }

public:
    // ── Default constructor ─────────────────────────────────────────────
    dense2D() : mem_{}, dims_{}, ldim_{0} {
        if constexpr (is_fixed) set_ldim();
    }

    // ── (rows, cols) constructor ────────────────────────────────────────
    dense2D(size_type rows, size_type cols)
        : mem_(rows * cols), dims_{}, ldim_{0}
    {
        if constexpr (!is_fixed) {
            dims_.set_dimensions(rows, cols);
        } else {
            assert(rows == dim_type::rows && cols == dim_type::cols
                   && "Size must match fixed dimensions");
        }
        set_ldim();
    }

    // ── Dimension object constructor ────────────────────────────────────
    explicit dense2D(const dim_type& d) requires (!is_fixed)
        : mem_(d.num_rows() * d.num_cols()), dims_(d), ldim_{0}
    {
        set_ldim();
    }

    // ── External pointer constructor ────────────────────────────────────
    dense2D(size_type rows, size_type cols, Value* ptr)
        : mem_(ptr, rows * cols, false), dims_{}, ldim_{0}
    {
        if constexpr (!is_fixed) {
            dims_.set_dimensions(rows, cols);
        }
        set_ldim();
    }

    // ── Initializer list of initializer lists ───────────────────────────
    dense2D(std::initializer_list<std::initializer_list<Value>> il)
        : dense2D(il.size(), il.size() > 0 ? il.begin()->size() : 0)
    {
        size_type r = 0;
        for (const auto& row : il) {
            size_type c = 0;
            for (const auto& val : row) {
                mem_[compute_offset(r, c, ldim_)] = val;
                ++c;
            }
            ++r;
        }
    }

    // ── Copy / Move ─────────────────────────────────────────────────────
    dense2D(const dense2D&) = default;
    dense2D& operator=(const dense2D&) = default;
    ~dense2D() = default;

    dense2D(dense2D&& other) noexcept
        : mem_(std::move(other.mem_)), dims_(other.dims_), ldim_(other.ldim_)
    {
        if constexpr (!is_fixed) other.dims_.set_dimensions(0, 0);
        other.ldim_ = 0;
    }

    dense2D& operator=(dense2D&& other) noexcept {
        if (this != &other) {
            mem_ = std::move(other.mem_);
            dims_ = other.dims_;
            ldim_ = other.ldim_;
            if constexpr (!is_fixed) other.dims_.set_dimensions(0, 0);
            other.ldim_ = 0;
        }
        return *this;
    }

    // ── Element access ──────────────────────────────────────────────────
    reference operator()(size_type r, size_type c) {
        auto ri = index_type::to_internal(r);
        auto ci = index_type::to_internal(c);
        if constexpr (bounds_checking) {
            if (ri >= num_rows() || ci >= num_cols())
                throw std::out_of_range("dense2D: index out of range");
        }
        return mem_[compute_offset(ri, ci, ldim_)];
    }

    const_reference operator()(size_type r, size_type c) const {
        auto ri = index_type::to_internal(r);
        auto ci = index_type::to_internal(c);
        if constexpr (bounds_checking) {
            if (ri >= num_rows() || ci >= num_cols())
                throw std::out_of_range("dense2D: index out of range");
        }
        return mem_[compute_offset(ri, ci, ldim_)];
    }

    // ── Size / shape ────────────────────────────────────────────────────
    size_type num_rows() const {
        if constexpr (is_fixed) return dim_type::rows;
        else return dims_.num_rows();
    }

    size_type num_cols() const {
        if constexpr (is_fixed) return dim_type::cols;
        else return dims_.num_cols();
    }

    size_type size() const { return num_rows() * num_cols(); }

    size_type get_ldim() const { return ldim_; }

    // ── Data access ─────────────────────────────────────────────────────
    pointer       data()       { return mem_.data(); }
    const_pointer data() const { return mem_.data(); }

    pointer       begin()       { return data(); }
    const_pointer begin() const { return data(); }
    pointer       end()         { return data() + size(); }
    const_pointer end()   const { return data() + size(); }

    // ── Dimension modification ──────────────────────────────────────────
    void change_dim(size_type r, size_type c) {
        if constexpr (is_fixed) {
            assert(r == dim_type::rows && c == dim_type::cols
                   && "Cannot change fixed dimensions");
        } else {
            auto new_sz = r * c;
            if (new_sz != size()) {
                mem_.realloc(new_sz);
            }
            dims_.set_dimensions(r, c);
            set_ldim();
        }
    }

    void checked_change_dim(size_type r, size_type c) {
        if constexpr (is_fixed) {
            if (r != dim_type::rows || c != dim_type::cols)
                throw std::domain_error("dense2D: cannot resize fixed-size matrix");
        } else {
            change_dim(r, c);
        }
    }

    // ── Swap ────────────────────────────────────────────────────────────
    void swap(dense2D& other) noexcept {
        mem_.swap(other.mem_);
        if constexpr (!is_fixed) {
            auto tr = dims_.num_rows(), tc = dims_.num_cols();
            dims_.set_dimensions(other.dims_.num_rows(), other.dims_.num_cols());
            other.dims_.set_dimensions(tr, tc);
        }
        using std::swap;
        swap(ldim_, other.ldim_);
    }

private:
    memory_type mem_;
    [[no_unique_address]] dim_type dims_;
    size_type ldim_;
};

// ── Free functions in mtl::mat ──────────────────────────────────────────

template <typename Value, typename Parameters>
auto num_rows(const dense2D<Value, Parameters>& m) { return m.num_rows(); }

template <typename Value, typename Parameters>
auto num_cols(const dense2D<Value, Parameters>& m) { return m.num_cols(); }

template <typename Value, typename Parameters>
auto size(const dense2D<Value, Parameters>& m) { return m.size(); }

} // namespace mtl::mat

// ── Traits specializations ─────────────────────────────────────────────

namespace mtl::traits {

template <typename Value, typename Parameters>
struct category<mat::dense2D<Value, Parameters>> {
    using type = tag::dense;
};

} // namespace mtl::traits

namespace mtl::ashape {

template <typename Value, typename Parameters>
struct ashape<::mtl::mat::dense2D<Value, Parameters>> {
    using type = mat<Value>;
};

} // namespace mtl::ashape

// ── Convenience alias ──────────────────────────────────────────────────
namespace mtl { using mat::dense2D; }
