#pragma once
// MTL5 — Dense vector (replaces MTL4 dense_vector)
// Composition-based design with contiguous_memory_block, C++20 concepts
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <mtl/config.hpp>
#include <mtl/tag/orientation.hpp>
#include <mtl/tag/storage.hpp>
#include <mtl/tag/sparsity.hpp>
#include <mtl/traits/category.hpp>
#include <mtl/traits/ashape.hpp>
#include <mtl/concepts/scalar.hpp>
#include <mtl/concepts/vector.hpp>
#include <mtl/traits/is_expression.hpp>
#include <mtl/detail/contiguous_memory_block.hpp>
#include <mtl/vec/dimension.hpp>
#include <mtl/vec/parameter.hpp>

namespace mtl::vec {

/// Dense vector with configurable orientation, storage, and dimensions.
template <typename Value, typename Parameters = parameters<>>
class dense_vector {
public:
    // ── Parameter extraction ────────────────────────────────────────────
    using param_type   = Parameters;
    using orientation  = typename Parameters::orientation;
    using dim_type     = typename Parameters::dimensions_type;
    using size_type    = typename Parameters::size_type;

    // Derive static size: N for fixed, 0 for dynamic
    static constexpr std::size_t static_size = dim_type::value;
    using actual_storage = typename Parameters::storage;
    using memory_type = detail::contiguous_memory_block<Value, actual_storage, static_size>;

    // ── Standard type aliases ───────────────────────────────────────────
    using value_type      = Value;
    using reference       = Value&;
    using const_reference = const Value&;
    using pointer         = Value*;
    using const_pointer   = const Value*;

    // ── Default constructor ─────────────────────────────────────────────
    dense_vector() : mem_{}, dim_{} {}

    // ── Size constructor (dynamic only) ─────────────────────────────────
    explicit dense_vector(size_type n) : mem_(n), dim_{} {
        if constexpr (!dim_type::is_fixed) {
            dim_.set_size(n);
        } else {
            assert(n == static_size && "Size must match fixed dimension");
        }
    }

    // ── Size + fill value constructor ───────────────────────────────────
    dense_vector(size_type n, const Value& val) : dense_vector(n) {
        std::fill_n(mem_.data(), size(), val);
    }

    // ── External pointer constructor ────────────────────────────────────
    dense_vector(size_type n, Value* ptr) : mem_(ptr, n, false), dim_{} {
        if constexpr (!dim_type::is_fixed) {
            dim_.set_size(n);
        }
    }

    // ── Initializer list constructor ────────────────────────────────────
    dense_vector(std::initializer_list<Value> il) : dense_vector(il.size()) {
        std::copy(il.begin(), il.end(), mem_.data());
    }

    // ── Construct from std::vector ──────────────────────────────────────
    explicit dense_vector(const std::vector<Value>& v) : dense_vector(v.size()) {
        std::copy(v.begin(), v.end(), mem_.data());
    }

    // ── Copy / Move ─────────────────────────────────────────────────────
    dense_vector(const dense_vector&) = default;
    dense_vector& operator=(const dense_vector&) = default;
    ~dense_vector() = default;

    dense_vector(dense_vector&& other) noexcept
        : mem_(std::move(other.mem_)), dim_(other.dim_)
    {
        if constexpr (!dim_type::is_fixed) other.dim_.set_size(0);
    }

    dense_vector& operator=(dense_vector&& other) noexcept {
        if (this != &other) {
            mem_ = std::move(other.mem_);
            dim_ = other.dim_;
            if constexpr (!dim_type::is_fixed) other.dim_.set_size(0);
        }
        return *this;
    }

    // ── Expression template construction / assignment ───────────────────
    template <typename Expr>
        requires (Vector<Expr> && traits::is_expression_v<Expr>
                  && std::convertible_to<typename Expr::value_type, Value>)
    dense_vector(const Expr& expr) : dense_vector(static_cast<size_type>(expr.size())) {
        for (size_type i = 0; i < size(); ++i)
            (*this)(i) = static_cast<Value>(expr(i));
    }

    template <typename Expr>
        requires (Vector<Expr> && traits::is_expression_v<Expr>
                  && std::convertible_to<typename Expr::value_type, Value>)
    dense_vector& operator=(const Expr& expr) {
        change_dim(static_cast<size_type>(expr.size()));
        for (size_type i = 0; i < size(); ++i)
            (*this)(i) = static_cast<Value>(expr(i));
        return *this;
    }

    template <typename Expr>
        requires (Vector<Expr> && traits::is_expression_v<Expr>
                  && std::convertible_to<typename Expr::value_type, Value>)
    dense_vector& operator+=(const Expr& expr) {
        assert(size() == expr.size());
        for (size_type i = 0; i < size(); ++i)
            (*this)(i) += static_cast<Value>(expr(i));
        return *this;
    }

    template <typename Expr>
        requires (Vector<Expr> && traits::is_expression_v<Expr>
                  && std::convertible_to<typename Expr::value_type, Value>)
    dense_vector& operator-=(const Expr& expr) {
        assert(size() == expr.size());
        for (size_type i = 0; i < size(); ++i)
            (*this)(i) -= static_cast<Value>(expr(i));
        return *this;
    }

    // ── Element access ──────────────────────────────────────────────────
    reference operator()(size_type i) {
        if constexpr (bounds_checking) {
            if (i >= size()) throw std::out_of_range("dense_vector: index out of range");
        }
        return mem_[i];
    }

    const_reference operator()(size_type i) const {
        if constexpr (bounds_checking) {
            if (i >= size()) throw std::out_of_range("dense_vector: index out of range");
        }
        return mem_[i];
    }

    reference operator[](size_type i) { return operator()(i); }
    const_reference operator[](size_type i) const { return operator()(i); }

    // ── Size / shape ────────────────────────────────────────────────────
    size_type size() const {
        if constexpr (dim_type::is_fixed) return static_size;
        else return dim_.size();
    }

    bool empty() const { return size() == 0; }

    /// Orientation-aware row/column dimensions
    size_type num_rows() const {
        if constexpr (std::is_same_v<orientation, tag::col_major>) return size();
        else return 1;
    }

    size_type num_cols() const {
        if constexpr (std::is_same_v<orientation, tag::col_major>) return 1;
        else return size();
    }

    /// Stride is always 1 for dense vectors
    static constexpr size_type stride() { return 1; }

    // ── Data access ─────────────────────────────────────────────────────
    pointer       data()       { return mem_.data(); }
    const_pointer data() const { return mem_.data(); }

    pointer       begin()       { return data(); }
    const_pointer begin() const { return data(); }
    pointer       end()         { return data() + size(); }
    const_pointer end()   const { return data() + size(); }

    // ── Dimension modification ──────────────────────────────────────────
    void change_dim(size_type n) {
        if constexpr (dim_type::is_fixed) {
            assert(n == static_size && "Cannot change fixed dimension");
        } else {
            if (n != size()) {
                mem_.realloc(n);
                dim_.set_size(n);
            }
        }
    }

    void checked_change_dim(size_type n) {
        if constexpr (dim_type::is_fixed) {
            if (n != static_size)
                throw std::domain_error("dense_vector: cannot resize fixed-size vector");
        } else {
            change_dim(n);
        }
    }

    // ── Compound assignment operators ───────────────────────────────────
    dense_vector& operator+=(const dense_vector& other) {
        assert(size() == other.size());
        for (size_type i = 0; i < size(); ++i) mem_[i] += other.mem_[i];
        return *this;
    }

    dense_vector& operator-=(const dense_vector& other) {
        assert(size() == other.size());
        for (size_type i = 0; i < size(); ++i) mem_[i] -= other.mem_[i];
        return *this;
    }

    template <typename S> requires Scalar<S>
    dense_vector& operator*=(const S& alpha) {
        for (size_type i = 0; i < size(); ++i) mem_[i] *= alpha;
        return *this;
    }

    template <typename S> requires Field<S>
    dense_vector& operator/=(const S& alpha) {
        for (size_type i = 0; i < size(); ++i) mem_[i] /= alpha;
        return *this;
    }

    // ── Swap ────────────────────────────────────────────────────────────
    void swap(dense_vector& other) noexcept {
        mem_.swap(other.mem_);
        if constexpr (!dim_type::is_fixed) {
            using std::swap;
            auto tmp = dim_.size();
            dim_.set_size(other.dim_.size());
            other.dim_.set_size(tmp);
        }
    }

private:
    memory_type mem_;
    [[no_unique_address]] dim_type dim_;
};

// ── Free functions in mtl::vec ──────────────────────────────────────────

template <typename Value, typename Parameters>
auto size(const dense_vector<Value, Parameters>& v) { return v.size(); }

template <typename Value, typename Parameters>
auto num_rows(const dense_vector<Value, Parameters>& v) { return v.num_rows(); }

template <typename Value, typename Parameters>
auto num_cols(const dense_vector<Value, Parameters>& v) { return v.num_cols(); }

template <typename Value, typename Parameters>
void fill(dense_vector<Value, Parameters>& v, const Value& val) {
    std::fill(v.begin(), v.end(), val);
}

} // namespace mtl::vec

// ── Traits specializations ─────────────────────────────────────────────

namespace mtl::traits {

template <typename Value, typename Parameters>
struct category<vec::dense_vector<Value, Parameters>> {
    using type = tag::dense;
};

} // namespace mtl::traits

namespace mtl::ashape {

template <typename Value, typename Parameters>
struct ashape<vec::dense_vector<Value, Parameters>> {
    using type = std::conditional_t<
        std::is_same_v<typename Parameters::orientation, tag::col_major>,
        cvec<Value>,
        rvec<Value>
    >;
};

} // namespace mtl::ashape

// ── Convenience alias ──────────────────────────────────────────────────
namespace mtl { using vec::dense_vector; }
