#pragma once
// MTL5 -- Mathematical tensor with compile-time rank and dimension
//
// tensor<T, Rank, Dim> stores Dim^Rank components on the stack and
// provides multi-index access, arithmetic, trace, and determinant.
//
// This is a true mathematical tensor, not an N-dimensional array.
// For NumPy-style arrays, use mtl::array::ndarray.

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <type_traits>

#include <mtl/config.hpp>
#include <mtl/tag/sparsity.hpp>
#include <mtl/tag/storage.hpp>
#include <mtl/traits/category.hpp>
#include <mtl/traits/ashape.hpp>
#include <mtl/traits/is_expression.hpp>
#include <mtl/detail/contiguous_memory_block.hpp>

namespace mtl::tensor {

// ── Helpers ────────────────────────────────────────────────────────

namespace detail {

/// Compile-time integer power: base^exp.
constexpr std::size_t ipow(std::size_t base, std::size_t exp) {
    std::size_t result = 1;
    for (std::size_t i = 0; i < exp; ++i) result *= base;
    return result;
}

/// Convert a variadic index pack (i, j, k, ...) to a flat offset.
/// In row-major-like order: offset = i * Dim^(Rank-1) + j * Dim^(Rank-2) + ... + last
template <std::size_t Dim, std::size_t Rank, typename... Indices>
constexpr std::size_t flat_index(Indices... indices) {
    static_assert(sizeof...(Indices) == Rank, "Number of indices must match tensor rank");
    const std::array<std::size_t, Rank> idx{static_cast<std::size_t>(indices)...};
    std::size_t offset = 0;
    std::size_t stride = 1;
    for (std::size_t i = Rank; i > 0; --i) {
        offset += idx[i - 1] * stride;
        stride *= Dim;
    }
    return offset;
}

/// Convert an array-based index to flat offset.
template <std::size_t Dim, std::size_t Rank>
constexpr std::size_t flat_index_arr(const std::array<std::size_t, Rank>& idx) {
    std::size_t offset = 0;
    std::size_t stride = 1;
    for (std::size_t i = Rank; i > 0; --i) {
        offset += idx[i - 1] * stride;
        stride *= Dim;
    }
    return offset;
}

} // namespace detail

// ── Core tensor class ──────────────────────────────────────────────

/// Mathematical tensor with compile-time rank and spatial dimension.
///
/// @tparam Value  scalar type (double, float, posit, etc.)
/// @tparam Rank   tensor rank (0=scalar, 1=vector, 2=matrix, 4=elasticity, ...)
/// @tparam Dim    spatial dimension (2D, 3D, 4D, ...)
template <typename Value, std::size_t Rank, std::size_t Dim>
class tensor {
public:
    // -- Type aliases --------------------------------------------------------
    using value_type      = Value;
    using reference       = Value&;
    using const_reference = const Value&;
    using pointer         = Value*;
    using const_pointer   = const Value*;
    using size_type       = std::size_t;

    static constexpr size_type rank      = Rank;
    static constexpr size_type dimension = Dim;
    static constexpr size_type num_components = detail::ipow(Dim, Rank);

private:
    using memory_type = mtl::detail::contiguous_memory_block<Value, tag::on_stack, num_components>;
    memory_type mem_;

public:
    // -- Constructors --------------------------------------------------------

    /// Default: zero-initialized.
    tensor() = default;

    /// Fill all components with a value.
    explicit tensor(const Value& val) {
        std::fill_n(mem_.data(), num_components, val);
    }

    /// Construct from initializer list (row-major order for rank-2).
    tensor(std::initializer_list<Value> il) {
        assert(il.size() <= num_components);
        std::copy(il.begin(), il.end(), mem_.data());
    }

    // -- Element access ------------------------------------------------------

    /// Multi-index access: T(i, j), T(i, j, k, l), etc.
    template <typename... Indices>
        requires (sizeof...(Indices) == Rank)
    reference operator()(Indices... indices) {
        const size_type off = detail::flat_index<Dim, Rank>(indices...);
        if constexpr (bounds_checking) check_bounds(off);
        return mem_[off];
    }

    template <typename... Indices>
        requires (sizeof...(Indices) == Rank)
    const_reference operator()(Indices... indices) const {
        const size_type off = detail::flat_index<Dim, Rank>(indices...);
        if constexpr (bounds_checking) check_bounds(off);
        return mem_[off];
    }

    /// Array-index access.
    reference operator[](const std::array<size_type, Rank>& idx) {
        const size_type off = detail::flat_index_arr<Dim, Rank>(idx);
        if constexpr (bounds_checking) check_bounds(off);
        return mem_[off];
    }

    const_reference operator[](const std::array<size_type, Rank>& idx) const {
        const size_type off = detail::flat_index_arr<Dim, Rank>(idx);
        if constexpr (bounds_checking) check_bounds(off);
        return mem_[off];
    }

    // -- Size / shape --------------------------------------------------------

    static constexpr size_type size() { return num_components; }

    // -- Data access ---------------------------------------------------------

    pointer       data()       { return mem_.data(); }
    const_pointer data() const { return mem_.data(); }

    pointer       begin()       { return mem_.begin(); }
    const_pointer begin() const { return mem_.begin(); }
    pointer       end()         { return mem_.end(); }
    const_pointer end()   const { return mem_.end(); }

    // -- Fill ----------------------------------------------------------------

    void fill(const Value& val) {
        std::fill_n(mem_.data(), num_components, val);
    }

    void set_to_zero() { fill(Value{0}); }

    // -- Arithmetic (element-wise) -------------------------------------------

    tensor& operator+=(const tensor& other) {
        for (size_type i = 0; i < num_components; ++i) mem_[i] += other.mem_[i];
        return *this;
    }

    tensor& operator-=(const tensor& other) {
        for (size_type i = 0; i < num_components; ++i) mem_[i] -= other.mem_[i];
        return *this;
    }

    tensor& operator*=(const Value& s) {
        for (size_type i = 0; i < num_components; ++i) mem_[i] *= s;
        return *this;
    }

    tensor& operator/=(const Value& s) {
        for (size_type i = 0; i < num_components; ++i) mem_[i] /= s;
        return *this;
    }

    friend tensor operator+(tensor a, const tensor& b) { return a += b; }
    friend tensor operator-(tensor a, const tensor& b) { return a -= b; }
    friend tensor operator*(tensor a, const Value& s) { return a *= s; }
    friend tensor operator*(const Value& s, tensor a) { return a *= s; }
    friend tensor operator/(tensor a, const Value& s) { return a /= s; }

    friend tensor operator-(tensor a) {
        for (size_type i = 0; i < num_components; ++i) a.mem_[i] = -a.mem_[i];
        return a;
    }

    // -- Comparison ----------------------------------------------------------

    bool operator==(const tensor& other) const {
        for (size_type i = 0; i < num_components; ++i)
            if (mem_[i] != other.mem_[i]) return false;
        return true;
    }

    bool operator!=(const tensor& other) const { return !(*this == other); }

private:
    void check_bounds(size_type off) const {
        if (off >= num_components)
            throw std::out_of_range("tensor: index out of bounds");
    }
};

// ── Free functions on tensor ───────────────────────────────────────

/// Trace of a rank-2 tensor: sum of diagonal elements.
template <typename V, std::size_t D>
V trace(const tensor<V, 2, D>& t) {
    V result = V{0};
    for (std::size_t i = 0; i < D; ++i) result += t(i, i);
    return result;
}

/// Determinant of a rank-2 tensor (2D and 3D specializations).
template <typename V>
V determinant(const tensor<V, 2, 2>& t) {
    return t(0, 0) * t(1, 1) - t(0, 1) * t(1, 0);
}

template <typename V>
V determinant(const tensor<V, 2, 3>& t) {
    return t(0, 0) * (t(1, 1) * t(2, 2) - t(1, 2) * t(2, 1))
         - t(0, 1) * (t(1, 0) * t(2, 2) - t(1, 2) * t(2, 0))
         + t(0, 2) * (t(1, 0) * t(2, 1) - t(1, 1) * t(2, 0));
}

/// Frobenius norm: sqrt(sum of squared components).
template <typename V, std::size_t R, std::size_t D>
V frobenius_norm(const tensor<V, R, D>& t) {
    V result = V{0};
    for (std::size_t i = 0; i < tensor<V, R, D>::num_components; ++i)
        result += t.data()[i] * t.data()[i];
    using std::sqrt;
    return sqrt(result);
}

/// Transpose of a rank-2 tensor.
template <typename V, std::size_t D>
tensor<V, 2, D> transpose(const tensor<V, 2, D>& t) {
    tensor<V, 2, D> result;
    for (std::size_t i = 0; i < D; ++i)
        for (std::size_t j = 0; j < D; ++j)
            result(i, j) = t(j, i);
    return result;
}

/// Identity tensor (rank-2): Kronecker delta.
template <typename V, std::size_t D>
tensor<V, 2, D> identity() {
    tensor<V, 2, D> result;
    for (std::size_t i = 0; i < D; ++i) result(i, i) = V{1};
    return result;
}

} // namespace mtl::tensor

// -- Trait specializations ---------------------------------------------------

namespace mtl::traits {

template <typename V, std::size_t R, std::size_t D>
struct category<::mtl::tensor::tensor<V, R, D>> {
    using type = tag::dense;
};

template <typename V, std::size_t R, std::size_t D>
struct is_expression<::mtl::tensor::tensor<V, R, D>> : std::false_type {};

} // namespace mtl::traits

namespace mtl::ashape {

template <typename V, std::size_t R, std::size_t D>
struct ashape<::mtl::tensor::tensor<V, R, D>> {
    using type = nonscal;
};

} // namespace mtl::ashape
