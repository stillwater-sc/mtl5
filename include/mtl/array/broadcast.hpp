#pragma once
// MTL5 -- Broadcasting element-wise operations for ndarray
//
// Implements NumPy broadcasting rules via lazy expression templates.
// Binary operations (+, -, *, /) between ndarrays of compatible shapes
// produce broadcast_expr objects that are evaluated on assignment.

#include <array>
#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <type_traits>

#include <mtl/array/ndarray.hpp>
#include <mtl/array/shape.hpp>

namespace mtl::array {

// ── Broadcast expression ───────────────────────────────────────────

/// Lazy expression for element-wise binary operations with broadcasting.
template <typename LHS, typename RHS, typename Op, std::size_t N>
class broadcast_expr {
    const LHS& lhs_;
    const RHS& rhs_;
    Op op_;
    shape<N> result_shape_;
    std::array<std::size_t, N> lhs_strides_;
    std::array<std::size_t, N> rhs_strides_;

public:
    using value_type = decltype(std::declval<Op>()(
        std::declval<typename LHS::value_type>(),
        std::declval<typename RHS::value_type>()));
    using size_type  = std::size_t;
    static constexpr size_type rank = N;

    broadcast_expr(const LHS& lhs, const RHS& rhs, Op op)
        : lhs_(lhs), rhs_(rhs), op_(op) {
        bool ok = broadcast_shape(lhs.get_shape(), rhs.get_shape(), result_shape_);
        if (!ok) throw std::invalid_argument("ndarray: incompatible shapes for broadcasting");
        lhs_strides_ = broadcast_strides(lhs.get_shape(), lhs.get_strides(), result_shape_);
        rhs_strides_ = broadcast_strides(rhs.get_shape(), rhs.get_strides(), result_shape_);
    }

    const ::mtl::array::shape<N>& get_shape() const { return result_shape_; }

    size_type size() const { return result_shape_.total_size(); }

    /// Evaluate at a multi-index.
    value_type operator[](const std::array<size_type, N>& idx) const {
        size_type lhs_off = compute_offset(idx, lhs_strides_);
        size_type rhs_off = compute_offset(idx, rhs_strides_);
        return op_(lhs_.data()[lhs_off], rhs_.data()[rhs_off]);
    }

    /// Materialize the expression into a new ndarray.
    ndarray<value_type, N> eval() const {
        ndarray<value_type, N> result(result_shape_);
        result.iterate_indices([&](const std::array<size_type, N>& idx) {
            result[idx] = (*this)[idx];
        });
        return result;
    }

    /// Allow implicit conversion to ndarray (materialization).
    operator ndarray<value_type, N>() const { return eval(); }
};

// ── Functors ───────────────────────────────────────────────────────

namespace ops {
    struct plus  { template <typename T, typename U> auto operator()(T a, U b) const { return a + b; } };
    struct minus { template <typename T, typename U> auto operator()(T a, U b) const { return a - b; } };
    struct times { template <typename T, typename U> auto operator()(T a, U b) const { return a * b; } };
    struct divides { template <typename T, typename U> auto operator()(T a, U b) const { return a / b; } };
} // namespace ops

// ── Operators ──────────────────────────────────────────────────────

template <typename V1, std::size_t N, typename O1, typename V2, typename O2>
auto operator+(const ndarray<V1, N, O1>& a, const ndarray<V2, N, O2>& b) {
    return broadcast_expr<ndarray<V1, N, O1>, ndarray<V2, N, O2>, ops::plus, N>(a, b, ops::plus{});
}

template <typename V1, std::size_t N, typename O1, typename V2, typename O2>
auto operator-(const ndarray<V1, N, O1>& a, const ndarray<V2, N, O2>& b) {
    return broadcast_expr<ndarray<V1, N, O1>, ndarray<V2, N, O2>, ops::minus, N>(a, b, ops::minus{});
}

template <typename V1, std::size_t N, typename O1, typename V2, typename O2>
auto operator*(const ndarray<V1, N, O1>& a, const ndarray<V2, N, O2>& b) {
    return broadcast_expr<ndarray<V1, N, O1>, ndarray<V2, N, O2>, ops::times, N>(a, b, ops::times{});
}

template <typename V1, std::size_t N, typename O1, typename V2, typename O2>
auto operator/(const ndarray<V1, N, O1>& a, const ndarray<V2, N, O2>& b) {
    return broadcast_expr<ndarray<V1, N, O1>, ndarray<V2, N, O2>, ops::divides, N>(a, b, ops::divides{});
}

} // namespace mtl::array
