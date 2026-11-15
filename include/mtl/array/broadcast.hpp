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
#include <utility>

#include <mtl/config.hpp>
#include <mtl/array/ndarray.hpp>
#include <mtl/array/shape.hpp>

namespace mtl::array {

// -- Broadcast expression -------------------------------------------

/// Lazy expression for element-wise binary operations with broadcasting.
///
/// LHS and RHS are the operand types as stored: `const A&` for an lvalue
/// operand, `A` itself for an rvalue one, so `make_arr() + b` and
/// `auto e = a.transpose() + b` keep their temporaries alive (#856: both were
/// held by const reference and dangled). An ndarray view is cheap to copy; an
/// owning temporary is moved in.
template <typename LHS, typename RHS, typename Op, std::size_t N>
class broadcast_expr {
    using lhs_array = std::remove_cvref_t<LHS>;
    using rhs_array = std::remove_cvref_t<RHS>;

    LHS lhs_;
    RHS rhs_;
    Op op_;
    shape<N> result_shape_;
    std::array<std::size_t, N> lhs_strides_;
    std::array<std::size_t, N> rhs_strides_;

public:
    using value_type = std::remove_cv_t<decltype(std::declval<Op>()(
        std::declval<typename lhs_array::value_type>(),
        std::declval<typename rhs_array::value_type>()))>;
    using size_type  = std::size_t;
    static constexpr size_type rank = N;

    template <typename L, typename R>
    broadcast_expr(L&& lhs, R&& rhs, Op op)
        : lhs_(std::forward<L>(lhs)), rhs_(std::forward<R>(rhs)), op_(op) {
        bool ok = broadcast_shape(lhs_.get_shape(), rhs_.get_shape(), result_shape_);
        if (!ok) throw std::invalid_argument("ndarray: incompatible shapes for broadcasting");
        lhs_strides_ = broadcast_strides(lhs_.get_shape(), lhs_.get_strides(), result_shape_);
        rhs_strides_ = broadcast_strides(rhs_.get_shape(), rhs_.get_strides(), result_shape_);
    }

    const ::mtl::array::shape<N>& get_shape() const { return result_shape_; }

    size_type size() const { return result_shape_.total_size(); }

    /// Evaluate at a multi-index. Checked against the result's shape when
    /// bounds_checking is on, as ndarray's operator[] is (#856).
    value_type operator[](const std::array<size_type, N>& idx) const {
        if constexpr (bounds_checking) {
            for (size_type i = 0; i < N; ++i)
                if (idx[i] >= result_shape_[i])
                    throw std::out_of_range("broadcast_expr: index out of bounds");
        }
        size_type lhs_off = compute_offset(idx, lhs_strides_);
        size_type rhs_off = compute_offset(idx, rhs_strides_);
        return op_(lhs_.data()[lhs_off], rhs_.data()[rhs_off]);
    }

    /// Materialize the expression into a new ndarray: one walk over the
    /// result, with the three arrays' offsets advanced together (#856: two
    /// offset computations and a checked write per element before).
    ndarray<value_type, N> eval() const {
        ndarray<value_type, N> result(result_shape_);
        value_type* dst = result.data();
        const auto* l = lhs_.data();
        const auto* r = rhs_.data();
        detail_nd::walk_offsets<c_order>(
            result_shape_,
            std::array<std::array<size_type, N>, 3>{result.get_strides(), lhs_strides_, rhs_strides_},
            [&](const auto& o) { dst[o[0]] = op_(l[o[1]], r[o[2]]); });
        return result;
    }

    /// Allow implicit conversion to ndarray (materialization).
    operator ndarray<value_type, N>() const { return eval(); }
};

namespace detail_bcast {

template <typename T> struct is_ndarray : std::false_type {};
template <typename V, std::size_t N, typename O>
struct is_ndarray<ndarray<V, N, O>> : std::true_type {};

template <typename T>
concept Array = is_ndarray<std::remove_cvref_t<T>>::value;

/// How an operand is stored: by const reference if it is an lvalue, by value
/// if it is a temporary.
template <typename T>
using stored_t = std::conditional_t<std::is_lvalue_reference_v<T>,
                                    const std::remove_reference_t<T>&, std::remove_cvref_t<T>>;

template <typename Op, typename A, typename B>
auto make(A&& a, B&& b) {
    constexpr std::size_t N = std::remove_cvref_t<A>::rank;
    static_assert(N == std::remove_cvref_t<B>::rank, "broadcasting needs operands of one rank");
    return broadcast_expr<stored_t<A>, stored_t<B>, Op, N>(std::forward<A>(a), std::forward<B>(b), Op{});
}

} // namespace detail_bcast

// -- Functors -------------------------------------------------------

namespace ops {
    struct plus  { template <typename T, typename U> auto operator()(T a, U b) const { return a + b; } };
    struct minus { template <typename T, typename U> auto operator()(T a, U b) const { return a - b; } };
    struct times { template <typename T, typename U> auto operator()(T a, U b) const { return a * b; } };
    struct divides { template <typename T, typename U> auto operator()(T a, U b) const { return a / b; } };
} // namespace ops

// -- Operators ------------------------------------------------------

template <detail_bcast::Array A, detail_bcast::Array B>
auto operator+(A&& a, B&& b) {
    return detail_bcast::make<ops::plus>(std::forward<A>(a), std::forward<B>(b));
}

template <detail_bcast::Array A, detail_bcast::Array B>
auto operator-(A&& a, B&& b) {
    return detail_bcast::make<ops::minus>(std::forward<A>(a), std::forward<B>(b));
}

template <detail_bcast::Array A, detail_bcast::Array B>
auto operator*(A&& a, B&& b) {
    return detail_bcast::make<ops::times>(std::forward<A>(a), std::forward<B>(b));
}

template <detail_bcast::Array A, detail_bcast::Array B>
auto operator/(A&& a, B&& b) {
    return detail_bcast::make<ops::divides>(std::forward<A>(a), std::forward<B>(b));
}

} // namespace mtl::array
