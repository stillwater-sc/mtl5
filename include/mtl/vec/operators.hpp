#pragma once
// MTL5 -- Operator overloads for vector types (expression template returns)
#include <type_traits>
#include <cassert>
#include <mtl/concepts/vector.hpp>
#include <mtl/traits/is_expression.hpp>
#include <mtl/detail/expr_storage.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/functor/scalar/plus.hpp>
#include <mtl/functor/scalar/minus.hpp>
#include <mtl/functor/scalar/times.hpp>
#include <mtl/functor/scalar/divide.hpp>
#include <mtl/vec/expr/vec_vec_op_expr.hpp>
#include <mtl/vec/expr/vec_scal_op_expr.hpp>
#include <mtl/vec/expr/vec_negate_expr.hpp>

namespace mtl::vec {

// -- Binary arithmetic (lazy) -------------------------------------------
// Forwarding references: lvalue operands stored by const ref, rvalue by value.

template <typename V1, typename V2>
    requires (Vector<std::remove_cvref_t<V1>> && Vector<std::remove_cvref_t<V2>>)
auto operator+(V1&& a, V2&& b) {
    using S1 = detail::expr_store_t<V1>;
    using S2 = detail::expr_store_t<V2>;
    return expr::vec_vec_op_expr<S1, S2, functor::scalar::plus>(
        std::forward<V1>(a), std::forward<V2>(b));
}

template <typename V1, typename V2>
    requires (Vector<std::remove_cvref_t<V1>> && Vector<std::remove_cvref_t<V2>>)
auto operator-(V1&& a, V2&& b) {
    using S1 = detail::expr_store_t<V1>;
    using S2 = detail::expr_store_t<V2>;
    return expr::vec_vec_op_expr<S1, S2, functor::scalar::minus>(
        std::forward<V1>(a), std::forward<V2>(b));
}

// -- Unary negation (lazy) ----------------------------------------------

template <typename V>
    requires Vector<std::remove_cvref_t<V>>
auto operator-(V&& v) {
    using SV = detail::expr_store_t<V>;
    return expr::vec_negate_expr<SV>(std::forward<V>(v));
}

// -- Scalar-vector multiply (lazy) --------------------------------------

template <typename S, typename V>
    requires (std::is_arithmetic_v<S> && Vector<std::remove_cvref_t<V>>)
auto operator*(const S& alpha, V&& v) {
    using SV = detail::expr_store_t<V>;
    return expr::vec_scal_op_expr<S, SV, functor::scalar::times>(
        alpha, std::forward<V>(v));
}

template <typename V, typename S>
    requires (Vector<std::remove_cvref_t<V>> && std::is_arithmetic_v<S>)
auto operator*(V&& v, const S& alpha) {
    using SV = detail::expr_store_t<V>;
    return expr::vec_scal_op_expr<S, SV, functor::scalar::times>(
        alpha, std::forward<V>(v));
}

template <typename V, typename S>
    requires (Vector<std::remove_cvref_t<V>> && std::is_arithmetic_v<S>)
auto operator/(V&& v, const S& alpha) {
    using SV = detail::expr_store_t<V>;
    return expr::vec_rscal_op_expr<SV, S, functor::scalar::divide>(
        std::forward<V>(v), alpha);
}

} // namespace mtl::vec
