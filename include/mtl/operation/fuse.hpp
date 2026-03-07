#pragma once
// MTL5 -- Fused evaluation: evaluate expressions directly into pre-allocated targets
#include <cassert>
#include <mtl/traits/is_expression.hpp>
#include <mtl/concepts/matrix.hpp>
#include <mtl/concepts/vector.hpp>

namespace mtl {

/// Evaluate a matrix expression into a pre-allocated target: dest = expr
template <typename Dest, Matrix Expr>
    requires traits::is_expression_v<Expr>
void fused_assign(Dest& dest, const Expr& expr) {
    using size_type = typename Expr::size_type;
    dest.change_dim(static_cast<typename Dest::size_type>(expr.num_rows()),
                    static_cast<typename Dest::size_type>(expr.num_cols()));
    for (size_type r = 0; r < expr.num_rows(); ++r)
        for (size_type c = 0; c < expr.num_cols(); ++c)
            dest(r, c) = static_cast<typename Dest::value_type>(expr(r, c));
}

/// Evaluate a vector expression into a pre-allocated target: dest = expr
template <typename Dest, Vector Expr>
    requires (traits::is_expression_v<Expr> && !Matrix<Expr>)
void fused_assign(Dest& dest, const Expr& expr) {
    using size_type = typename Expr::size_type;
    dest.change_dim(static_cast<typename Dest::size_type>(expr.size()));
    for (size_type i = 0; i < expr.size(); ++i)
        dest(i) = static_cast<typename Dest::value_type>(expr(i));
}

/// Evaluate a matrix expression and add to target: dest += expr
template <typename Dest, Matrix Expr>
    requires traits::is_expression_v<Expr>
void fused_plus_assign(Dest& dest, const Expr& expr) {
    using size_type = typename Expr::size_type;
    assert(dest.num_rows() == expr.num_rows() && dest.num_cols() == expr.num_cols());
    for (size_type r = 0; r < expr.num_rows(); ++r)
        for (size_type c = 0; c < expr.num_cols(); ++c)
            dest(r, c) += static_cast<typename Dest::value_type>(expr(r, c));
}

/// Evaluate a vector expression and add to target: dest += expr
template <typename Dest, Vector Expr>
    requires (traits::is_expression_v<Expr> && !Matrix<Expr>)
void fused_plus_assign(Dest& dest, const Expr& expr) {
    using size_type = typename Expr::size_type;
    assert(dest.size() == expr.size());
    for (size_type i = 0; i < expr.size(); ++i)
        dest(i) += static_cast<typename Dest::value_type>(expr(i));
}

} // namespace mtl
