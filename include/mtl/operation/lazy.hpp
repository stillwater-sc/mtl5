#pragma once
// MTL5 -- Lazy evaluation: materialize expressions into concrete types
#include <mtl/traits/is_expression.hpp>
#include <mtl/concepts/matrix.hpp>
#include <mtl/concepts/vector.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>

namespace mtl {

/// Materialize a matrix expression into a concrete dense2D
template <typename Expr>
    requires (Matrix<Expr> && traits::is_expression_v<Expr>)
auto evaluate(const Expr& expr) {
    using V = typename Expr::value_type;
    mat::dense2D<V> result(expr.num_rows(), expr.num_cols());
    using size_type = typename Expr::size_type;
    for (size_type r = 0; r < expr.num_rows(); ++r)
        for (size_type c = 0; c < expr.num_cols(); ++c)
            result(r, c) = expr(r, c);
    return result;
}

/// Materialize a vector expression into a concrete dense_vector
template <typename Expr>
    requires (Vector<Expr> && !Matrix<Expr> && traits::is_expression_v<Expr>)
auto evaluate(const Expr& expr) {
    using V = typename Expr::value_type;
    vec::dense_vector<V> result(expr.size());
    using size_type = typename Expr::size_type;
    for (size_type i = 0; i < expr.size(); ++i)
        result(i) = expr(i);
    return result;
}

/// Pass-through for concrete types (no-op)
template <typename T>
    requires (!traits::is_expression_v<T>)
const T& evaluate(const T& x) {
    return x;
}

} // namespace mtl
