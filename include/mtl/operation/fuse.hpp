#pragma once
// MTL5 -- Fused evaluation: evaluate expressions directly into pre-allocated targets
#include <cassert>
#include <mtl/traits/is_expression.hpp>
#include <mtl/concepts/matrix.hpp>
#include <mtl/concepts/vector.hpp>
#include <mtl/detail/same_object.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/expr/expr_alias.hpp>

namespace mtl {

namespace detail {

/// Whether writing expr into dest element by element could read an element of
/// dest other than the one being written (issue 887): the test dense2D's
/// operator= makes (#824). A target whose element storage can't be seen is
/// assumed to be read.
template <typename Dest, typename Expr>
bool fused_reads_dest_elsewhere(const Dest& dest, const Expr& expr) {
    if constexpr (contiguous_storage<Dest>) {
        const auto [lo, hi] = storage_bytes(dest);
        return alias_reads_target_elsewhere(expr, alias_target{&dest, lo, hi});
    } else {
        return true;
    }
}

/// expr evaluated into a dense matrix of Dest's value type.
template <typename Dest, typename Expr>
mat::dense2D<typename Dest::value_type> fused_evaluate(const Expr& expr) {
    using V = typename Dest::value_type;
    mat::dense2D<V> tmp(expr.num_rows(), expr.num_cols());
    for (typename Expr::size_type r = 0; r < expr.num_rows(); ++r)
        for (typename Expr::size_type c = 0; c < expr.num_cols(); ++c)
            tmp(r, c) = static_cast<V>(expr(r, c));
    return tmp;
}

} // namespace detail

/// Evaluate a matrix expression into a pre-allocated target: dest = expr.
///
/// In place, unless the expression reads dest anywhere but at the element
/// being written -- A = trans(A), A = A * B -- or dest's storage can't be
/// seen: then through a temporary. Without the test, fused_assign(A,
/// 1.0 * trans(A)) on [[1,2],[3,4]] gave [[1,3],[3,4]], and change_dim ran
/// before expr was read, so a reallocation could leave expr dangling (issue
/// 887).
template <typename Dest, Matrix Expr>
    requires traits::is_expression_v<Expr>
void fused_assign(Dest& dest, const Expr& expr) {
    using size_type = typename Expr::size_type;
    using dsize = typename Dest::size_type;
    const bool same_shape = static_cast<std::size_t>(dest.num_rows()) == static_cast<std::size_t>(expr.num_rows()) &&
                            static_cast<std::size_t>(dest.num_cols()) == static_cast<std::size_t>(expr.num_cols());
    if (!same_shape || detail::fused_reads_dest_elsewhere(dest, expr)) {
        const auto tmp = detail::fused_evaluate<Dest>(expr);   // read expr before resizing dest
        dest.change_dim(static_cast<dsize>(tmp.num_rows()), static_cast<dsize>(tmp.num_cols()));
        for (std::size_t r = 0; r < tmp.num_rows(); ++r)
            for (std::size_t c = 0; c < tmp.num_cols(); ++c)
                dest(static_cast<dsize>(r), static_cast<dsize>(c)) = tmp(r, c);
        return;
    }
    for (size_type r = 0; r < expr.num_rows(); ++r)
        for (size_type c = 0; c < expr.num_cols(); ++c)
            dest(r, c) = static_cast<typename Dest::value_type>(expr(r, c));
}

/// Evaluate a vector expression into a pre-allocated target: dest = expr.
/// The vector expressions are element-wise (a product with a matrix is
/// evaluated eagerly, not an expression here), so element i reads only
/// element i of every operand, dest included: in place is always safe.
template <typename Dest, Vector Expr>
    requires (traits::is_expression_v<Expr> && !Matrix<Expr>)
void fused_assign(Dest& dest, const Expr& expr) {
    using size_type = typename Expr::size_type;
    dest.change_dim(static_cast<typename Dest::size_type>(expr.size()));
    for (size_type i = 0; i < expr.size(); ++i)
        dest(i) = static_cast<typename Dest::value_type>(expr(i));
}

/// Evaluate a matrix expression and add to target: dest += expr
/// In place unless expr reads dest elsewhere (see fused_assign): then the
/// sum is formed from a temporary of expr. fused_plus_assign(C, trans(C) *
/// 1.0) on [[1,2],[3,4]] gave [[2,5],[8,8]] (issue 887).
template <typename Dest, Matrix Expr>
    requires traits::is_expression_v<Expr>
void fused_plus_assign(Dest& dest, const Expr& expr) {
    using size_type = typename Expr::size_type;
    using dsize = typename Dest::size_type;
    assert(dest.num_rows() == expr.num_rows() && dest.num_cols() == expr.num_cols());
    if (detail::fused_reads_dest_elsewhere(dest, expr)) {
        const auto tmp = detail::fused_evaluate<Dest>(expr);
        for (std::size_t r = 0; r < tmp.num_rows(); ++r)
            for (std::size_t c = 0; c < tmp.num_cols(); ++c)
                dest(static_cast<dsize>(r), static_cast<dsize>(c)) += tmp(r, c);
        return;
    }
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
