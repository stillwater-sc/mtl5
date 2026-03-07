#pragma once
// MTL5 -- Operator overloads for matrix types (expression template returns)
#include <type_traits>
#include <cassert>
#include <mtl/concepts/matrix.hpp>
#include <mtl/concepts/vector.hpp>
#include <mtl/traits/is_expression.hpp>
#include <mtl/detail/expr_storage.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/view/transposed_view.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/math/identity.hpp>
#include <mtl/functor/scalar/plus.hpp>
#include <mtl/functor/scalar/minus.hpp>
#include <mtl/functor/scalar/times.hpp>
#include <mtl/functor/scalar/divide.hpp>
#include <mtl/mat/expr/mat_mat_op_expr.hpp>
#include <mtl/mat/expr/mat_scal_op_expr.hpp>
#include <mtl/mat/expr/mat_negate_expr.hpp>
#include <mtl/mat/expr/mat_mat_times_expr.hpp>

namespace mtl::mat {

// -- Binary arithmetic (lazy) -------------------------------------------
// Forwarding references: lvalue operands stored by const ref, rvalue by value.

template <typename M1, typename M2>
    requires (Matrix<std::remove_cvref_t<M1>> && Matrix<std::remove_cvref_t<M2>>)
auto operator+(M1&& a, M2&& b) {
    using S1 = detail::expr_store_t<M1>;
    using S2 = detail::expr_store_t<M2>;
    return expr::mat_mat_op_expr<S1, S2, functor::scalar::plus>(
        std::forward<M1>(a), std::forward<M2>(b));
}

template <typename M1, typename M2>
    requires (Matrix<std::remove_cvref_t<M1>> && Matrix<std::remove_cvref_t<M2>>)
auto operator-(M1&& a, M2&& b) {
    using S1 = detail::expr_store_t<M1>;
    using S2 = detail::expr_store_t<M2>;
    return expr::mat_mat_op_expr<S1, S2, functor::scalar::minus>(
        std::forward<M1>(a), std::forward<M2>(b));
}

// -- Unary negation (lazy) ----------------------------------------------

template <typename M>
    requires Matrix<std::remove_cvref_t<M>>
auto operator-(M&& m) {
    using SM = detail::expr_store_t<M>;
    return expr::mat_negate_expr<SM>(std::forward<M>(m));
}

// -- Scalar-matrix multiply (lazy) --------------------------------------

template <typename S, typename M>
    requires (std::is_arithmetic_v<S> && Matrix<std::remove_cvref_t<M>>)
auto operator*(const S& alpha, M&& m) {
    using SM = detail::expr_store_t<M>;
    return expr::mat_scal_op_expr<S, SM, functor::scalar::times>(
        alpha, std::forward<M>(m));
}

template <typename M, typename S>
    requires (Matrix<std::remove_cvref_t<M>> && std::is_arithmetic_v<S>)
auto operator*(M&& m, const S& alpha) {
    using SM = detail::expr_store_t<M>;
    return expr::mat_scal_op_expr<S, SM, functor::scalar::times>(
        alpha, std::forward<M>(m));
}

template <typename M, typename S>
    requires (Matrix<std::remove_cvref_t<M>> && std::is_arithmetic_v<S>)
auto operator/(M&& m, const S& alpha) {
    using SM = detail::expr_store_t<M>;
    return expr::mat_rscal_op_expr<SM, S, functor::scalar::divide>(
        std::forward<M>(m), alpha);
}

// -- Matrix-vector multiply (stays eager) -------------------------------
// No expression template: immediately evaluates to dense_vector.

template <Matrix M, Vector V>
    requires (!Matrix<V>)
auto operator*(const M& A, const V& x) {
    assert(A.num_cols() == x.size());
    using result_t = std::common_type_t<typename M::value_type, typename V::value_type>;
    vec::dense_vector<result_t> y(A.num_rows());
    for (typename M::size_type r = 0; r < A.num_rows(); ++r) {
        auto acc = math::zero<result_t>();
        for (typename M::size_type c = 0; c < A.num_cols(); ++c) {
            acc += A(r, c) * x(c);
        }
        y(r) = acc;
    }
    return y;
}

// -- Matrix-matrix multiply (eager) -------------------------------------
// Stays eager to avoid aliasing issues (e.g., auto C = A*B; then modify A)
// and O(n^3)-per-element lazy recomputation. For lazy matmul, use
// mat_mat_times_expr directly.

template <Matrix M1, Matrix M2>
auto operator*(const M1& A, const M2& B) {
    assert(A.num_cols() == B.num_rows());
    using result_t = std::common_type_t<typename M1::value_type, typename M2::value_type>;
    dense2D<result_t> C(A.num_rows(), B.num_cols());
    for (typename M1::size_type r = 0; r < A.num_rows(); ++r) {
        for (typename M2::size_type c = 0; c < B.num_cols(); ++c) {
            auto acc = math::zero<result_t>();
            for (typename M1::size_type k = 0; k < A.num_cols(); ++k) {
                acc += A(r, k) * B(k, c);
            }
            C(r, c) = acc;
        }
    }
    return C;
}

// -- Sparse (CRS) matvec: compressed2D * dense_vector (stays eager) -----

template <typename V, typename P, typename VV, typename VP>
auto operator*(const compressed2D<V, P>& A,
               const vec::dense_vector<VV, VP>& x) {
    assert(A.num_cols() == x.size());
    using result_t = std::common_type_t<V, VV>;
    using size_type = typename compressed2D<V, P>::size_type;
    vec::dense_vector<result_t> y(A.num_rows(), math::zero<result_t>());
    const auto& starts  = A.ref_major();
    const auto& indices = A.ref_minor();
    const auto& data    = A.ref_data();
    for (size_type r = 0; r < A.num_rows(); ++r) {
        auto acc = math::zero<result_t>();
        for (size_type k = starts[r]; k < starts[r + 1]; ++k) {
            acc += static_cast<result_t>(data[k]) * static_cast<result_t>(x(indices[k]));
        }
        y(r) = acc;
    }
    return y;
}

// -- Transposed sparse matvec (stays eager) -----------------------------

template <typename V, typename P, typename VV, typename VP>
auto operator*(const view::transposed_view<compressed2D<V, P>>& At,
               const vec::dense_vector<VV, VP>& x) {
    const auto& A = At.base();
    assert(A.num_rows() == x.size());
    using result_t = std::common_type_t<V, VV>;
    using size_type = typename compressed2D<V, P>::size_type;
    vec::dense_vector<result_t> y(A.num_cols(), math::zero<result_t>());
    const auto& starts  = A.ref_major();
    const auto& indices = A.ref_minor();
    const auto& data    = A.ref_data();
    for (size_type r = 0; r < A.num_rows(); ++r) {
        for (size_type k = starts[r]; k < starts[r + 1]; ++k) {
            y(indices[k]) += static_cast<result_t>(data[k]) * static_cast<result_t>(x(r));
        }
    }
    return y;
}

} // namespace mtl::mat
