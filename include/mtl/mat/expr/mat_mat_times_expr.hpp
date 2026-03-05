#pragma once
// MTL5 — Matrix-matrix multiplication expression template
// Stores operands (by value or reference); operator()(r,c) computes inner product.
// For performance-critical code, prefer mult(A, B, C) into pre-allocated target.
#include <cassert>
#include <cstddef>
#include <type_traits>
#include <mtl/traits/is_expression.hpp>
#include <mtl/traits/category.hpp>
#include <mtl/traits/ashape.hpp>
#include <mtl/tag/sparsity.hpp>
#include <mtl/math/identity.hpp>

namespace mtl::mat::expr {

/// Lazy matrix-matrix multiply expression.
/// E1/E2 encode storage: const X& for lvalue, X for rvalue.
template <typename E1, typename E2>
class mat_mat_times_expr {
    using raw_e1 = std::remove_cvref_t<E1>;
    using raw_e2 = std::remove_cvref_t<E2>;

public:
    using value_type = std::common_type_t<typename raw_e1::value_type, typename raw_e2::value_type>;
    using size_type  = std::size_t;

    template <typename A1, typename A2>
    mat_mat_times_expr(A1&& e1, A2&& e2)
        : e1_(std::forward<A1>(e1)), e2_(std::forward<A2>(e2))
    {
        assert(e1_.num_cols() == e2_.num_rows());
    }

    size_type num_rows() const { return e1_.num_rows(); }
    size_type num_cols() const { return e2_.num_cols(); }
    size_type size()     const { return num_rows() * num_cols(); }

    value_type operator()(size_type r, size_type c) const {
        auto acc = math::zero<value_type>();
        for (size_type k = 0; k < e1_.num_cols(); ++k)
            acc += static_cast<value_type>(e1_(r, k)) * static_cast<value_type>(e2_(k, c));
        return acc;
    }

private:
    E1 e1_;
    E2 e2_;
};

} // namespace mtl::mat::expr

// ── Trait specializations ──────────────────────────────────────────────

namespace mtl::traits {

template <typename E1, typename E2>
struct is_expression<mat::expr::mat_mat_times_expr<E1, E2>> : std::true_type {};

template <typename E1, typename E2>
struct category<mat::expr::mat_mat_times_expr<E1, E2>> {
    using type = tag::dense;
};

} // namespace mtl::traits

namespace mtl::ashape {

template <typename E1, typename E2>
struct ashape<::mtl::mat::expr::mat_mat_times_expr<E1, E2>> {
    using type = mat<std::common_type_t<
        typename std::remove_cvref_t<E1>::value_type,
        typename std::remove_cvref_t<E2>::value_type>>;
};

} // namespace mtl::ashape
