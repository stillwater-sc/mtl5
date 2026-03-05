#pragma once
// MTL5 — Binary element-wise matrix expression template
#include <cassert>
#include <cstddef>
#include <type_traits>
#include <mtl/traits/is_expression.hpp>
#include <mtl/traits/category.hpp>
#include <mtl/traits/ashape.hpp>
#include <mtl/tag/sparsity.hpp>

namespace mtl::mat::expr {

/// Lazy binary element-wise operation on two matrix operands.
/// E1/E2 encode storage: const X& for lvalue operands, X for rvalue operands.
/// SF is a scalar functor with static apply(a, b).
template <typename E1, typename E2, template <typename, typename> class SF>
class mat_mat_op_expr {
    using raw_e1 = std::remove_cvref_t<E1>;
    using raw_e2 = std::remove_cvref_t<E2>;

public:
    using value_type = typename SF<typename raw_e1::value_type, typename raw_e2::value_type>::result_type;
    using size_type  = std::size_t;

    template <typename A1, typename A2>
    mat_mat_op_expr(A1&& e1, A2&& e2)
        : e1_(std::forward<A1>(e1)), e2_(std::forward<A2>(e2))
    {
        assert(e1_.num_rows() == e2_.num_rows() && e1_.num_cols() == e2_.num_cols());
    }

    size_type num_rows() const { return e1_.num_rows(); }
    size_type num_cols() const { return e1_.num_cols(); }
    size_type size()     const { return num_rows() * num_cols(); }

    value_type operator()(size_type r, size_type c) const {
        return SF<typename raw_e1::value_type, typename raw_e2::value_type>::apply(e1_(r, c), e2_(r, c));
    }

private:
    E1 e1_;
    E2 e2_;
};

} // namespace mtl::mat::expr

// ── Trait specializations ──────────────────────────────────────────────

namespace mtl::traits {

template <typename E1, typename E2, template <typename, typename> class SF>
struct is_expression<mat::expr::mat_mat_op_expr<E1, E2, SF>> : std::true_type {};

template <typename E1, typename E2, template <typename, typename> class SF>
struct category<mat::expr::mat_mat_op_expr<E1, E2, SF>> {
    using type = tag::dense;
};

} // namespace mtl::traits

namespace mtl::ashape {

template <typename E1, typename E2, template <typename, typename> class SF>
struct ashape<::mtl::mat::expr::mat_mat_op_expr<E1, E2, SF>> {
    using type = mat<typename ::mtl::mat::expr::mat_mat_op_expr<E1, E2, SF>::value_type>;
};

} // namespace mtl::ashape
