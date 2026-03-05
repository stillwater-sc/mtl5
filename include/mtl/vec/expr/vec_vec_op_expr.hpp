#pragma once
// MTL5 — Binary element-wise vector expression template
#include <cassert>
#include <cstddef>
#include <type_traits>
#include <mtl/traits/is_expression.hpp>
#include <mtl/traits/category.hpp>
#include <mtl/traits/ashape.hpp>
#include <mtl/tag/sparsity.hpp>

namespace mtl::vec::expr {

/// Lazy binary element-wise operation on two vector operands.
/// E1/E2 encode storage: const X& for lvalue operands, X for rvalue operands.
template <typename E1, typename E2, template <typename, typename> class SF>
class vec_vec_op_expr {
    using raw_e1 = std::remove_cvref_t<E1>;
    using raw_e2 = std::remove_cvref_t<E2>;

public:
    using value_type = typename SF<typename raw_e1::value_type, typename raw_e2::value_type>::result_type;
    using size_type  = std::size_t;

    template <typename A1, typename A2>
    vec_vec_op_expr(A1&& e1, A2&& e2)
        : e1_(std::forward<A1>(e1)), e2_(std::forward<A2>(e2))
    {
        assert(e1_.size() == e2_.size());
    }

    size_type size() const { return e1_.size(); }

    size_type num_rows() const { return size(); }
    size_type num_cols() const { return 1; }

    value_type operator()(size_type i) const {
        return SF<typename raw_e1::value_type, typename raw_e2::value_type>::apply(e1_(i), e2_(i));
    }

private:
    E1 e1_;
    E2 e2_;
};

} // namespace mtl::vec::expr

// ── Trait specializations ──────────────────────────────────────────────

namespace mtl::traits {

template <typename E1, typename E2, template <typename, typename> class SF>
struct is_expression<vec::expr::vec_vec_op_expr<E1, E2, SF>> : std::true_type {};

template <typename E1, typename E2, template <typename, typename> class SF>
struct category<vec::expr::vec_vec_op_expr<E1, E2, SF>> {
    using type = tag::dense;
};

} // namespace mtl::traits

namespace mtl::ashape {

template <typename E1, typename E2, template <typename, typename> class SF>
struct ashape<::mtl::vec::expr::vec_vec_op_expr<E1, E2, SF>> {
    using type = cvec<typename ::mtl::vec::expr::vec_vec_op_expr<E1, E2, SF>::value_type>;
};

} // namespace mtl::ashape
