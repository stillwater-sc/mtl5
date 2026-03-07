#pragma once
// MTL5 -- Unary negation expression for matrices
#include <cstddef>
#include <type_traits>
#include <mtl/traits/is_expression.hpp>
#include <mtl/traits/category.hpp>
#include <mtl/traits/ashape.hpp>
#include <mtl/tag/sparsity.hpp>

namespace mtl::mat::expr {

/// Lazy unary negation of a matrix expression.
/// E encodes storage: const X& for lvalue, X for rvalue.
template <typename E>
class mat_negate_expr {
    using raw_e = std::remove_cvref_t<E>;

public:
    using value_type = typename raw_e::value_type;
    using size_type  = std::size_t;

    template <typename AE>
    explicit mat_negate_expr(AE&& e) : e_(std::forward<AE>(e)) {}

    size_type num_rows() const { return e_.num_rows(); }
    size_type num_cols() const { return e_.num_cols(); }
    size_type size()     const { return num_rows() * num_cols(); }

    value_type operator()(size_type r, size_type c) const {
        return -e_(r, c);
    }

private:
    E e_;
};

} // namespace mtl::mat::expr

// -- Trait specializations ----------------------------------------------

namespace mtl::traits {

template <typename E>
struct is_expression<mat::expr::mat_negate_expr<E>> : std::true_type {};

template <typename E>
struct category<mat::expr::mat_negate_expr<E>> {
    using type = tag::dense;
};

} // namespace mtl::traits

namespace mtl::ashape {

template <typename E>
struct ashape<::mtl::mat::expr::mat_negate_expr<E>> {
    using type = mat<typename std::remove_cvref_t<E>::value_type>;
};

} // namespace mtl::ashape
