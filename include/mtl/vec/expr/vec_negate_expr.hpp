#pragma once
// MTL5 — Unary negation expression for vectors
#include <cstddef>
#include <type_traits>
#include <mtl/traits/is_expression.hpp>
#include <mtl/traits/category.hpp>
#include <mtl/traits/ashape.hpp>
#include <mtl/tag/sparsity.hpp>

namespace mtl::vec::expr {

/// Lazy unary negation of a vector expression.
/// E encodes storage: const X& for lvalue, X for rvalue.
template <typename E>
class vec_negate_expr {
    using raw_e = std::remove_cvref_t<E>;

public:
    using value_type = typename raw_e::value_type;
    using size_type  = std::size_t;

    template <typename AE>
    explicit vec_negate_expr(AE&& e) : e_(std::forward<AE>(e)) {}

    size_type size()     const { return e_.size(); }
    size_type num_rows() const { return size(); }
    size_type num_cols() const { return 1; }

    value_type operator()(size_type i) const {
        return -e_(i);
    }

private:
    E e_;
};

} // namespace mtl::vec::expr

// ── Trait specializations ──────────────────────────────────────────────

namespace mtl::traits {

template <typename E>
struct is_expression<vec::expr::vec_negate_expr<E>> : std::true_type {};

template <typename E>
struct category<vec::expr::vec_negate_expr<E>> {
    using type = tag::dense;
};

} // namespace mtl::traits

namespace mtl::ashape {

template <typename E>
struct ashape<::mtl::vec::expr::vec_negate_expr<E>> {
    using type = cvec<typename std::remove_cvref_t<E>::value_type>;
};

} // namespace mtl::ashape
