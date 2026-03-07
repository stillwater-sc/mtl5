#pragma once
// MTL5 -- Scalar-vector expression templates
#include <cstddef>
#include <type_traits>
#include <mtl/traits/is_expression.hpp>
#include <mtl/traits/category.hpp>
#include <mtl/traits/ashape.hpp>
#include <mtl/tag/sparsity.hpp>

namespace mtl::vec::expr {

/// Lazy scalar op vector expression: SF::apply(scalar, vec_element)
/// V encodes storage: const X& for lvalue, X for rvalue.
template <typename S, typename V, template <typename, typename> class SF>
class vec_scal_op_expr {
    using raw_v = std::remove_cvref_t<V>;

public:
    using value_type = typename SF<S, typename raw_v::value_type>::result_type;
    using size_type  = std::size_t;

    template <typename AV>
    vec_scal_op_expr(const S& s, AV&& v) : s_(s), v_(std::forward<AV>(v)) {}

    size_type size()     const { return v_.size(); }
    size_type num_rows() const { return size(); }
    size_type num_cols() const { return 1; }

    value_type operator()(size_type i) const {
        return SF<S, typename raw_v::value_type>::apply(s_, v_(i));
    }

private:
    S s_;
    V v_;
};

/// Lazy vector op scalar expression: SF::apply(vec_element, scalar)
/// V encodes storage: const X& for lvalue, X for rvalue.
template <typename V, typename S, template <typename, typename> class SF>
class vec_rscal_op_expr {
    using raw_v = std::remove_cvref_t<V>;

public:
    using value_type = typename SF<typename raw_v::value_type, S>::result_type;
    using size_type  = std::size_t;

    template <typename AV>
    vec_rscal_op_expr(AV&& v, const S& s) : v_(std::forward<AV>(v)), s_(s) {}

    size_type size()     const { return v_.size(); }
    size_type num_rows() const { return size(); }
    size_type num_cols() const { return 1; }

    value_type operator()(size_type i) const {
        return SF<typename raw_v::value_type, S>::apply(v_(i), s_);
    }

private:
    V v_;
    S s_;
};

} // namespace mtl::vec::expr

// -- Trait specializations ----------------------------------------------

namespace mtl::traits {

template <typename S, typename V, template <typename, typename> class SF>
struct is_expression<vec::expr::vec_scal_op_expr<S, V, SF>> : std::true_type {};

template <typename V, typename S, template <typename, typename> class SF>
struct is_expression<vec::expr::vec_rscal_op_expr<V, S, SF>> : std::true_type {};

template <typename S, typename V, template <typename, typename> class SF>
struct category<vec::expr::vec_scal_op_expr<S, V, SF>> {
    using type = tag::dense;
};

template <typename V, typename S, template <typename, typename> class SF>
struct category<vec::expr::vec_rscal_op_expr<V, S, SF>> {
    using type = tag::dense;
};

} // namespace mtl::traits

namespace mtl::ashape {

template <typename S, typename V, template <typename, typename> class SF>
struct ashape<::mtl::vec::expr::vec_scal_op_expr<S, V, SF>> {
    using type = cvec<typename ::mtl::vec::expr::vec_scal_op_expr<S, V, SF>::value_type>;
};

template <typename V, typename S, template <typename, typename> class SF>
struct ashape<::mtl::vec::expr::vec_rscal_op_expr<V, S, SF>> {
    using type = cvec<typename ::mtl::vec::expr::vec_rscal_op_expr<V, S, SF>::value_type>;
};

} // namespace mtl::ashape
