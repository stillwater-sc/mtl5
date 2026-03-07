#pragma once
// MTL5 -- Scalar-matrix expression templates
#include <cstddef>
#include <type_traits>
#include <mtl/traits/is_expression.hpp>
#include <mtl/traits/category.hpp>
#include <mtl/traits/ashape.hpp>
#include <mtl/tag/sparsity.hpp>

namespace mtl::mat::expr {

/// Lazy scalar * matrix expression: SF::apply(scalar, matrix_element)
/// M encodes storage: const X& for lvalue, X for rvalue.
template <typename S, typename M, template <typename, typename> class SF>
class mat_scal_op_expr {
    using raw_m = std::remove_cvref_t<M>;

public:
    using value_type = typename SF<S, typename raw_m::value_type>::result_type;
    using size_type  = std::size_t;

    template <typename AM>
    mat_scal_op_expr(const S& s, AM&& m) : s_(s), m_(std::forward<AM>(m)) {}

    size_type num_rows() const { return m_.num_rows(); }
    size_type num_cols() const { return m_.num_cols(); }
    size_type size()     const { return num_rows() * num_cols(); }

    value_type operator()(size_type r, size_type c) const {
        return SF<S, typename raw_m::value_type>::apply(s_, m_(r, c));
    }

private:
    S s_;          // scalar by value
    M m_;          // matrix: const ref or value depending on lvalue/rvalue
};

/// Lazy matrix op scalar expression: SF::apply(matrix_element, scalar)
/// M encodes storage: const X& for lvalue, X for rvalue.
template <typename M, typename S, template <typename, typename> class SF>
class mat_rscal_op_expr {
    using raw_m = std::remove_cvref_t<M>;

public:
    using value_type = typename SF<typename raw_m::value_type, S>::result_type;
    using size_type  = std::size_t;

    template <typename AM>
    mat_rscal_op_expr(AM&& m, const S& s) : m_(std::forward<AM>(m)), s_(s) {}

    size_type num_rows() const { return m_.num_rows(); }
    size_type num_cols() const { return m_.num_cols(); }
    size_type size()     const { return num_rows() * num_cols(); }

    value_type operator()(size_type r, size_type c) const {
        return SF<typename raw_m::value_type, S>::apply(m_(r, c), s_);
    }

private:
    M m_;
    S s_;
};

} // namespace mtl::mat::expr

// -- Trait specializations ----------------------------------------------

namespace mtl::traits {

template <typename S, typename M, template <typename, typename> class SF>
struct is_expression<mat::expr::mat_scal_op_expr<S, M, SF>> : std::true_type {};

template <typename M, typename S, template <typename, typename> class SF>
struct is_expression<mat::expr::mat_rscal_op_expr<M, S, SF>> : std::true_type {};

template <typename S, typename M, template <typename, typename> class SF>
struct category<mat::expr::mat_scal_op_expr<S, M, SF>> {
    using type = tag::dense;
};

template <typename M, typename S, template <typename, typename> class SF>
struct category<mat::expr::mat_rscal_op_expr<M, S, SF>> {
    using type = tag::dense;
};

} // namespace mtl::traits

namespace mtl::ashape {

template <typename S, typename M, template <typename, typename> class SF>
struct ashape<::mtl::mat::expr::mat_scal_op_expr<S, M, SF>> {
    using type = mat<typename ::mtl::mat::expr::mat_scal_op_expr<S, M, SF>::value_type>;
};

template <typename M, typename S, template <typename, typename> class SF>
struct ashape<::mtl::mat::expr::mat_rscal_op_expr<M, S, SF>> {
    using type = mat<typename ::mtl::mat::expr::mat_rscal_op_expr<M, S, SF>::value_type>;
};

} // namespace mtl::ashape
