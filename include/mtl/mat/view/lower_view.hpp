#pragma once
// MTL5 — Read-only lower triangular view (includes diagonal)
// lower_view(A): A_view(r,c) = A(r,c) if r >= c, else 0
#include <cstddef>
#include <mtl/math/identity.hpp>
#include <mtl/traits/category.hpp>
#include <mtl/traits/ashape.hpp>

namespace mtl::mat::view {

/// Non-owning lower triangular view: returns zero above the diagonal.
template <typename Matrix>
class lower_view {
public:
    using value_type      = typename Matrix::value_type;
    using size_type       = typename Matrix::size_type;
    using const_reference = value_type;
    using reference       = value_type;

    explicit lower_view(const Matrix& m) : ref_(m) {}

    value_type operator()(size_type r, size_type c) const {
        return (r >= c) ? ref_(r, c) : math::zero<value_type>();
    }

    size_type num_rows() const { return ref_.num_rows(); }
    size_type num_cols() const { return ref_.num_cols(); }
    size_type size()     const { return ref_.size(); }

    const Matrix& base() const { return ref_; }

private:
    const Matrix& ref_;
};

} // namespace mtl::mat::view

namespace mtl::traits {
template <typename Matrix>
struct category<mat::view::lower_view<Matrix>> {
    using type = category_t<Matrix>;
};
} // namespace mtl::traits

namespace mtl::ashape {
template <typename Matrix>
struct ashape<::mtl::mat::view::lower_view<Matrix>> {
    using type = mat<typename Matrix::value_type>;
};
} // namespace mtl::ashape

namespace mtl {
/// Create a lower triangular view (includes diagonal).
template <typename Matrix>
auto lower(const Matrix& A) {
    return mat::view::lower_view<Matrix>(A);
}
} // namespace mtl
