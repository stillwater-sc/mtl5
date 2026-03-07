#pragma once
// MTL5 -- Hermitian view: reads upper triangle and mirrors with conjugation.
// For real matrices, behaves as a symmetric view.
#include <cstddef>
#include <mtl/functor/scalar/conj.hpp>
#include <mtl/traits/category.hpp>
#include <mtl/traits/ashape.hpp>

namespace mtl::mat::view {

/// Non-owning hermitian view: A(r,c) = conj(A(c,r)) when r > c.
/// Assumes the upper triangle of the underlying matrix is stored.
template <typename Matrix>
class hermitian_view {
public:
    using value_type      = typename Matrix::value_type;
    using size_type       = typename Matrix::size_type;
    using const_reference = value_type;
    using reference       = value_type;

    explicit hermitian_view(const Matrix& m) : ref_(m) {}

    value_type operator()(size_type r, size_type c) const {
        if (r <= c)
            return ref_(r, c);
        return functor::scalar::conj<value_type>::apply(ref_(c, r));
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
struct category<mat::view::hermitian_view<Matrix>> {
    using type = category_t<Matrix>;
};
} // namespace mtl::traits

namespace mtl::ashape {
template <typename Matrix>
struct ashape<::mtl::mat::view::hermitian_view<Matrix>> {
    using type = mat<typename Matrix::value_type>;
};
} // namespace mtl::ashape

namespace mtl {
/// Create a hermitian view of matrix A (upper triangle stored).
template <typename Matrix>
auto hermitian(const Matrix& A) {
    return mat::view::hermitian_view<Matrix>(A);
}
} // namespace mtl
