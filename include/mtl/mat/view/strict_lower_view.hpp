#pragma once
// MTL5 — Read-only strict lower triangular view (excludes diagonal)
// strict_lower_view(A): A_view(r,c) = A(r,c) if r > c, else 0
// Also provides MATLAB-compatible tril(A, k) with diagonal offset.
#include <cstddef>
#include <mtl/math/identity.hpp>
#include <mtl/traits/category.hpp>
#include <mtl/traits/ashape.hpp>

namespace mtl::mat::view {

/// Non-owning strict lower triangular view: returns zero on and above diagonal.
template <typename Matrix>
class strict_lower_view {
public:
    using value_type      = typename Matrix::value_type;
    using size_type       = typename Matrix::size_type;
    using const_reference = value_type;
    using reference       = value_type;

    /// @param m underlying matrix
    /// @param offset diagonal offset (0 = exclude main diagonal, -1 = also exclude first subdiag, etc.)
    explicit strict_lower_view(const Matrix& m, std::ptrdiff_t offset = 0)
        : ref_(m), offset_(offset) {}

    value_type operator()(size_type r, size_type c) const {
        auto diff = static_cast<std::ptrdiff_t>(c) - static_cast<std::ptrdiff_t>(r);
        return (diff < offset_) ? ref_(r, c) : math::zero<value_type>();
    }

    size_type num_rows() const { return ref_.num_rows(); }
    size_type num_cols() const { return ref_.num_cols(); }
    size_type size()     const { return ref_.size(); }

    const Matrix& base() const { return ref_; }

private:
    const Matrix& ref_;
    std::ptrdiff_t offset_;
};

} // namespace mtl::mat::view

namespace mtl::traits {
template <typename Matrix>
struct category<mat::view::strict_lower_view<Matrix>> {
    using type = category_t<Matrix>;
};
} // namespace mtl::traits

namespace mtl::ashape {
template <typename Matrix>
struct ashape<::mtl::mat::view::strict_lower_view<Matrix>> {
    using type = mat<typename Matrix::value_type>;
};
} // namespace mtl::ashape

namespace mtl {
/// Create a strict lower triangular view (excludes diagonal).
template <typename Matrix>
auto strict_lower(const Matrix& A) {
    return mat::view::strict_lower_view<Matrix>(A);
}

/// MATLAB-compatible tril(A, k): elements on diag k and below.
/// tril(A, 0) = lower(A), tril(A, -1) = strict_lower(A), etc.
template <typename Matrix>
auto tril(const Matrix& A, std::ptrdiff_t k = 0) {
    // Return elements where c - r <= k, i.e., exclude where c - r > k
    return mat::view::strict_lower_view<Matrix>(A, k + 1);
}
} // namespace mtl
