#pragma once
// MTL5 -- Read-only view restricting access to a band [lower, upper] around diagonal
// banded_view(A, lower, upper): returns zero outside the band.
#include <cstddef>
#include <mtl/math/identity.hpp>
#include <mtl/traits/category.hpp>
#include <mtl/traits/ashape.hpp>

namespace mtl::mat::view {

/// Non-owning banded view: A(r,c) is zero unless c - r in [-lower, upper].
template <typename Matrix>
class banded_view {
public:
    using value_type      = typename Matrix::value_type;
    using size_type       = typename Matrix::size_type;
    using const_reference = value_type;  // returns by value (may be zero)
    using reference       = value_type;

    banded_view(const Matrix& m, std::ptrdiff_t lower, std::ptrdiff_t upper)
        : ref_(m), lower_(lower), upper_(upper) {}

    value_type operator()(size_type r, size_type c) const {
        auto diff = static_cast<std::ptrdiff_t>(c) - static_cast<std::ptrdiff_t>(r);
        if (diff >= -lower_ && diff <= upper_)
            return ref_(r, c);
        return math::zero<value_type>();
    }

    size_type num_rows() const { return ref_.num_rows(); }
    size_type num_cols() const { return ref_.num_cols(); }
    size_type size()     const { return ref_.size(); }

    std::ptrdiff_t lower_bandwidth() const { return lower_; }
    std::ptrdiff_t upper_bandwidth() const { return upper_; }

    const Matrix& base() const { return ref_; }

private:
    const Matrix& ref_;
    std::ptrdiff_t lower_, upper_;
};

} // namespace mtl::mat::view

namespace mtl::traits {
template <typename Matrix>
struct category<mat::view::banded_view<Matrix>> {
    using type = category_t<Matrix>;
};
} // namespace mtl::traits

namespace mtl::ashape {
template <typename Matrix>
struct ashape<::mtl::mat::view::banded_view<Matrix>> {
    using type = mat<typename Matrix::value_type>;
};
} // namespace mtl::ashape

namespace mtl {
/// Create a banded view of matrix A with given lower and upper bandwidth.
template <typename Matrix>
auto banded(const Matrix& A, std::ptrdiff_t lower, std::ptrdiff_t upper) {
    return mat::view::banded_view<Matrix>(A, lower, upper);
}
} // namespace mtl
