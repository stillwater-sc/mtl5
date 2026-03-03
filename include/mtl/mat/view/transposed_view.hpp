#pragma once
// MTL5 — Lightweight non-owning transposed view of a matrix
#include <cstddef>
#include <mtl/traits/category.hpp>
#include <mtl/traits/ashape.hpp>
#include <mtl/traits/transposed_orientation.hpp>
#include <mtl/tag/sparsity.hpp>

namespace mtl::mat::view {

/// Non-owning transposed view: swaps row/col access
template <typename Matrix>
class transposed_view {
public:
    using value_type      = typename Matrix::value_type;
    using size_type       = typename Matrix::size_type;
    using const_reference = typename Matrix::const_reference;
    using reference       = const_reference;  // view is read-only

    explicit transposed_view(const Matrix& m) : ref_(m) {}

    const_reference operator()(size_type r, size_type c) const {
        return ref_(c, r);
    }

    size_type num_rows() const { return ref_.num_cols(); }
    size_type num_cols() const { return ref_.num_rows(); }
    size_type size()     const { return ref_.size(); }

private:
    const Matrix& ref_;
};

} // namespace mtl::mat::view

// ── Traits specializations ─────────────────────────────────────────────

namespace mtl::traits {

template <typename Matrix>
struct category<mat::view::transposed_view<Matrix>> {
    using type = tag::dense;
};

} // namespace mtl::traits

namespace mtl::ashape {

template <typename Matrix>
struct ashape<::mtl::mat::view::transposed_view<Matrix>> {
    using type = mat<typename Matrix::value_type>;
};

} // namespace mtl::ashape
