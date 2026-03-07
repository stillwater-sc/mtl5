#pragma once
// MTL5 -- Map view: reindex matrix access via row and column permutation vectors.
// map_view(A, row_map, col_map): A_view(i,j) = A(row_map[i], col_map[j])
#include <cstddef>
#include <vector>
#include <mtl/traits/category.hpp>
#include <mtl/traits/ashape.hpp>

namespace mtl::mat::view {

/// Non-owning view with index remapping via permutation vectors.
template <typename Matrix>
class map_view {
public:
    using value_type      = typename Matrix::value_type;
    using size_type       = typename Matrix::size_type;
    using const_reference = value_type;
    using reference       = value_type;

    map_view(const Matrix& m,
             const std::vector<size_type>& row_map,
             const std::vector<size_type>& col_map)
        : ref_(m), row_map_(row_map), col_map_(col_map) {}

    value_type operator()(size_type r, size_type c) const {
        return ref_(row_map_[r], col_map_[c]);
    }

    size_type num_rows() const { return row_map_.size(); }
    size_type num_cols() const { return col_map_.size(); }
    size_type size()     const { return num_rows() * num_cols(); }

    const Matrix& base() const { return ref_; }

private:
    const Matrix& ref_;
    std::vector<size_type> row_map_;
    std::vector<size_type> col_map_;
};

} // namespace mtl::mat::view

namespace mtl::traits {
template <typename Matrix>
struct category<mat::view::map_view<Matrix>> {
    using type = category_t<Matrix>;
};
} // namespace mtl::traits

namespace mtl::ashape {
template <typename Matrix>
struct ashape<::mtl::mat::view::map_view<Matrix>> {
    using type = mat<typename Matrix::value_type>;
};
} // namespace mtl::ashape

namespace mtl {
/// Create a map view with row and column remapping.
template <typename Matrix>
auto mapped(const Matrix& A,
            const std::vector<typename Matrix::size_type>& row_map,
            const std::vector<typename Matrix::size_type>& col_map) {
    return mat::view::map_view<Matrix>(A, row_map, col_map);
}
} // namespace mtl
