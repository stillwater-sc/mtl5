#pragma once
// MTL5 -- Return transposed view of a matrix
#include <mtl/concepts/matrix.hpp>
#include <mtl/mat/view/transposed_view.hpp>

namespace mtl {

/// Returns a lightweight transposed view of matrix m
template <Matrix M>
auto trans(const M& m) {
    return mat::view::transposed_view<M>(m);
}

/// Deleted for an rvalue (#695): the view refers to its matrix, so a view of a
/// temporary dangles once the full expression ends. Name the matrix first.
template <Matrix M>
void trans(const M&&) = delete;

} // namespace mtl
