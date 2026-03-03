#pragma once
// MTL5 — Return transposed view of a matrix
#include <mtl/concepts/matrix.hpp>
#include <mtl/mat/view/transposed_view.hpp>

namespace mtl {

/// Returns a lightweight transposed view of matrix m
template <Matrix M>
auto trans(const M& m) {
    return mat::view::transposed_view<M>(m);
}

} // namespace mtl
