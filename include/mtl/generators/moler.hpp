#pragma once
// MTL5 — Moler matrix generator (factory, returns dense2D)
// SPD matrix M = L*L^T where L is unit lower triangular with alpha below diagonal.
#include <algorithm>
#include <cstddef>
#include <mtl/mat/dense2D.hpp>

namespace mtl::generators {

/// Moler matrix: M = L*L^T where L(i,i) = 1, L(i,j) = alpha for i > j.
/// Default alpha = -1. SPD with eigenvalues that cluster near zero.
/// M(i,i) = i*alpha^2 + 1, M(i,j) = min(i,j)*alpha^2 + alpha for i != j.
template <typename T = double>
auto moler(std::size_t n, T alpha = T(-1)) {
    mat::dense2D<T> M(n, n);
    T a2 = alpha * alpha;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            if (i == j) {
                M(i, j) = T(i) * a2 + T(1);
            } else {
                M(i, j) = T(std::min(i, j)) * a2 + alpha;
            }
        }
    }
    return M;
}

} // namespace mtl::generators
