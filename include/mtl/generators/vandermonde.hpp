#pragma once
// MTL5 — Vandermonde matrix generator (factory, returns dense2D)
// V(i,j) = x_i^j. Ill-conditioned for non-Chebyshev nodes.
#include <cstddef>
#include <vector>
#include <mtl/mat/dense2D.hpp>

namespace mtl::generators {

/// Vandermonde matrix: V(i,j) = nodes[i]^j.
/// The resulting matrix is n x n where n = nodes.size().
/// Ill-conditioned for uniformly-spaced nodes.
template <typename T = double>
auto vandermonde(const std::vector<T>& nodes) {
    std::size_t n = nodes.size();
    mat::dense2D<T> V(n, n);

    for (std::size_t i = 0; i < n; ++i) {
        T xpow = T(1);
        for (std::size_t j = 0; j < n; ++j) {
            V(i, j) = xpow;
            xpow *= nodes[i];
        }
    }
    return V;
}

} // namespace mtl::generators
