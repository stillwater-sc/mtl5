#pragma once
// MTL5 -- Frank matrix generator (factory, returns dense2D)
// Upper Hessenberg matrix with known eigenvalues.
#include <cstddef>
#include <mtl/mat/dense2D.hpp>

namespace mtl::generators {

/// Frank matrix: upper Hessenberg with F(i,j) = n+1-max(i+1,j+1)
/// for j >= i-1, zero below the sub-diagonal.
/// Has known, real, positive eigenvalues.
template <typename T = double>
auto frank(std::size_t n) {
    mat::dense2D<T> F(n, n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            if (j + 1 >= i) {  // j >= i-1 (using +1 to avoid underflow)
                std::size_t m = std::max(i + 1, j + 1);
                F(i, j) = T(n + 1 - m);
            } else {
                F(i, j) = T(0);
            }
        }
    }
    return F;
}

} // namespace mtl::generators
