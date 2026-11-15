#pragma once
// MTL5 -- Forsythe matrix generator (factory, returns dense2D)
// Perturbed Jordan block. Tests eigensolvers near defective matrices.
#include <cstddef>
#include <mtl/mat/dense2D.hpp>

namespace mtl::generators {

/// Forsythe matrix: n x n Jordan block with eigenvalue lambda,
/// plus alpha in position (n-1, 0).
/// F(i,i) = lambda, F(i,i+1) = 1, F(n-1,0) = alpha.
/// When alpha = 0, it's a standard Jordan block. For n >= 2 its eigenvalues
/// are lambda + alpha^(1/n) * (the n-th roots of unity). For n = 1 the corner is
/// the diagonal, so the matrix is the 1 x 1 Jordan block [lambda] and alpha is
/// not used (MATLAB's gallery('forsythe', 1) gives [alpha] instead).
template <typename T = double>
auto forsythe(std::size_t n, T alpha = T(1e-10), T lambda = T(0)) {
    mat::dense2D<T> F(n, n);

    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            F(i, j) = T(0);

    for (std::size_t i = 0; i < n; ++i)
        F(i, i) = lambda;

    for (std::size_t i = 0; i + 1 < n; ++i)
        F(i, i + 1) = T(1);

    // Perturbation in the corner, which for n = 1 would overwrite lambda
    if (n > 1)
        F(n - 1, 0) = alpha;

    return F;
}

} // namespace mtl::generators
