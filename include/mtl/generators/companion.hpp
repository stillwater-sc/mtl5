#pragma once
// MTL5 — Companion matrix generator (factory, returns dense2D)
// Eigenvalues are roots of the given polynomial.
#include <cstddef>
#include <vector>
#include <mtl/mat/dense2D.hpp>

namespace mtl::generators {

/// Companion matrix for polynomial p(x) = x^n + c[n-1]*x^{n-1} + ... + c[0].
/// The coefficients vector contains [c_0, c_1, ..., c_{n-1}] (low to high).
/// Eigenvalues of the companion matrix are the roots of p(x).
template <typename T = double>
auto companion(const std::vector<T>& coeffs) {
    std::size_t n = coeffs.size();
    mat::dense2D<T> C(n, n);

    // Initialize to zero
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            C(i, j) = T(0);

    // Sub-diagonal of 1s
    for (std::size_t i = 1; i < n; ++i)
        C(i, i - 1) = T(1);

    // Last column: -c_i (negated coefficients)
    for (std::size_t i = 0; i < n; ++i)
        C(i, n - 1) = -coeffs[i];

    return C;
}

} // namespace mtl::generators
