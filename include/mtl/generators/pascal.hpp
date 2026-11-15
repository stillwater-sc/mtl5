#pragma once
// MTL5 -- Pascal matrix generator (factory, returns dense2D)
// Symmetric matrix of binomial coefficients. det(P) = 1.
#include <cstddef>
#include <mtl/mat/dense2D.hpp>

namespace mtl::generators {

/// Pascal matrix: P(i,j) = C(i+j, i) = (i+j)! / (i! * j!).
/// Symmetric positive definite with determinant 1.
/// The entries grow like 4^n / sqrt(pi n): they are exact in T only up to the
/// largest integer T represents exactly (2^24 in float: n <= 14; 2^53 in
/// double: n <= 29), and the largest, P(n-1,n-1), overflows to inf in float
/// for n >= 67 and in double for n >= 516.
template <typename T = double>
auto pascal(std::size_t n) {
    mat::dense2D<T> P(n, n);
    // Use recurrence: P(i,0) = P(0,j) = 1
    //                 P(i,j) = P(i-1,j) + P(i,j-1)
    for (std::size_t i = 0; i < n; ++i)
        P(i, 0) = T(1);
    for (std::size_t j = 0; j < n; ++j)
        P(0, j) = T(1);
    for (std::size_t i = 1; i < n; ++i)
        for (std::size_t j = 1; j < n; ++j)
            P(i, j) = P(i - 1, j) + P(i, j - 1);
    return P;
}

} // namespace mtl::generators
