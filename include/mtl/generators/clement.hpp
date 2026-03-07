#pragma once
// MTL5 -- Clement matrix generator (factory, returns dense2D)
// Tridiagonal matrix with known integer eigenvalues (for even n).
#include <cmath>
#include <cstddef>
#include <mtl/mat/dense2D.hpp>

namespace mtl::generators {

/// Clement (tridiagonal) matrix: C(i,i+1) = sqrt(i*(n-1-i)) on superdiagonal,
/// C(i+1,i) = sqrt((i+1)*(n-2-i)) on subdiagonal (symmetric form).
/// For even n, eigenvalues are +/-(n-1), +/-(n-3), ..., +/-1.
template <typename T = double>
auto clement(std::size_t n) {
    mat::dense2D<T> C(n, n);
    // Initialize to zero
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            C(i, j) = T(0);

    // Symmetric Clement: sub- and super-diagonal entries are the same
    // C(i,i+1) = C(i+1,i) = sqrt((i+1)*(n-1-i))
    for (std::size_t i = 0; i + 1 < n; ++i) {
        T val = std::sqrt(T((i + 1) * (n - 1 - i)));
        C(i, i + 1) = val;
        C(i + 1, i) = val;
    }
    return C;
}

} // namespace mtl::generators
