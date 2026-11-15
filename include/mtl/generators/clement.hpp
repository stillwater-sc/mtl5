#pragma once
// MTL5 -- Clement matrix generator (factory, returns dense2D)
// Tridiagonal matrix with known integer eigenvalues (for even n).
#include <cmath>
#include <cstddef>
#include <mtl/mat/dense2D.hpp>

namespace mtl::generators {

/// Clement (tridiagonal) matrix, symmetric form: zero diagonal and
/// C(i,i+1) = C(i+1,i) = sqrt((i+1)*(n-1-i)) for i = 0..n-2.
/// Its eigenvalues are n-1, n-3, ..., -(n-1): +/-(n-1), ..., +/-1 for even n,
/// and +/-(n-1), ..., +/-2, 0 for odd n.
template <typename T = double>
auto clement(std::size_t n) {
    mat::dense2D<T> C(n, n);
    // Initialize to zero
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            C(i, j) = T(0);

    for (std::size_t i = 0; i + 1 < n; ++i) {
        using std::sqrt;
        T val = sqrt(T((i + 1) * (n - 1 - i)));
        C(i, i + 1) = val;
        C(i + 1, i) = val;
    }
    return C;
}

} // namespace mtl::generators
