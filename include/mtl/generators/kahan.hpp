#pragma once
// MTL5 — Kahan matrix generator (factory, returns dense2D)
// Upper triangular, ill-conditioned. Tests QR factorization.
#include <cmath>
#include <cstddef>
#include <mtl/mat/dense2D.hpp>

namespace mtl::generators {

/// Kahan matrix: upper triangular, ill-conditioned.
/// K(i,j) = 0 if i>j, -s*c^(i) if i<j, s^(i)*zeta if i==j (scaled).
/// Standard formulation: diag(s^0,s^1,...,s^{n-1}) * (I - c * upper_ones).
template <typename T = double>
auto kahan(std::size_t n, T theta = T(1.2), T zeta = T(25)) {
    mat::dense2D<T> K(n, n);
    T s = std::sin(theta);
    T c = std::cos(theta);

    for (std::size_t i = 0; i < n; ++i) {
        // Compute s^i
        T si = T(1);
        for (std::size_t k = 0; k < i; ++k)
            si *= s;

        for (std::size_t j = 0; j < n; ++j) {
            if (j < i) {
                K(i, j) = T(0);
            } else if (j == i) {
                K(i, j) = si * zeta;
            } else {
                K(i, j) = -c * si;
            }
        }
    }
    return K;
}

} // namespace mtl::generators
