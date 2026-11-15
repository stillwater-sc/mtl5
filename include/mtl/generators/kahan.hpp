#pragma once
// MTL5 -- Kahan matrix generator (factory, returns dense2D)
// Upper triangular, ill-conditioned. Tests QR factorization.
#include <cmath>
#include <cstddef>
#include <limits>
#include <mtl/mat/dense2D.hpp>

namespace mtl::generators {

/// Kahan matrix, as MATLAB's gallery('kahan', n, theta, pert):
///
///   K = diag(1, s, s^2, ..., s^(n-1)) * (I - c * triu(ones(n), 1))
///       + pert * eps * diag(n, n-1, ..., 1),        s = sin(theta), c = cos(theta)
///
/// so K(i,i) = s^i + pert*eps*(n-i), K(i,j) = -c*s^i for i < j, and 0 below the
/// diagonal (0-based i, j; eps is std::numeric_limits<T>::epsilon()). Its
/// condition number grows exponentially in n: the inverse of the unscaled
/// factor has entries c*(1+c)^(j-i-1) above the diagonal. The perturbation
/// (MATLAB's default pert = 25) keeps column-pivoted QR from swapping columns
/// that tie only through rounding; pert = 0 gives the unperturbed matrix.
template <typename T = double>
auto kahan(std::size_t n, T theta = T(1.2), T pert = T(25)) {
    using std::sin;
    using std::cos;
    mat::dense2D<T> K(n, n);
    const T s = sin(theta);
    const T c = cos(theta);
    const T eps = std::numeric_limits<T>::epsilon();

    T si = T(1);   // s^i
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            if (j < i)
                K(i, j) = T(0);
            else if (j == i)
                K(i, j) = si + pert * eps * T(n - i);
            else
                K(i, j) = -c * si;
        }
        si *= s;
    }
    return K;
}

} // namespace mtl::generators
