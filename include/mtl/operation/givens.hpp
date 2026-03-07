#pragma once
// MTL5 -- Givens rotation utilities for GMRES Hessenberg QR factorization
#include <cmath>

namespace mtl {

/// Apply a previously stored Givens rotation to column k of Hessenberg H
/// at rows i, i+1.
/// Transforms H(i,k) and H(i+1,k) in-place.
template <typename Matrix, typename Vector>
void apply_stored_rotation(Matrix& H, const Vector& c, const Vector& s,
                           typename Matrix::size_type i,
                           typename Matrix::size_type k) {
    auto tmp      = c(i) * H(i, k) + s(i) * H(i + 1, k);
    H(i + 1, k)   = -s(i) * H(i, k) + c(i) * H(i + 1, k);
    H(i, k)       = tmp;
}

/// Compute new Givens rotation for column k of Hessenberg H,
/// apply it to H and RHS vector g, and store the rotation in c, s.
template <typename Matrix, typename VecG, typename VecC, typename VecS>
void apply_givens_rotation(Matrix& H, VecG& g, VecC& c, VecS& s,
                           typename Matrix::size_type k) {
    using std::sqrt;
    using std::abs;
    using value_type = typename Matrix::value_type;

    auto a = H(k, k);
    auto b = H(k + 1, k);
    auto r = sqrt(a * a + b * b);

    if (r == value_type(0)) {
        c(k) = value_type(1);
        s(k) = value_type(0);
    } else {
        c(k) = a / r;
        s(k) = b / r;
    }

    H(k, k)     = r;
    H(k + 1, k) = value_type(0);

    // Apply to RHS vector g
    auto tmp   = c(k) * g(k) + s(k) * g(k + 1);
    g(k + 1)   = -s(k) * g(k) + c(k) * g(k + 1);
    g(k)       = tmp;
}

} // namespace mtl
