#pragma once
// MTL5 -- Givens rotation utilities for GMRES Hessenberg QR factorization
#include <cmath>
#include <mtl/math/pow2_scaling.hpp>

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
    using value_type = typename Matrix::value_type;

    // Scaled by an exact power of two, so r neither overflows nor underflows
    // where a*a + b*b would (#671).
    const auto rot = math::givens_rotation<value_type>(H(k, k), H(k + 1, k));
    c(k) = rot.c;
    s(k) = rot.s;

    H(k, k)     = rot.r;
    H(k + 1, k) = value_type(0);

    // Apply to RHS vector g
    auto tmp   = c(k) * g(k) + s(k) * g(k + 1);
    g(k + 1)   = -s(k) * g(k) + c(k) * g(k + 1);
    g(k)       = tmp;
}

} // namespace mtl
