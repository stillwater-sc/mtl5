#pragma once
// MTL5 -- Tolerances for structural predicates on matrices (symmetric?
// Hermitian? real diagonal?).
//
// These exist because an EXACT structural test is the wrong test. A matrix
// assembled in floating point -- from an FEM pass, or from a blocked B^H*B
// whose (i,j) and (j,i) sums accumulate in different orders -- carries its
// structure only to rounding. An exact test passes on every matrix a test
// author types as a literal and fails on the ones a user computes: during the
// #352 work a ONE-ULP perturbation of a Hermitian matrix defeated an exact
// guard and produced an answer wrong in the first significant digit under
// info == 0, which is the failure the guard existed to prevent.
//
// Shared by ldlt.hpp and cholesky.hpp so the two use the same threshold.

#include <cmath>
#include <cstddef>
#include <limits>

#include <mtl/concepts/magnitude.hpp>
#include <mtl/functor/scalar/real.hpp>
#include <mtl/functor/scalar/imag.hpp>

namespace mtl::detail {

/// LAPACK's CABS1: |Re z| + |Im z|. Within a factor of sqrt(2) of |z| and free
/// of the sqrt/hypot that abs() would cost -- which is what a structural noise
/// threshold wants, since the threshold itself is only order-of-magnitude.
/// The prepass is O(n^2) against an O(n^3/3) factorization, so avoiding a hypot
/// per entry matters at small n.
template <typename T>
constexpr magnitude_t<T> cabs1(const T& z) {
    using std::abs;
    return abs(functor::scalar::real<T>::apply(z))
         + abs(functor::scalar::imag<T>::apply(z));
}

/// Threshold separating "this matrix has this structure" from "rounding noise",
/// scaled by the size of the problem and the magnitude of the data.
///
/// For a type without a std::numeric_limits specialization epsilon() is 0, so
/// this degrades to an exact test rather than to something arbitrary -- which
/// matters for the custom number types this library targets.
template <typename Mag>
constexpr Mag structure_tol(std::size_t n, Mag scale) {
    return static_cast<Mag>(n) * std::numeric_limits<Mag>::epsilon() * scale;
}

}  // namespace mtl::detail
