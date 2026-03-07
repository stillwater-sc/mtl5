#pragma once
// MTL5 -- Random orthogonal matrix generator (factory, returns dense2D)
// Building block for randsvd, randsym, randspd.
// Constructs Q via QR factorization of a random matrix.
#include <cstddef>
#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/random.hpp>
#include <mtl/operation/qr.hpp>

namespace mtl::generators {

/// Random orthogonal matrix of size n x n.
/// Constructed via QR factorization of a random matrix: A = Q*R, return Q.
/// Q satisfies Q^T * Q = I to machine precision.
template <typename T = double>
auto randorth(std::size_t n) {
    auto A = random_matrix<T>(n, n, T(-1), T(1));
    vec::dense_vector<T> tau;
    qr_factor(A, tau);
    return qr_extract_Q(A, tau);
}

} // namespace mtl::generators
