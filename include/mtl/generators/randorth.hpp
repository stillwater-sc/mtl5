#pragma once
// MTL5 -- Random orthogonal matrix generator (factory, returns dense2D)
// Building block for randsvd, randsym, randspd.
// Constructs Q via QR factorization of a random matrix.
#include <cstddef>
#include <cstdint>
#include <random>
#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/random.hpp>
#include <mtl/operation/qr.hpp>

namespace mtl::generators {

/// Random orthogonal matrix of size n x n, from the random matrix drawn from
/// `gen` (operation/random.hpp, uniform on [-1, 1)).
/// Constructed via QR factorization of a random matrix: A = Q*R, return Q.
/// Q satisfies Q^T * Q = I to machine precision.
///
/// Q is taken from the QR with R's diagonal non-negative, which is unique.
/// qr_factor's own R has LAPACK's sign (#672), so without the normalization
/// each column of Q would carry a sign that depends on A's pivot entries, and
/// Q's distribution would differ from the one randorth gave before 5.22.0.
template <typename T = double>
auto randorth(std::size_t n, std::mt19937_64& gen) {
    auto A = random_matrix<T>(n, n, T(-1), T(1), gen);
    vec::dense_vector<T> tau;
    qr_factor(A, tau);
    auto Q = qr_extract_Q(A, tau);
    auto R = qr_extract_R(A);
    qr_normalize_signs(Q, R);
    return Q;
}

/// Random orthogonal matrix, reproducibly from `seed`: the random matrix is the
/// same with every compiler and standard library, so Q is too, up to the
/// rounding of the QR factorization.
template <typename T = double>
auto randorth(std::size_t n, std::uint64_t seed) {
    std::mt19937_64 gen(seed);
    return randorth<T>(n, gen);
}

/// Random orthogonal matrix, non-deterministically.
template <typename T = double>
auto randorth(std::size_t n) {
    auto gen = mtl::detail::random_engine();
    return randorth<T>(n, gen);
}

} // namespace mtl::generators
