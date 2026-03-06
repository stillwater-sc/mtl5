#pragma once
// MTL5 — Random symmetric matrix with prescribed eigenvalues (factory, returns dense2D)
// Constructs A = Q * Lambda * Q^T where Q is a random orthogonal matrix.
// Result is symmetric by construction.
#include <cassert>
#include <cstddef>
#include <vector>
#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/operators.hpp>
#include <mtl/operation/trans.hpp>
#include <mtl/math/identity.hpp>
#include <mtl/generators/randorth.hpp>
#include <mtl/generators/randsvd.hpp>  // for detail::make_sigma

namespace mtl::generators {

/// Construct a symmetric matrix with prescribed eigenvalues.
/// A = Q * diag(eigenvalues) * Q^T where Q is random orthogonal.
/// eigenvalues.size() must equal n.
template <typename T = double>
auto randsym(std::size_t n, const std::vector<T>& eigenvalues) {
    assert(eigenvalues.size() == n);

    auto Q = randorth<T>(n);

    // Build Lambda as n x n diagonal
    mat::dense2D<T> Lambda(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            Lambda(i, j) = math::zero<T>();
    for (std::size_t i = 0; i < n; ++i)
        Lambda(i, i) = eigenvalues[i];

    // A = Q * Lambda * Q^T
    auto QL = Q * Lambda;
    mat::dense2D<T> result = QL * trans(Q);
    return result;
}

/// Construct a symmetric matrix with eigenvalues determined by condition number and mode.
/// Uses the same sigma distribution as randsvd (modes 1-5).
/// Eigenvalues are in [1/kappa, 1].
template <typename T = double>
auto randsym(std::size_t n, T kappa, int mode = 3) {
    auto eigenvalues = detail::make_sigma(n, kappa, mode);
    return randsym<T>(n, eigenvalues);
}

} // namespace mtl::generators
