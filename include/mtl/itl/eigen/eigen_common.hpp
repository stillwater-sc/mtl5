#pragma once
// MTL5 -- Shared types/utilities for the iterative (Krylov) eigensolvers.
#include <mtl/vec/dense_vector.hpp>
#include <mtl/mat/dense2D.hpp>

namespace mtl::itl {

/// Which end of the spectrum a Krylov eigensolver should return.
/// Algebraic selectors apply to real (symmetric) spectra; magnitude selectors
/// apply to any spectrum (including complex Arnoldi Ritz values).
enum class eigen_which {
    largest_magnitude,
    smallest_magnitude,
    largest_algebraic,
    smallest_algebraic
};

/// A set of computed Ritz pairs (approximate eigenpairs from a Krylov subspace).
/// `Value` is real for symmetric (Lanczos) solvers and std::complex for general
/// (Arnoldi) solvers.
template <typename Value>
struct ritz_pairs {
    vec::dense_vector<Value> values;   ///< k wanted Ritz values
    mat::dense2D<Value> vectors;       ///< n x k, column i is the Ritz vector for values(i)
    int subspace = 0;                  ///< Krylov subspace dimension actually built
    bool converged = false;            ///< all wanted pairs met the residual tolerance
};

/// Apply a LinearOperator and materialize the result as a dense_vector.
template <typename LinearOp, typename T>
vec::dense_vector<T> ev_matvec(const LinearOp& A, const vec::dense_vector<T>& x) {
    auto w = A * x;
    const auto n = x.size();
    vec::dense_vector<T> y(n);
    for (typename vec::dense_vector<T>::size_type i = 0; i < n; ++i)
        y(i) = w(i);
    return y;
}

} // namespace mtl::itl
