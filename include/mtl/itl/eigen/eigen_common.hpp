#pragma once
// MTL5 -- Shared types/utilities for the iterative (Krylov) eigensolvers.
#include <type_traits>
#include <mtl/concepts/matrix.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/operation/mult.hpp>

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
///
/// Accumulator (#591): a non-void accumulator reaches the matvec when `A` is a
/// Matrix -- dense or compressed2D -- through `mult<Accumulator>`, as in the ITL
/// linear solvers (cg.hpp). A matrix-free operator only offers `A * x`, so it has
/// no accumulator to take, and the solvers' Accumulator then covers only their
/// dot products. `void` keeps `A * x` for every operator, byte-identical to the
/// helper before #591.
template <typename Accumulator = void, typename LinearOp, typename T>
vec::dense_vector<T> ev_matvec(const LinearOp& A, const vec::dense_vector<T>& x) {
    if constexpr (!std::is_void_v<Accumulator> && Matrix<LinearOp>) {
        vec::dense_vector<T> y(A.num_rows());
        mtl::mult<Accumulator>(A, x, y);
        return y;
    } else {
        auto w = A * x;
        const auto n = x.size();
        vec::dense_vector<T> y(n);
        for (typename vec::dense_vector<T>::size_type i = 0; i < n; ++i)
            y(i) = w(i);
        return y;
    }
}

} // namespace mtl::itl
