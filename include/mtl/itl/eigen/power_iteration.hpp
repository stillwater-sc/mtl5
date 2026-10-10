#pragma once
// MTL5 -- Power iteration for the dominant eigenpair of a linear operator.
// Matrix-free: only needs A * x (the LinearOperator concept), so it applies to
// dense2D, compressed2D, and user-supplied matrix-free operators alike.
#include <cmath>
#include <type_traits>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/dot.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/itl/eigen/eigen_common.hpp>
#include <mtl/math/accumulator_traits.hpp>

namespace mtl::itl {

/// A single computed eigenpair.
template <typename T>
struct eigenpair {
    T value;                       ///< eigenvalue estimate (Rayleigh quotient)
    vec::dense_vector<T> vector;   ///< unit-norm eigenvector estimate
    int iterations = 0;
    bool converged = false;
};

/// Power iteration for the eigenvalue of largest magnitude (assumed real and
/// dominant, as for symmetric / SPD operators or any operator with a real
/// dominant eigenvalue). Returns the Rayleigh-quotient eigenvalue and unit
/// eigenvector. Converges when the Ritz residual ||A v - lambda v|| <= tol.
///
/// `A` is any LinearOperator (A * v yields a vector); `v0` is the starting
/// vector (need not be normalized). O(1) vectors of storage; one matvec/iter.
template <typename Accumulator = void, typename LinearOp, typename T>
eigenpair<T> power_iteration(const LinearOp& A, vec::dense_vector<T> v0,
                             int max_iter = 1000, std::type_identity_t<T> tol = T(1e-10)) {
    // void (the default) accumulates in T, as the former `Accumulator = T` did.
    using Acc = std::conditional_t<std::is_void_v<Accumulator>, T, Accumulator>;
    using AT  = mtl::math::accumulator_traits<Acc, T>;
    using std::abs;
    using size_type = typename vec::dense_vector<T>::size_type;
    const size_type n = v0.size();

    eigenpair<T> result;
    result.vector = v0;
    if (n == 0) return result;

    // Normalize the start vector.
    T nv = mtl::two_norm(v0);
    if (nv == T(0)) { v0(0) = T(1); nv = T(1); }
    for (size_type i = 0; i < n; ++i) v0(i) /= nv;

    T lambda = T(0);
    int it = 0;
    for (; it < max_iter; ++it) {
        vec::dense_vector<T> w = ev_matvec(A, v0);

        // Rayleigh quotient (v is unit norm): lambda = v^T A v.
        lambda = mtl::dot<Acc, T>(v0, w);

        // Ritz residual r = A v - lambda v.
        Acc res_acc{};
        AT::clear(res_acc);
        for (size_type i = 0; i < n; ++i) {
            T ri = w(i) - lambda * v0(i);
            AT::add_product(res_acc, ri, ri);
        }
        T res = std::sqrt(AT::template value<T>(res_acc));

        // Normalize w to become the next iterate.
        T nw = mtl::two_norm(w);
        if (nw == T(0)) break;               // A v = 0: v spans a null vector
        for (size_type i = 0; i < n; ++i) v0(i) = w(i) / nw;

        if (res <= tol * (abs(lambda) + tol)) {
            result.converged = true;
            ++it;
            break;
        }
    }

    result.value = lambda;
    result.vector = v0;
    result.iterations = it;
    return result;
}

} // namespace mtl::itl
