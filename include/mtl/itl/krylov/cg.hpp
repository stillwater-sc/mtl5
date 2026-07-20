#pragma once
// MTL5 -- Conjugate Gradient solver for symmetric positive definite systems
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/dot.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/operation/mult.hpp>

namespace mtl::itl {

/// Preconditioned Conjugate Gradient method.
/// Solves A*x = b where A is symmetric positive definite.
/// M is a preconditioner satisfying M.solve(x, b).
/// iter is a convergence controller (basic_iteration or derived).
///
/// Accumulator (optional): accumulation type for the two dot products and
/// the matrix-vector product (see math/accumulator_traits.hpp, #158).
/// Defaults to void, matching dot()/mult()'s own default -- unspecified
/// behavior is unchanged.
///
/// Returns the iteration object (convertible to int error code).
template <typename LinearOp, typename VecX, typename VecB,
          typename PC, typename Iter,
          typename Accumulator = void>
int cg(const LinearOp& A, VecX& x, const VecB& b, const PC& M, Iter& iter) {
    using value_type = typename VecX::value_type;
    using size_type  = typename VecX::size_type;
    const size_type n = x.size();

    // Workspace vectors
    vec::dense_vector<value_type> r(n), z(n), p(n), q(n);

    // r = b - A*x
    vec::dense_vector<value_type> Ax(n);
    mtl::mult<Accumulator>(A, x, Ax);
    for (size_type i = 0; i < n; ++i)
        r(i) = b(i) - Ax(i);

    // z = M^{-1} r
    M.solve(z, r);

    // rho = dot(r, z)
    value_type rho = mtl::dot<Accumulator, value_type>(r, z);
    value_type rho_1{};

    while (!iter.finished(r)) {
        ++iter;

        if (iter.first()) {
            // p = z
            for (size_type i = 0; i < n; ++i)
                p(i) = z(i);
        } else {
            // p = z + (rho / rho_1) * p
            value_type beta = rho / rho_1;
            for (size_type i = 0; i < n; ++i)
                p(i) = z(i) + beta * p(i);
        }

        // q = A * p
        mtl::mult<Accumulator>(A, p, q);

        // alpha = rho / dot(p, q)
        value_type alpha = rho / mtl::dot<Accumulator, value_type>(p, q);

        // x += alpha * p
        for (size_type i = 0; i < n; ++i)
            x(i) += alpha * p(i);

        // Save rho for next iteration
        rho_1 = rho;

        // r -= alpha * q
        for (size_type i = 0; i < n; ++i)
            r(i) -= alpha * q(i);

        // z = M^{-1} r
        M.solve(z, r);

        // rho = dot(r, z)
        rho = mtl::dot<Accumulator, value_type>(r, z);
    }

    return iter;
}

} // namespace mtl::itl
