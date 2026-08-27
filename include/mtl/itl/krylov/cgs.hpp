#pragma once
// MTL5 -- CGS (Conjugate Gradient Squared) solver
// Ported from MTL4. For non-symmetric systems.
#include <mtl/concepts/vector.hpp>   // FieldVector (#503)
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/dot.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/operation/mult.hpp>

namespace mtl::itl {

/// Conjugate Gradient Squared method for non-symmetric systems.
/// Solves A*x = b with preconditioner M and iteration controller iter.
///
/// Accumulator (optional): accumulation type for the dot products and the
/// matrix-vector products (see math/accumulator_traits.hpp, #158).
/// Defaults to void, matching dot()/mult()'s own default -- unspecified
/// behavior is unchanged.
template <typename LinearOp, typename VecX, typename VecB,
          typename PC, typename Iter,
          typename Accumulator = void>
    requires FieldVector<VecX>
int cgs(const LinearOp& A, VecX& x, const VecB& b, const PC& M, Iter& iter) {
    using value_type = typename VecX::value_type;
    using size_type  = typename VecX::size_type;
    const size_type n = x.size();

    // Workspace vectors
    vec::dense_vector<value_type> r(n), rtilde(n), p(n), phat(n);
    vec::dense_vector<value_type> u(n), uhat(n), q(n), vhat(n), qhat(n);

    // r = b - A*x
    vec::dense_vector<value_type> Ax(n);
    mtl::mult<Accumulator>(A, x, Ax);
    for (size_type i = 0; i < n; ++i) {
        r(i) = b(i) - Ax(i);
        rtilde(i) = r(i);
    }

    value_type rho_1 = value_type(0);
    value_type rho_2 = value_type(0);
    value_type alpha = value_type(0);
    value_type beta  = value_type(0);

    while (!iter.finished(r)) {
        ++iter;
        rho_1 = mtl::dot<Accumulator, value_type>(rtilde, r);

        if (rho_1 == value_type(0)) {
            iter.fail(2, "cgs breakdown: rho == 0");
            return iter;
        }

        if (iter.first()) {
            for (size_type i = 0; i < n; ++i) {
                u(i) = r(i);
                p(i) = r(i);
            }
        } else {
            beta = rho_1 / rho_2;
            // u = r + beta * q
            for (size_type i = 0; i < n; ++i)
                u(i) = r(i) + beta * q(i);
            // p = u + beta * (q + beta * p)
            for (size_type i = 0; i < n; ++i)
                p(i) = u(i) + beta * (q(i) + beta * p(i));
        }

        // phat = M^{-1} p
        M.solve(phat, p);

        // vhat = A * phat
        mtl::mult<Accumulator>(A, phat, vhat);

        // alpha = rho_1 / dot(rtilde, vhat)
        value_type rtv = mtl::dot<Accumulator, value_type>(rtilde, vhat);
        if (rtv == value_type(0)) {
            iter.fail(3, "cgs breakdown: dot(rtilde, vhat) == 0");
            return iter;
        }
        alpha = rho_1 / rtv;

        // q = u - alpha * vhat
        for (size_type i = 0; i < n; ++i)
            q(i) = u(i) - alpha * vhat(i);

        // u += q (accumulate u + q)
        for (size_type i = 0; i < n; ++i)
            u(i) += q(i);

        // uhat = M^{-1} u
        M.solve(uhat, u);

        // x += alpha * uhat
        for (size_type i = 0; i < n; ++i)
            x(i) += alpha * uhat(i);

        // qhat = A * uhat
        mtl::mult<Accumulator>(A, uhat, qhat);

        // r -= alpha * qhat
        for (size_type i = 0; i < n; ++i)
            r(i) -= alpha * qhat(i);

        rho_2 = rho_1;
    }

    return iter;
}

} // namespace mtl::itl
