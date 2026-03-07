#pragma once
// MTL5 -- BiConjugate Gradient (BiCG) solver for non-symmetric systems
// Maintains shadow residuals and uses trans(A) for the adjoint multiply.
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/dot.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/operation/trans.hpp>

namespace mtl::itl {

/// BiConjugate Gradient method.
/// Solves A*x = b for non-symmetric A. Uses trans(A) internally.
/// M must provide solve() and adjoint_solve().
template <typename LinearOp, typename VecX, typename VecB,
          typename PC, typename Iter>
int bicg(const LinearOp& A, VecX& x, const VecB& b, const PC& M, Iter& iter) {
    using value_type = typename VecX::value_type;
    using size_type  = typename VecX::size_type;
    const size_type n = x.size();

    // Workspace: 8 vectors
    vec::dense_vector<value_type> r(n), r_tilde(n);
    vec::dense_vector<value_type> z(n), z_tilde(n);
    vec::dense_vector<value_type> p(n), p_tilde(n);
    vec::dense_vector<value_type> q(n), q_tilde(n);

    // r = b - A*x
    auto Ax = A * x;
    for (size_type i = 0; i < n; ++i) {
        r(i) = b(i) - Ax(i);
        r_tilde(i) = r(i);  // shadow residual = r initially
    }

    // z = M^{-1} r,  z_tilde = M^{-T} r_tilde
    M.solve(z, r);
    M.adjoint_solve(z_tilde, r_tilde);

    value_type rho = mtl::dot(z_tilde, z);
    value_type rho_1{};

    while (!iter.finished(r)) {
        ++iter;

        if (iter.first()) {
            for (size_type i = 0; i < n; ++i) {
                p(i) = z(i);
                p_tilde(i) = z_tilde(i);
            }
        } else {
            value_type beta = rho / rho_1;
            for (size_type i = 0; i < n; ++i) {
                p(i) = z(i) + beta * p(i);
                p_tilde(i) = z_tilde(i) + beta * p_tilde(i);
            }
        }

        // q = A * p
        auto Ap = A * p;
        for (size_type i = 0; i < n; ++i)
            q(i) = Ap(i);

        // q_tilde = A^T * p_tilde
        auto Atp = trans(A) * p_tilde;
        for (size_type i = 0; i < n; ++i)
            q_tilde(i) = Atp(i);

        value_type alpha = rho / mtl::dot(p_tilde, q);

        // x += alpha * p
        for (size_type i = 0; i < n; ++i)
            x(i) += alpha * p(i);

        // r -= alpha * q
        for (size_type i = 0; i < n; ++i)
            r(i) -= alpha * q(i);

        // r_tilde -= alpha * q_tilde
        for (size_type i = 0; i < n; ++i)
            r_tilde(i) -= alpha * q_tilde(i);

        // z = M^{-1} r, z_tilde = M^{-T} r_tilde
        M.solve(z, r);
        M.adjoint_solve(z_tilde, r_tilde);

        rho_1 = rho;
        rho = mtl::dot(z_tilde, z);

        if (rho == value_type(0)) {
            iter.fail(2, "bicg breakdown: rho == 0");
            return iter;
        }
    }

    return iter;
}

} // namespace mtl::itl
