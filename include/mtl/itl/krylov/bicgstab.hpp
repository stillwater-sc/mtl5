#pragma once
// MTL5 -- BiCGSTAB (Bi-Conjugate Gradient Stabilized) solver
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/dot.hpp>
#include <mtl/operation/norms.hpp>

namespace mtl::itl {

/// BiCGSTAB method for non-symmetric systems.
/// Solves A*x = b with preconditioner M and iteration controller iter.
template <typename LinearOp, typename VecX, typename VecB,
          typename PC, typename Iter>
int bicgstab(const LinearOp& A, VecX& x, const VecB& b, const PC& M, Iter& iter) {
    using value_type = typename VecX::value_type;
    using size_type  = typename VecX::size_type;
    const size_type n = x.size();

    // Workspace vectors
    vec::dense_vector<value_type> r(n), r_star(n), p(n), phat(n);
    vec::dense_vector<value_type> v(n), s(n), shat(n), t(n);

    // r = b - A*x
    auto Ax = A * x;
    for (size_type i = 0; i < n; ++i) {
        r(i) = b(i) - Ax(i);
        r_star(i) = r(i);
    }

    value_type rho_1 = value_type(1);
    value_type alpha = value_type(1);
    value_type omega = value_type(1);

    // v = 0, p = 0
    for (size_type i = 0; i < n; ++i) {
        v(i) = value_type(0);
        p(i) = value_type(0);
    }

    while (!iter.finished(r)) {
        ++iter;

        value_type rho = mtl::dot(r_star, r);

        if (rho == value_type(0)) {
            iter.fail(2, "bicgstab breakdown: rho == 0");
            return iter;
        }

        value_type beta = (rho / rho_1) * (alpha / omega);

        // p = r + beta * (p - omega * v)
        for (size_type i = 0; i < n; ++i)
            p(i) = r(i) + beta * (p(i) - omega * v(i));

        // phat = M^{-1} p
        M.solve(phat, p);

        // v = A * phat
        auto Aphat = A * phat;
        for (size_type i = 0; i < n; ++i)
            v(i) = Aphat(i);

        // alpha = rho / dot(r_star, v)
        alpha = rho / mtl::dot(r_star, v);

        // s = r - alpha * v
        for (size_type i = 0; i < n; ++i)
            s(i) = r(i) - alpha * v(i);

        // Check convergence of s
        if (iter.finished(mtl::two_norm(s))) {
            // x += alpha * phat
            for (size_type i = 0; i < n; ++i)
                x(i) += alpha * phat(i);
            return iter;
        }

        // shat = M^{-1} s
        M.solve(shat, s);

        // t = A * shat
        auto Ashat = A * shat;
        for (size_type i = 0; i < n; ++i)
            t(i) = Ashat(i);

        // omega = dot(t, s) / dot(t, t)
        omega = mtl::dot(t, s) / mtl::dot(t, t);

        // x += alpha * phat + omega * shat
        for (size_type i = 0; i < n; ++i)
            x(i) += alpha * phat(i) + omega * shat(i);

        // r = s - omega * t
        for (size_type i = 0; i < n; ++i)
            r(i) = s(i) - omega * t(i);

        rho_1 = rho;
    }

    return iter;
}

} // namespace mtl::itl
