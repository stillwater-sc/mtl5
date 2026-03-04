#pragma once
// MTL5 — QMR (Quasi-Minimal Residual) solver
// Based on Barrett et al. "Templates for the Solution of Linear Systems" Algorithm 7.3.
// Requires trans(A). M must provide solve() and adjoint_solve().
#include <cmath>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/dot.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/operation/trans.hpp>

namespace mtl::itl {

/// QMR solver for non-symmetric systems A*x = b.
template <typename LinearOp, typename VecX, typename VecB,
          typename PC, typename Iter>
int qmr(const LinearOp& A, VecX& x, const VecB& b, const PC& M, Iter& iter) {
    using value_type = typename VecX::value_type;
    using size_type  = typename VecX::size_type;
    using std::sqrt;
    using std::abs;
    const size_type n = x.size();

    vec::dense_vector<value_type> r(n), v_tilde(n), w_tilde(n);
    vec::dense_vector<value_type> v(n), w(n), y(n), z(n);
    vec::dense_vector<value_type> p(n), q(n), d(n), s(n), p_tilde(n);

    // r = b - A*x
    auto Ax = A * x;
    for (size_type i = 0; i < n; ++i) {
        r(i) = b(i) - Ax(i);
        v_tilde(i) = r(i);
        w_tilde(i) = r(i);
        d(i) = value_type(0);
        s(i) = value_type(0);
    }

    // y = M^{-1} v_tilde;  z = M^{-T} w_tilde
    M.solve(y, v_tilde);
    M.adjoint_solve(z, w_tilde);

    value_type rho = mtl::two_norm(y);
    value_type xi = mtl::two_norm(z);
    value_type gamma = value_type(1);
    value_type eta = value_type(-1);
    value_type epsilon = value_type(1);
    value_type theta = value_type(0);

    while (!iter.finished(r)) {
        ++iter;

        if (rho == value_type(0) || xi == value_type(0)) {
            iter.fail(2, "qmr breakdown: rho or xi == 0");
            return iter;
        }

        // v = v_tilde / rho;  y = y / rho
        for (size_type i = 0; i < n; ++i) {
            v(i) = v_tilde(i) / rho;
            y(i) = y(i) / rho;
        }
        // w = w_tilde / xi;  z = z / xi
        for (size_type i = 0; i < n; ++i) {
            w(i) = w_tilde(i) / xi;
            z(i) = z(i) / xi;
        }

        value_type delta = mtl::dot(z, y);
        if (delta == value_type(0)) {
            iter.fail(2, "qmr breakdown: delta == 0");
            return iter;
        }

        // For left-only preconditioning: y_tilde = y, z_tilde = z
        // p = y_tilde - (xi*delta/epsilon)*p;  q = z_tilde - (rho*delta/epsilon)*q
        if (iter.first()) {
            for (size_type i = 0; i < n; ++i) {
                p(i) = y(i);
                q(i) = z(i);
            }
        } else {
            value_type pcoeff = xi * delta / epsilon;
            value_type qcoeff = rho * delta / epsilon;
            for (size_type i = 0; i < n; ++i) {
                p(i) = y(i) - pcoeff * p(i);
                q(i) = z(i) - qcoeff * q(i);
            }
        }

        // p_tilde = A * p
        auto Ap = A * p;
        for (size_type i = 0; i < n; ++i)
            p_tilde(i) = Ap(i);

        epsilon = mtl::dot(q, p_tilde);
        if (epsilon == value_type(0)) {
            iter.fail(2, "qmr breakdown: epsilon == 0");
            return iter;
        }

        value_type beta = epsilon / delta;

        // v_tilde = p_tilde - beta * v
        for (size_type i = 0; i < n; ++i)
            v_tilde(i) = p_tilde(i) - beta * v(i);

        // y = M^{-1} v_tilde
        M.solve(y, v_tilde);

        value_type rho_new = mtl::two_norm(y);

        // w_tilde = A^T * q - beta * w
        auto Atq = trans(A) * q;
        for (size_type i = 0; i < n; ++i)
            w_tilde(i) = Atq(i) - beta * w(i);

        // z = M^{-T} w_tilde
        M.adjoint_solve(z, w_tilde);

        value_type xi_new = mtl::two_norm(z);

        value_type theta_new = rho_new / (gamma * abs(beta));
        value_type gamma_new = value_type(1) / sqrt(value_type(1) + theta_new * theta_new);
        value_type eta_new = -eta * rho * gamma_new * gamma_new / (beta * gamma * gamma);

        // d = eta_new * p + (theta * gamma_new)^2 * d
        // s = eta_new * p_tilde + (theta * gamma_new)^2 * s
        value_type ratio = (theta * gamma_new) * (theta * gamma_new);
        if (iter.first()) {
            for (size_type i = 0; i < n; ++i) {
                d(i) = eta_new * p(i);
                s(i) = eta_new * p_tilde(i);
            }
        } else {
            for (size_type i = 0; i < n; ++i) {
                d(i) = eta_new * p(i) + ratio * d(i);
                s(i) = eta_new * p_tilde(i) + ratio * s(i);
            }
        }

        // x += d;  r -= s
        for (size_type i = 0; i < n; ++i) {
            x(i) += d(i);
            r(i) -= s(i);
        }

        rho = rho_new;
        xi = xi_new;
        theta = theta_new;
        gamma = gamma_new;
        eta = eta_new;
    }

    return iter;
}

} // namespace mtl::itl
