#pragma once
// MTL5 -- MINRES (Minimum Residual) solver for symmetric indefinite systems
// Lanczos-based 3-term recurrence with Givens rotations.
// Guarantees monotonically decreasing residual norm for symmetric A.
// Reference: Paige & Saunders (1975), Saad Algorithm 6.13.
#include <cmath>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/dot.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/math/identity.hpp>

namespace mtl::itl {

/// MINRES method for symmetric (possibly indefinite) systems.
/// Solves A*x = b with preconditioner M and iteration controller iter.
/// A must be symmetric.
template <typename LinearOp, typename VecX, typename VecB,
          typename PC, typename Iter>
int minres(const LinearOp& A, VecX& x, const VecB& b, const PC& M, Iter& iter) {
    using value_type = typename VecX::value_type;
    using size_type  = typename VecX::size_type;
    using std::sqrt;
    using std::abs;

    const size_type n = x.size();
    const auto zero = math::zero<value_type>();

    // Workspace
    vec::dense_vector<value_type> v_old(n), v(n), v_new(n);
    vec::dense_vector<value_type> d(n), d_old(n), d_older(n);
    vec::dense_vector<value_type> z(n), w(n);

    // Initial residual: r = b - A*x
    auto Ax = A * x;
    for (size_type i = 0; i < n; ++i)
        v(i) = b(i) - Ax(i);

    // For preconditioned MINRES, beta = sqrt(r^T M^{-1} r)
    // but for simplicity with right preconditioning, we use beta = ||r||
    value_type beta = mtl::two_norm(v);

    if (beta == zero) return iter;

    // v = r / beta  (normalized Lanczos vector)
    for (size_type i = 0; i < n; ++i) {
        v(i) /= beta;
        v_old(i) = zero;
        d(i) = zero;
        d_old(i) = zero;
        d_older(i) = zero;
    }

    // Givens rotation state: G_{j-2} and G_{j-1}
    value_type cs_old = value_type(1);   // c_{j-2}
    value_type sn_old = zero;            // s_{j-2}
    value_type cs     = value_type(1);   // c_{j-1}
    value_type sn     = zero;            // s_{j-1}

    // Transformed RHS: |eta| tracks the residual norm
    value_type eta = beta;

    while (!iter.finished(abs(eta))) {
        ++iter;

        // Lanczos step: w = A * M^{-1} * v (right preconditioned)
        M.solve(z, v);
        auto Az = A * z;
        for (size_type i = 0; i < n; ++i)
            w(i) = Az(i);

        // alpha = v^T * w
        value_type alpha = mtl::dot(v, w);

        // v_new = w - alpha*v - beta*v_old
        for (size_type i = 0; i < n; ++i)
            v_new(i) = w(i) - alpha * v(i) - beta * v_old(i);

        // beta_new = ||v_new||
        value_type beta_new = mtl::two_norm(v_new);

        // Apply previous Givens rotations to column [beta, alpha, beta_new]
        // of the tridiagonal matrix.

        // Step 1: Apply G_{j-2} to (0, beta) -> (epsilon, delta_bar)
        value_type epsilon   = sn_old * beta;
        value_type delta_bar = cs_old * beta;

        // Step 2: Apply G_{j-1} to (delta_bar, alpha) -> (delta, gamma_bar)
        value_type delta     = cs * delta_bar + sn * alpha;
        value_type gamma_bar = -sn * delta_bar + cs * alpha;

        // Step 3: Compute new Givens rotation G_j to zero beta_new
        value_type gamma = sqrt(gamma_bar * gamma_bar + beta_new * beta_new);

        if (gamma == zero) {
            iter.fail(2, "minres breakdown: gamma == 0");
            return iter;
        }

        value_type cs_new = gamma_bar / gamma;
        value_type sn_new = beta_new / gamma;

        // Update direction vectors
        // d_new = (M^{-1}*v - delta*d_old - epsilon*d_older) / gamma
        M.solve(z, v);
        for (size_type i = 0; i < n; ++i)
            d_older(i) = d_old(i);
        for (size_type i = 0; i < n; ++i)
            d_old(i) = d(i);
        for (size_type i = 0; i < n; ++i)
            d(i) = (z(i) - delta * d_old(i) - epsilon * d_older(i)) / gamma;

        // Update solution
        value_type tau = cs_new * eta;
        for (size_type i = 0; i < n; ++i)
            x(i) += tau * d(i);

        // Update residual tracking
        eta = -sn_new * eta;

        // Shift Lanczos vectors: v_old <- v, v <- v_new/beta_new
        for (size_type i = 0; i < n; ++i)
            v_old(i) = v(i);
        if (beta_new != zero) {
            for (size_type i = 0; i < n; ++i)
                v(i) = v_new(i) / beta_new;
        } else {
            for (size_type i = 0; i < n; ++i)
                v(i) = zero;
        }

        // Shift Givens state
        cs_old = cs;
        sn_old = sn;
        cs = cs_new;
        sn = sn_new;
        beta = beta_new;
    }

    return iter;
}

} // namespace mtl::itl
