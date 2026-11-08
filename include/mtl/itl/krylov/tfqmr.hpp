#pragma once
// MTL5 -- TFQMR (Transpose-Free Quasi-Minimal Residual) solver
// Does not require trans(A). Based on Freund (1993).
#include <mtl/concepts/vector.hpp>   // FieldVector (#503)
#include <cmath>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/dot.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/operation/mult.hpp>
#include <mtl/itl/krylov/apply_operator.hpp>

namespace mtl::itl {

/// TFQMR solver for non-symmetric systems A*x = b.
/// Does not require A^T (transpose-free). Right preconditioner M: it solves
/// (A M^-1) z = b with x = M^-1 z, so the residual it tracks is b - A x.
///
/// Freund (SIAM J. Sci. Comput. 14, 1993); Saad, Iterative Methods for Sparse
/// Linear Systems, 2nd ed., Algorithm 7.8. Each iteration is two half-steps
/// with two search vectors, u_m and u_{m+1} = u_m - alpha v_m; the next pair
/// starts from u_{m+2} = w + beta u_{m+1}, and v carries A M^-1 u through the
/// recurrence v_{m+2} = A M^-1 u_{m+2} + beta (A M^-1 u_{m+1} + beta v_m).
/// The u vectors are kept preconditioned (y = M^-1 u), so the update
/// direction d lives in x-space. Before #731 the u recurrence was missing --
/// w stood in for u_{m+1} and for u_{m+2} -- so the method stalled on
/// unsymmetric systems and converged slowly on SPD ones.
///
/// The convergence test is on tau, the quasi-residual norm, which bounds the
/// true residual by sqrt(m + 1) tau after m half-steps; in practice the two
/// are close.
///
/// Cost per iteration: two products with A, two preconditioner solves, three
/// dot products and two norms.
///
/// Accumulator (optional): accumulation type for the dot products and the
/// matrix-vector products (see math/accumulator_traits.hpp, #158).
/// Defaults to void, matching dot()/mult()'s own default -- unspecified
/// behavior is unchanged.
/// A is a Matrix or any LinearOperator (A * x). With a matrix-free A the
/// Accumulator reaches only the dot products (apply_operator.hpp, #278).
template <typename Accumulator = void,
          typename LinearOp, typename VecX, typename VecB,
          typename PC, typename Iter>
    requires FieldVector<VecX>
int tfqmr(const LinearOp& A, VecX& x, const VecB& b, const PC& M, Iter& iter) {
    using value_type = typename VecX::value_type;
    using size_type  = typename VecX::size_type;
    using std::sqrt;
    const size_type n = x.size();

    // u1 = u_m and u2 = u_{m+1}; y1, y2 their preconditioned images M^-1 u;
    // Au1, Au2 = A y1, A y2.
    vec::dense_vector<value_type> r_star(n), w(n), d(n), v(n);
    vec::dense_vector<value_type> u1(n), u2(n), y1(n), y2(n), Au1(n), Au2(n);


    value_type tau = value_type(0), theta = value_type(0), eta = value_type(0), rho = value_type(0);

    // w = b - A x, the true residual; returns its norm.
    auto true_residual = [&]() {
        apply_operator<Accumulator>(A, x, Au1);
        for (size_type i = 0; i < n; ++i)
            w(i) = b(i) - Au1(i);
        return mtl::two_norm(w);
    };

    // (Re)start the recurrences from w = b - A x: u_0 = r* = w, d = 0,
    // v_0 = A M^-1 u_0.
    auto restart = [&]() {
        for (size_type i = 0; i < n; ++i) {
            r_star(i) = w(i);
            u1(i)     = w(i);
            d(i)      = value_type(0);
        }
        M.solve(y1, u1);
        apply_operator<Accumulator>(A, y1, Au1);
        for (size_type i = 0; i < n; ++i)
            v(i) = Au1(i);
        tau   = mtl::two_norm(w);
        theta = value_type(0);
        eta   = value_type(0);
        rho   = mtl::dot<Accumulator, value_type>(r_star, w);
    };

    // One half-step: w -= alpha A M^-1 u; d = M^-1 u + (theta^2 eta / alpha) d;
    // the quasi-residual update; x += eta d.
    auto half_step = [&](const vec::dense_vector<value_type>& y,
                         const vec::dense_vector<value_type>& Au, value_type alpha) {
        for (size_type i = 0; i < n; ++i)
            w(i) -= alpha * Au(i);
        const value_type factor = theta * theta * eta / alpha;
        for (size_type i = 0; i < n; ++i)
            d(i) = y(i) + factor * d(i);
        theta = mtl::two_norm(w) / tau;
        const value_type c = value_type(1) / sqrt(value_type(1) + theta * theta);
        tau *= theta * c;
        eta = c * c * alpha;
        for (size_type i = 0; i < n; ++i)
            x(i) += eta * d(i);
    };

    // Each pass runs until tau meets the tolerance (or the iteration limit),
    // then checks the true residual. tau tracks the recursively updated w,
    // which drifts from b - A x by rounding; like CGS, whose polynomial it
    // shares, TFQMR can stall there well above the tolerance (2e-7 for an
    // unpreconditioned 32x32 convection-diffusion problem at rtol 1e-8).
    // If the true residual has not converged, restart from the current x
    // with fresh recurrences -- as MATLAB's tfqmr does. The check costs one
    // product with A per pass, a restart one more and a preconditioner solve. The returned iter reflects the true residual.
    if (iter.finished(true_residual())) return iter;
    restart();
    for (;;) {
        bool tau_done = false;
        while (!tau_done && !iter.finished(tau)) {
            ++iter;

            const value_type sigma = mtl::dot<Accumulator, value_type>(r_star, v);
            if (sigma == value_type(0)) {
                iter.fail(2, "tfqmr breakdown: sigma == 0");
                return iter;
            }
            const value_type alpha = rho / sigma;

            // First half-step, along u_m.
            half_step(y1, Au1, alpha);
            if (iter.finished(tau)) { tau_done = true; break; }

            // Second half-step, along u_{m+1} = u_m - alpha v_m.
            for (size_type i = 0; i < n; ++i)
                u2(i) = u1(i) - alpha * v(i);
            M.solve(y2, u2);
            apply_operator<Accumulator>(A, y2, Au2);
            half_step(y2, Au2, alpha);

            // u_{m+2} = w + beta u_{m+1};
            // v_{m+2} = A M^-1 u_{m+2} + beta (A M^-1 u_{m+1} + beta v_m).
            if (rho == value_type(0)) {
                iter.fail(2, "tfqmr breakdown: rho == 0");
                return iter;
            }
            const value_type rho_new = mtl::dot<Accumulator, value_type>(r_star, w);
            const value_type beta = rho_new / rho;
            rho = rho_new;

            for (size_type i = 0; i < n; ++i)
                u1(i) = w(i) + beta * u2(i);
            M.solve(y1, u1);
            apply_operator<Accumulator>(A, y1, Au1);
            for (size_type i = 0; i < n; ++i)
                v(i) = Au1(i) + beta * (Au2(i) + beta * v(i));
        }
        // tau converged, or the iteration limit was reached: judge by the
        // true residual. finished() returning true means converged or out
        // of iterations; false means restart.
        if (iter.finished(true_residual())) return iter;
        restart();
    }
}

} // namespace mtl::itl
