#pragma once
// MTL5 -- MINRES (Minimum Residual) solver for symmetric indefinite systems
// Lanczos-based 3-term recurrence with Givens rotations.
// Guarantees monotonically decreasing residual norm for symmetric A.
// Reference: Paige & Saunders (1975), Saad Algorithm 6.13.
#include <mtl/concepts/vector.hpp>   // FieldVector (#503)
#include <cmath>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/dot.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/operation/mult.hpp>
#include <mtl/itl/krylov/apply_operator.hpp>
#include <mtl/itl/krylov/scaled_rhs.hpp>
#include <mtl/math/identity.hpp>
#include <mtl/math/pow2_scaling.hpp>

namespace mtl::itl {

/// MINRES method for symmetric (possibly indefinite) systems.
/// Solves A*x = b with preconditioner M and iteration controller iter.
/// A must be symmetric and M symmetric positive definite (Hermitian for a
/// complex system): the Lanczos process runs in the M^{-1} inner product.
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
int minres(const LinearOp& A, VecX& x, const VecB& b, const PC& M, Iter& iter) {
    using value_type = typename VecX::value_type;
    using size_type  = typename VecX::size_type;
    using std::sqrt;
    using std::abs;

    const size_type n = x.size();
    const auto zero = math::zero<value_type>();

    // b's own copy, so x may alias b (issue 924). minres scales its own dot
    // products (#671), so the copy is not scaled.
    detail::scaled_rhs<VecX, VecB> rhs(x, b, false);
    const auto& bs = rhs.b();

    // Preconditioned Lanczos in the M^{-1} inner product (Paige & Saunders;
    // Greenbaum, Iterative Methods, Sec. 8.2). With M SPD this is MINRES on
    // M^{-1/2} A M^{-1/2}, which is symmetric. The former loop ran Lanczos on
    // A M^{-1} in the plain inner product, which is symmetric only for M = cI:
    // a diagonal PC with a varying diagonal did not converge (issue 866).
    //
    //   v: the Lanczos vectors in A's range, scaled so that v^T M^{-1} v = 1;
    //   z = M^{-1} v, the direction the solution is built from;
    //   beta = sqrt(v^T M^{-1} v), alpha = z^T A z.
    // One preconditioner solve and one product with A per step.
    vec::dense_vector<value_type> v_old(n), v(n), v_new(n);
    vec::dense_vector<value_type> z(n), z_new(n);
    vec::dense_vector<value_type> d(n), d_old(n), d_older(n);
    vec::dense_vector<value_type> w(n);

    // ||v||_{M^{-1}} = sqrt(v^T z), z = M^{-1} v; real and positive for an SPD
    // (HPD) M, which MINRES requires. Formed as sqrt(a) sqrt(c) sqrt(|(v/a)^T (z/c)|)
    // with a = ||v||_2 (two_norm scales) and c = max |z_i|, so v^T z cannot
    // overflow or underflow for an operator of 2^+-600 (#671), nor when z's
    // entries are finite but its 2-norm is not (z = 2^1023 ones with n = 16).
    // `tmp` holds v/a and `zs` z/c. For M = I it is a.
    vec::dense_vector<value_type> zs(n);
    auto m_norm = [&](const vec::dense_vector<value_type>& vv, const vec::dense_vector<value_type>& zz,
                      vec::dense_vector<value_type>& tmp) -> value_type {
        const auto a = mtl::two_norm(vv);
        if (a == decltype(a)(0)) return zero;
        const auto c = mtl::infinity_norm(zz);
        if (c == decltype(c)(0)) return zero;
        for (size_type i = 0; i < n; ++i) {
            tmp(i) = vv(i) / a;
            zs(i)  = zz(i) / c;
        }
        return value_type(sqrt(a) * sqrt(c) * sqrt(abs(mtl::dot<Accumulator, value_type>(tmp, zs))));
    };

    // Initial residual: r = b - A*x
    apply_operator<Accumulator>(A, x, w);
    for (size_type i = 0; i < n; ++i)
        v(i) = bs(i) - w(i);
    const auto r0_norm = mtl::two_norm(v);
    M.solve(z, v);
    value_type beta = m_norm(v, z, w);

    // ||r0||_{M^-1} = 0: r0 = 0, or an M^-1 that maps r0 to 0. The iteration
    // object records either, rather than keeping the residual and finished
    // state of an earlier solve (issue 924).
    if (beta == zero) {
        if (r0_norm == decltype(r0_norm)(0)) iter.finished(r0_norm);
        else iter.fail(2, "minres breakdown: ||r0||_M == 0 for r0 != 0");
        return iter;
    }

    for (size_type i = 0; i < n; ++i) {
        v(i) /= beta;
        z(i) /= beta;
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

    // Transformed RHS: |eta| is the residual in the M^{-1} norm. Scaled by
    // ||r0||_2 / ||r0||_{M^{-1}}, it estimates the 2-norm the iteration object
    // tests (exactly so for M = I); when the estimate meets the tolerance the
    // true residual decides, and the recurrence goes on if it has not.
    value_type eta = beta;
    const auto scale = r0_norm / abs(beta);
    auto true_residual = [&]() {
        apply_operator<Accumulator>(A, x, w);
        for (size_type i = 0; i < n; ++i)
            w(i) = bs(i) - w(i);
        return mtl::two_norm(w);
    };

    for (;;) {
        if (iter.finished(abs(eta) * scale)) {
            // Converged by the estimate, or at the limit: the true residual
            // decides (finished() is true again at the limit).
            if (iter.finished(true_residual())) return iter;
        }
        ++iter;

        // Lanczos step: w = A z, alpha = z^T A z
        apply_operator<Accumulator>(A, z, w);
        value_type alpha = mtl::dot<Accumulator, value_type>(z, w);

        // v_new = w - alpha*v - beta*v_old; beta_new = ||v_new||_{M^{-1}}
        for (size_type i = 0; i < n; ++i)
            v_new(i) = w(i) - alpha * v(i) - beta * v_old(i);
        M.solve(z_new, v_new);
        value_type beta_new = m_norm(v_new, z_new, w);   // w is free until the next step

        // Apply previous Givens rotations to column [beta, alpha, beta_new]
        // of the tridiagonal matrix.

        // Step 1: Apply G_{j-2} to (0, beta) -> (epsilon, delta_bar)
        value_type epsilon   = sn_old * beta;
        value_type delta_bar = cs_old * beta;

        // Step 2: Apply G_{j-1} to (delta_bar, alpha) -> (delta, gamma_bar)
        value_type delta     = cs * delta_bar + sn * alpha;
        value_type gamma_bar = -sn * delta_bar + cs * alpha;

        // Step 3: Compute new Givens rotation G_j to zero beta_new, scaled by
        // an exact power of two so gamma does not overflow or underflow (#671)
        const auto g_new = math::givens_rotation(gamma_bar, beta_new);
        value_type gamma = g_new.r;

        if (gamma == zero) {
            iter.fail(2, "minres breakdown: gamma == 0");
            return iter;
        }

        value_type cs_new = g_new.c;
        value_type sn_new = g_new.s;

        // Update direction vectors
        // d_new = (z - delta*d_old - epsilon*d_older) / gamma, z = M^{-1} v
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

        // Shift Lanczos vectors: v_old <- v, (v, z) <- (v_new, z_new) / beta_new
        for (size_type i = 0; i < n; ++i)
            v_old(i) = v(i);
        if (beta_new != zero) {
            for (size_type i = 0; i < n; ++i) {
                v(i) = v_new(i) / beta_new;
                z(i) = z_new(i) / beta_new;
            }
        } else {
            for (size_type i = 0; i < n; ++i) {
                v(i) = zero;
                z(i) = zero;
            }
        }

        // Shift Givens state
        cs_old = cs;
        sn_old = sn;
        cs = cs_new;
        sn = sn_new;
        beta = beta_new;
    }
}

} // namespace mtl::itl
