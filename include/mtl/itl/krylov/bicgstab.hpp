#pragma once
// MTL5 -- BiCGSTAB (Bi-Conjugate Gradient Stabilized) solver
#include <mtl/concepts/vector.hpp>   // FieldVector (#503)
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/dot.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/operation/mult.hpp>
#include <mtl/itl/krylov/apply_operator.hpp>
#include <mtl/itl/krylov/scaled_rhs.hpp>
#include <mtl/itl/krylov/pow2_rescale.hpp>

namespace mtl::itl {

/// BiCGSTAB method for non-symmetric systems.
/// Solves A*x = b with preconditioner M and iteration controller iter.
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
int bicgstab(const LinearOp& A, VecX& x, const VecB& b, const PC& M, Iter& iter) {
    using value_type = typename VecX::value_type;
    using size_type  = typename VecX::size_type;
    const size_type n = x.size();

    // b's own copy, scaled near 1 when ||b|| is extreme (scaled_rhs.hpp, issue 924).
    detail::scaled_rhs<VecX, VecB> rhs(x, b, true);
    const auto& bs = rhs.b();

    // Workspace vectors
    vec::dense_vector<value_type> r(n), r_star(n), p(n), phat(n);
    vec::dense_vector<value_type> v(n), s(n), shat(n), t(n);

    vec::dense_vector<value_type> Ax(n);

    // r = b - A x, the true residual.
    auto true_residual = [&]() {
        apply_operator<Accumulator>(A, x, Ax);
        for (size_type i = 0; i < n; ++i)
            r(i) = bs(i) - Ax(i);
    };

    value_type rho_1 = value_type(1);
    value_type alpha = value_type(1);
    value_type omega = value_type(1);

    // The loop tests the residual it updates by recurrence (s = r - alpha v,
    // r = s - omega t), which drifts from b - A x by rounding. On a strongly
    // nonnormal convection-diffusion system without a PC, BiCGSTAB reported
    // 3e-11 at a true residual of 0.2 (issue 873). So when it has converged
    // (or reached the limit) the true residual decides, and if that has not
    // converged BiCGSTAB restarts from the current x with fresh recurrences
    // and shadow residual, as tfqmr does (#752). The check costs one product
    // with A per claim of convergence. The returned iter reflects the true
    // residual.
    true_residual();
    bool r_is_true = true;   // no step since r was computed from x
    for (;;) {
        // (Re)start: r* = r, p = v = 0, rho_1 = alpha = omega = 1.
        for (size_type i = 0; i < n; ++i) {
            r_star(i) = r(i);
            v(i) = value_type(0);
            p(i) = value_type(0);
        }
        rho_1 = alpha = omega = value_type(1);

        while (!iter.finished(rhs.norm(r))) {
            ++iter;
            r_is_true = false;

            value_type rho = mtl::dot<Accumulator, value_type>(r_star, r);

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
            apply_operator<Accumulator>(A, phat, v);

            // alpha = rho / dot(r_star, v)
            value_type rsv = mtl::dot<Accumulator, value_type>(r_star, v);
            if (rsv == value_type(0)) {
                iter.fail(3, "bicgstab breakdown: dot(r_star, v) == 0");
                return iter;
            }
            alpha = rho / rsv;

            // s = r - alpha * v
            for (size_type i = 0; i < n; ++i)
                s(i) = r(i) - alpha * v(i);

            // Check convergence of s. Only convergence: the iteration limit is
            // tested on the full step's r, so the last allowed iteration is not
            // cut to the half step x += alpha phat (issue 866). The true
            // residual then decides, below.
            if (iter.converged_now(rhs.norm(s))) {
                // x += alpha * phat
                for (size_type i = 0; i < n; ++i)
                    x(i) += alpha * phat(i);
                break;
            }

            // shat = M^{-1} s
            M.solve(shat, s);

            // t = A * shat
            apply_operator<Accumulator>(A, shat, t);

            // omega = dot(t, s) / dot(t, t)
            //
            // t^T t leaves the range first (issue 958). For a preconditioner
            // scaled by s, p-hat, v, s-hat and t scale as 1/s, and rho, r*^T v
            // and t^T s as 1/s, but t^T t as 1/s^2: at M = 4 * 2^600 I it
            // underflowed to 0 (a false breakdown, code 4), at 4 * 2^-600 I it
            // overflowed, omega was 0 (code 5). So when t^T t is not a normal
            // number omega is formed from t-hat = 2^-e t (pow2_rescale.hpp):
            //   omega = (t^T s) / (t^T t) = 2^-e (t-hat^T s) / (t-hat^T t-hat),
            // exactly, since a power of two scales without rounding. Where
            // t^T t is normal nothing changes, bit for bit. One that is zero
            // after the rescaling is a true zero: a breakdown. Ax holds t-hat;
            // it is free until the next true residual.
            value_type tt = mtl::dot<Accumulator, value_type>(t, t);
            if (detail::is_normal_magnitude(tt)) {
                omega = mtl::dot<Accumulator, value_type>(t, s) / tt;
            } else {
                const int e = detail::max_abs_exponent(t);
                detail::scale_by_pow2(t, -e, Ax);
                const value_type tth = mtl::dot<Accumulator, value_type>(Ax, Ax);
                if (tth == value_type(0)) {
                    iter.fail(4, "bicgstab breakdown: dot(t, t) == 0");
                    return iter;
                }
                omega = detail::scaled_quotient(mtl::dot<Accumulator, value_type>(Ax, s), tth, -e);
            }
            // omega == 0 stalls x and divides the next beta by zero (issue 866).
            if (omega == value_type(0)) {
                for (size_type i = 0; i < n; ++i)
                    x(i) += alpha * phat(i);
                iter.fail(5, "bicgstab breakdown: omega == 0");
                return iter;
            }

            // x += alpha * phat + omega * shat
            for (size_type i = 0; i < n; ++i)
                x(i) += alpha * phat(i) + omega * shat(i);

            // r = s - omega * t
            for (size_type i = 0; i < n; ++i)
                r(i) = s(i) - omega * t(i);

            rho_1 = rho;
        }
        // Converged by the recurrence, or at the limit: the true residual
        // decides (finished() is true again at the limit).
        if (r_is_true) return iter;
        true_residual();
        r_is_true = true;
        if (iter.finished(rhs.norm(r))) return iter;
    }
}

} // namespace mtl::itl
