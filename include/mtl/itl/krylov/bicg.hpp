#pragma once
// MTL5 -- BiConjugate Gradient (BiCG) solver for non-symmetric systems
// Maintains shadow residuals and uses trans(A) for the adjoint multiply.
#include <mtl/concepts/vector.hpp>   // FieldVector (#503)
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/dot.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/operation/trans.hpp>
#include <mtl/operation/mult.hpp>
#include <mtl/itl/krylov/apply_operator.hpp>

namespace mtl::itl {

/// BiConjugate Gradient method.
/// Solves A*x = b for non-symmetric A. Uses trans(A) internally.
/// M must provide solve() and adjoint_solve().
///
/// Accumulator (optional): accumulation type for the two dot products and
/// the two matrix-vector products (see math/accumulator_traits.hpp, #158).
/// Defaults to void, matching dot()/mult()\'s own default -- unspecified
/// behavior is unchanged.
/// A is a Matrix or any LinearOperator (A * x) that also provides trans(A)
/// by ADL. With a matrix-free A the Accumulator reaches only the dot products
/// (apply_operator.hpp, #278).
template <typename Accumulator = void,
          typename LinearOp, typename VecX, typename VecB,
          typename PC, typename Iter>
    requires FieldVector<VecX>
int bicg(const LinearOp& A, VecX& x, const VecB& b, const PC& M, Iter& iter) {
    using value_type = typename VecX::value_type;
    using size_type  = typename VecX::size_type;
    const size_type n = x.size();

    // Workspace: 8 vectors
    vec::dense_vector<value_type> r(n), r_tilde(n);
    vec::dense_vector<value_type> z(n), z_tilde(n);
    vec::dense_vector<value_type> p(n), p_tilde(n);
    vec::dense_vector<value_type> q(n), q_tilde(n);

    vec::dense_vector<value_type> Ax(n);

    // r = b - A x, the true residual.
    auto true_residual = [&]() {
        apply_operator<Accumulator>(A, x, Ax);
        for (size_type i = 0; i < n; ++i)
            r(i) = b(i) - Ax(i);
    };

    value_type rho{}, rho_1{};

    // The loop tests the residual it updates by recurrence, r -= alpha A p,
    // which drifts from b - A x by rounding: on a 2D convection-diffusion
    // system with ILU(0) at rtol 1e-13 it reported 4.9e-14 at a true 4.1e-11
    // (issue 873). So when it has converged (or reached the limit) the true
    // residual decides, and if that has not converged BiCG restarts from the
    // current x with a fresh shadow residual, as tfqmr does (#752). The check
    // costs one product with A per claim of convergence. The returned iter
    // reflects the true residual.
    true_residual();
    bool r_is_true = true;   // no step since r was computed from x
    for (;;) {
        for (size_type i = 0; i < n; ++i)
            r_tilde(i) = r(i);  // shadow residual = r initially

        // z = M^{-1} r,  z_tilde = M^{-T} r_tilde
        M.solve(z, r);
        M.adjoint_solve(z_tilde, r_tilde);

        // rho = z^T r~ : the PRECONDITIONED residual against the UNpreconditioned
        // shadow residual (Barrett et al., Templates, section 2.3.5; Saad,
        // Iterative Methods, Alg. 7.1).
        //
        // This used to be dot(z_tilde, z), which applies the preconditioner twice:
        //   z~ . z = (M^-T r~) . (M^-1 r) = r~^T M^-1 M^-1 r
        // where the algorithm needs r~^T M^-1 r. With pc::identity, z == r and
        // z~ == r~, so the two coincide -- which is exactly why identity was the
        // only preconditioner bicg converged with (#392).
        rho = mtl::dot<Accumulator, value_type>(r_tilde, z);

        // The recurrences start on this pass's first step, not iter's: a reused
        // iteration object is past its first (it made cg return NaN, issue 866),
        // and so is a restart.
        bool first_step = true;
        while (!iter.finished(r)) {
            ++iter;
            r_is_true = false;
            const bool first = first_step;
            first_step = false;

            if (first) {
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
            apply_operator<Accumulator>(A, p, q);

            // q_tilde = A^T * p_tilde
            apply_transpose<Accumulator>(A, p_tilde, q_tilde);

            // p~^T A p == 0 is a breakdown: alpha was rho / 0 (issue 866).
            const value_type ptq = mtl::dot<Accumulator, value_type>(p_tilde, q);
            if (ptq == value_type(0)) {
                iter.fail(3, "bicg breakdown: dot(p_tilde, A p) == 0");
                return iter;
            }
            value_type alpha = rho / ptq;

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
            rho = mtl::dot<Accumulator, value_type>(r_tilde, z);   // z^T r~, see above (#392)

            // rho == 0 is ambiguous between a true Lanczos breakdown and exact
            // convergence (r == 0 drives z == z_tilde == 0, hence rho == 0).
            // Only treat it as breakdown if we have not actually converged --
            // otherwise a solver that lands on the exact solution in one step
            // (e.g. 1x1 systems) gets misreported as failed.
            if (rho == value_type(0) && !iter.finished(r)) {
                iter.fail(2, "bicg breakdown: rho == 0");
                return iter;
            }
        }

        // Converged by the recurrence, or at the limit: the true residual
        // decides (finished() is true again at the limit).
        if (r_is_true) return iter;
        true_residual();
        r_is_true = true;
        if (iter.finished(r)) return iter;
    }
}

} // namespace mtl::itl
