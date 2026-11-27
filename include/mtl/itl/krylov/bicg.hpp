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
#include <mtl/itl/krylov/scaled_rhs.hpp>
#include <mtl/itl/krylov/pow2_rescale.hpp>
#include <mtl/functor/scalar/conj.hpp>

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
    using conj_t     = functor::scalar::conj<value_type>;
    const size_type n = x.size();

    // b's own copy, scaled near 1 when ||b|| is extreme (scaled_rhs.hpp, issue 924).
    detail::scaled_rhs<VecX, VecB> rhs(x, b, true);
    const auto& bs = rhs.b();

    // Workspace: 8 vectors
    vec::dense_vector<value_type> r(n), r_tilde(n);
    vec::dense_vector<value_type> z(n), z_tilde(n);
    vec::dense_vector<value_type> p(n), p_tilde(n);
    vec::dense_vector<value_type> q(n), q_tilde(n);

    vec::dense_vector<value_type> Ax(n);
    vec::dense_vector<value_type> tmp(is_complex_v<value_type> ? n : size_type(0));   // apply_adjoint's conj(x)

    // r = b - A x, the true residual.
    auto true_residual = [&]() {
        apply_operator<Accumulator>(A, x, Ax);
        for (size_type i = 0; i < n; ++i)
            r(i) = bs(i) - Ax(i);
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

        // z = M^{-1} r,  z_tilde = M^{-H} r_tilde
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
        while (!iter.finished(rhs.norm(r))) {
            ++iter;
            // rho == 0 with r not converged is a Lanczos breakdown: alpha
            // would be 0 and the next beta 0 / 0. Tested here, where the loop
            // has already found that r has not converged, so it covers the
            // first step of a pass too (issue 924).
            if (rho == value_type(0)) {
                iter.fail(2, "bicg breakdown: rho == 0");
                return iter;
            }
            r_is_true = false;
            const bool first = first_step;
            first_step = false;

            if (first) {
                for (size_type i = 0; i < n; ++i) {
                    p(i) = z(i);
                    p_tilde(i) = z_tilde(i);
                }
            } else {
                // The shadow sequence takes the conjugate coefficients: it is
                // the BiCG of A^H, so that p~^H A p_j = 0 for j < k (issue 924).
                const value_type beta  = rho / rho_1;
                const value_type cbeta = conj_t::apply(beta);
                for (size_type i = 0; i < n; ++i) {
                    p(i) = z(i) + beta * p(i);
                    p_tilde(i) = z_tilde(i) + cbeta * p_tilde(i);
                }
            }

            // q = A * p
            apply_operator<Accumulator>(A, p, q);

            // q_tilde = A^H * p_tilde (issue 924: A^T did not converge for complex)
            apply_adjoint<Accumulator>(A, p_tilde, q_tilde, tmp);

            // alpha = rho / p~^H A p, then x += alpha p, r -= alpha A p and
            // r~ -= conj(alpha) A^H p~.
            //
            // p~^H A p leaves the range first (issue 958). For a preconditioner
            // scaled by s, p and p~ each scale as 1/s and rho = r~^T z as 1/s,
            // but p~^H A p as 1/s^2: at M = 4 * 2^600 I it underflowed to 0 (a
            // false breakdown, code 3), at 4 * 2^-600 I it overflowed, alpha
            // was 0 and the solve ran to its limit. So when it is not a normal
            // number the step is redone on p-hat = 2^-e p and p~-hat = 2^-f p~,
            // each with its own power of two (pow2_rescale.hpp). With
            // d = p~-hat^H A p-hat = 2^-(e+f) p~^H A p, so alpha = 2^-(e+f) rho / d:
            //   alpha p    = (2^-f rho / d) p-hat,  alpha A p = (2^-f rho / d) A p-hat,
            //   conj(alpha) A^H p~ = conj(2^-e rho / d) A^H p~-hat,
            // exactly, since a power of two scales without rounding. Where
            // p~^H A p is normal nothing changes, bit for bit. One that is zero
            // after the rescaling is a true zero: a breakdown (issue 866).
            // z and z~ are recomputed below, so they and Ax hold the rescaled
            // vectors.
            const value_type ptq = mtl::dot<Accumulator, value_type>(p_tilde, q);
            value_type c  = value_type(0);    // x += c u, r -= c w
            value_type ct = value_type(0);    // r~ -= ct wt
            const vec::dense_vector<value_type>* u  = &p;
            const vec::dense_vector<value_type>* w  = &q;
            const vec::dense_vector<value_type>* wt = &q_tilde;
            if (detail::is_normal_magnitude(ptq)) {
                c  = rho / ptq;
                ct = conj_t::apply(c);
            } else {
                const int e = detail::max_abs_exponent(p);
                const int f = detail::max_abs_exponent(p_tilde);
                detail::scale_by_pow2(p, -e, z);                    // z  = p-hat
                detail::scale_by_pow2(p_tilde, -f, z_tilde);        // z~ = p~-hat
                apply_operator<Accumulator>(A, z, Ax);              // Ax = A p-hat
                apply_adjoint<Accumulator>(A, z_tilde, q_tilde, tmp);   // q~ = A^H p~-hat
                const value_type d = mtl::dot<Accumulator, value_type>(z_tilde, Ax);
                if (d == value_type(0)) {
                    iter.fail(3, "bicg breakdown: dot(p_tilde, A p) == 0");
                    return iter;
                }
                const value_type rd = rho / d;
                c  = detail::by_pow2(rd, -f);
                ct = conj_t::apply(detail::by_pow2(rd, -e));
                u = &z;
                w = &Ax;
            }

            // x += alpha * p
            for (size_type i = 0; i < n; ++i)
                x(i) += c * (*u)(i);

            // r -= alpha * q
            for (size_type i = 0; i < n; ++i)
                r(i) -= c * (*w)(i);

            // r_tilde -= conj(alpha) * q_tilde
            for (size_type i = 0; i < n; ++i)
                r_tilde(i) -= ct * (*wt)(i);

            // z = M^{-1} r, z_tilde = M^{-H} r_tilde
            M.solve(z, r);
            M.adjoint_solve(z_tilde, r_tilde);

            rho_1 = rho;
            rho = mtl::dot<Accumulator, value_type>(r_tilde, z);   // z^T r~, see above (#392)
            // rho == 0 is ambiguous between a Lanczos breakdown and exact
            // convergence (r == 0 drives z == z_tilde == 0). The test at the
            // top of the loop runs after the convergence test, so a solver that
            // lands on the exact solution in one step (a 1x1 system) is not
            // reported as failed.
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
