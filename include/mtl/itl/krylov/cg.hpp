#pragma once
// MTL5 -- Conjugate Gradient solver for symmetric positive definite systems
#include <mtl/concepts/vector.hpp>   // FieldVector (#503)
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/dot.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/operation/mult.hpp>
#include <mtl/itl/krylov/apply_operator.hpp>
#include <mtl/itl/krylov/scaled_rhs.hpp>
#include <mtl/itl/krylov/spd_check.hpp>

namespace mtl::itl {

/// Preconditioned Conjugate Gradient method.
/// Solves A*x = b where A is symmetric positive definite.
/// M is a preconditioner satisfying M.solve(x, b), symmetric (Hermitian)
/// positive definite. An M found not to be -- rho = r^H M^{-1} r negative
/// beyond rounding for an r that has not converged -- stops the solve with
/// error code 7 (basic_iteration::not_positive_definite), with x as the last
/// step left it (#949). An exactly zero rho stays a breakdown (2).
/// iter is a convergence controller (basic_iteration or derived).
///
/// Accumulator (optional): accumulation type for the two dot products and
/// the matrix-vector product (see math/accumulator_traits.hpp, #158).
/// Defaults to void, matching dot()/mult()'s own default -- unspecified
/// behavior is unchanged.
///
/// Returns the iteration object (convertible to int error code).
/// A is a Matrix or any LinearOperator (A * x). With a matrix-free A the
/// Accumulator reaches only the dot products (apply_operator.hpp, #278).
template <typename Accumulator = void,
          typename LinearOp, typename VecX, typename VecB,
          typename PC, typename Iter>
    requires FieldVector<VecX>
int cg(const LinearOp& A, VecX& x, const VecB& b, const PC& M, Iter& iter) {
    using value_type = typename VecX::value_type;
    using size_type  = typename VecX::size_type;
    const size_type n = x.size();

    // b's own copy, scaled near 1 when ||b|| is extreme (scaled_rhs.hpp, issue 924).
    detail::scaled_rhs<VecX, VecB> rhs(x, b, true);
    const auto& bs = rhs.b();

    // Workspace vectors
    vec::dense_vector<value_type> r(n), z(n), p(n), q(n);

    vec::dense_vector<value_type> Ax(n);

    // r = b - A x, the true residual.
    auto true_residual = [&]() {
        apply_operator<Accumulator>(A, x, Ax);
        for (size_type i = 0; i < n; ++i)
            r(i) = bs(i) - Ax(i);
    };

    value_type rho{}, rho_1{};

    // The loop tests the residual it updates by recurrence, r -= alpha A p,
    // which drifts from b - A x by rounding: on an unpreconditioned 2D
    // Laplacian at rtol 1e-13 it reported 9.6e-14 at a true 1.3e-12 (issue
    // 873). So when it has converged (or reached the limit) the true
    // residual decides, and if that has not converged CG restarts from the
    // current x, as tfqmr does (#752). The check costs one product with A per
    // claim of convergence. The returned iter reflects the true residual.
    true_residual();
    bool r_is_true = true;   // no step since r was computed from x
    for (;;) {
        // z = M^{-1} r, rho = dot(r, z)
        M.solve(z, r);
        rho = mtl::dot<Accumulator, value_type>(r, z);

        // The recurrences start on this pass's first step, not iter's: a reused
        // iteration object is past its first (it made cg return NaN, issue 866),
        // and so is a restart.
        bool first_step = true;
        while (!iter.finished(rhs.norm(r))) {
            ++iter;
            // rho = r^H M^-1 r is zero for an r that has not converged only
            // when M is singular or indefinite: alpha would be 0 and the next
            // beta 0 / 0 (issue 924).
            if (rho == value_type(0)) {
                iter.fail(2, "cg breakdown: rho == 0");
                return iter;
            }
            // A negative rho: M is not positive definite (#949). It ran on
            // before, with directions that are not A-conjugate in any norm.
            if (detail::not_positive_definite(rho, std::size_t(n),
                                              [&] { return mtl::two_norm(r) * mtl::two_norm(z); })) {
                iter.fail(detail::not_positive_definite_code,
                          "cg: the preconditioner is not positive definite (r^T M^-1 r < 0)");
                return iter;
            }
            r_is_true = false;
            const bool first = first_step;
            first_step = false;

            if (first) {
                // p = z
                for (size_type i = 0; i < n; ++i)
                    p(i) = z(i);
            } else {
                // p = z + (rho / rho_1) * p
                value_type beta = rho / rho_1;
                for (size_type i = 0; i < n; ++i)
                    p(i) = z(i) + beta * p(i);
            }

            // q = A * p
            apply_operator<Accumulator>(A, p, q);

            // alpha = rho / dot(p, q). On an indefinite A, p^T A p can be zero for
            // p != 0: alpha was rho / 0 and x became inf or NaN (issue 866).
            const value_type pq = mtl::dot<Accumulator, value_type>(p, q);
            if (pq == value_type(0)) {
                iter.fail(2, "cg breakdown: dot(p, A p) == 0");
                return iter;
            }
            value_type alpha = rho / pq;

            // x += alpha * p
            for (size_type i = 0; i < n; ++i)
                x(i) += alpha * p(i);

            // Save rho for next iteration
            rho_1 = rho;

            // r -= alpha * q
            for (size_type i = 0; i < n; ++i)
                r(i) -= alpha * q(i);

            // z = M^{-1} r
            M.solve(z, r);

            // rho = dot(r, z)
            rho = mtl::dot<Accumulator, value_type>(r, z);
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
