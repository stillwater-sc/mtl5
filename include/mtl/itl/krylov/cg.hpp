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
#include <mtl/math/pow2_scaling.hpp>
#include <limits>

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
            if (detail::not_positive_definite<Accumulator>(
                    rho, std::size_t(n), [&] { return std::pair{mtl::two_norm(r), mtl::two_norm(z)}; })) {
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

            // alpha = rho / dot(p, q), then x += alpha p and r -= alpha q.
            //
            // p^T A p leaves the range first (issue 955). For a preconditioner
            // scaled by s, z = M^{-1} r and p scale as 1/s, rho = r^T z as 1/s,
            // but p^T A p as 1/s^2: at M = 4 * 2^600 I it underflowed to 0 (a
            // false breakdown, code 2), at 4 * 2^-600 I it overflowed, alpha
            // was 0 and x never moved. So when p^T A p is not a normal number,
            // the step is redone on p-hat = 2^-e p, with 2^e <= max |p_i|:
            // p-hat^T A p-hat is near the scale of A, and
            //   alpha p = (rho / p-hat^T A p-hat) 2^-e p-hat,
            //   alpha q = (rho / p-hat^T A p-hat) 2^-e A p-hat,
            // exactly, since a power of two scales without rounding. Where
            // p^T A p is normal, nothing changes, bit for bit. A p^T A p that
            // is zero after the rescaling is a true zero, as on an indefinite A
            // (issue 866): a breakdown.
            const value_type pq = mtl::dot<Accumulator, value_type>(p, q);
            using mag_t = magnitude_t<value_type>;
            using std::abs;
            const mag_t apq = abs(pq);
            value_type c = value_type(0);       // the step: x += c u, r -= c w
            const vec::dense_vector<value_type>* u = &p;
            const vec::dense_vector<value_type>* w = &q;
            if (apq >= std::numeric_limits<mag_t>::min() && apq <= std::numeric_limits<mag_t>::max()) {
                c = rho / pq;
            } else {
                mag_t pmax(0);
                for (size_type i = 0; i < n; ++i) {
                    const mag_t a = abs(p(i));
                    if (a > pmax) pmax = a;
                }
                const int e = math::pow2_scaling<mag_t>::exponent(pmax);   // 0 for 0, NaN or inf
                const mag_t f = math::pow2_scaling<mag_t>::scale(mag_t(1), -e);
                for (size_type i = 0; i < n; ++i) q(i) = p(i) * f;           // q holds p-hat
                apply_operator<Accumulator>(A, q, Ax);                       // Ax = A p-hat
                const value_type pqh = mtl::dot<Accumulator, value_type>(q, Ax);
                if (pqh == value_type(0)) {
                    iter.fail(2, "cg breakdown: dot(p, A p) == 0");
                    return iter;
                }
                c = (rho / pqh) * f;
                u = &q;
                w = &Ax;
            }

            // x += alpha * p
            for (size_type i = 0; i < n; ++i)
                x(i) += c * (*u)(i);

            // Save rho for next iteration
            rho_1 = rho;

            // r -= alpha * q
            for (size_type i = 0; i < n; ++i)
                r(i) -= c * (*w)(i);

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
