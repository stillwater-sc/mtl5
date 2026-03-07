#pragma once
// MTL5 -- BiCGSTAB(ell) solver: BiCGSTAB with higher-order stabilization
// Ported from MTL4 (Jan Bos / Peter Gottschling). Uses single right PC.
#include <cstddef>
#include <vector>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/operation/dot.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/math/identity.hpp>

namespace mtl::itl {

/// BiCGSTAB(ell) method for non-symmetric systems.
/// ell >= 1 controls the degree of the stabilization polynomial.
/// Higher ell may improve convergence on stiff or highly non-normal problems.
template <typename LinearOp, typename VecX, typename VecB,
          typename PC, typename Iter>
int bicgstab_ell(const LinearOp& A, VecX& x, const VecB& b, const PC& M,
                 Iter& iter, std::size_t ell) {
    using value_type = typename VecX::value_type;
    using size_type  = typename VecX::size_type;
    const size_type n = x.size();
    const auto zero = math::zero<value_type>();
    const auto one  = math::one<value_type>();

    // Workspace: r_hat[0..ell] and u_hat[0..ell]
    std::vector<vec::dense_vector<value_type>> r_hat(ell + 1, vec::dense_vector<value_type>(n));
    std::vector<vec::dense_vector<value_type>> u_hat(ell + 1, vec::dense_vector<value_type>(n));

    // Shift problem: save x0, set x=0, compute r_hat[0] = b - A*x0
    vec::dense_vector<value_type> x0(n);
    for (size_type i = 0; i < n; ++i)
        x0(i) = x(i);

    bool x_is_zero = (mtl::two_norm(x) == zero);
    if (!x_is_zero) {
        auto Ax = A * x;
        for (size_type i = 0; i < n; ++i)
            r_hat[0](i) = b(i) - Ax(i);
        for (size_type i = 0; i < n; ++i)
            x(i) = zero;
    } else {
        for (size_type i = 0; i < n; ++i)
            r_hat[0](i) = b(i);
    }

    // r0_tilde = r_hat[0] / ||r_hat[0]|| (fixed shadow residual)
    vec::dense_vector<value_type> r0_tilde(n);
    value_type norm_r0 = mtl::two_norm(r_hat[0]);
    if (norm_r0 == zero) {
        for (size_type i = 0; i < n; ++i)
            x(i) = x0(i);
        return iter;
    }
    for (size_type i = 0; i < n; ++i)
        r0_tilde(i) = r_hat[0](i) / norm_r0;

    vec::dense_vector<value_type> y(n);

    for (size_type i = 0; i < n; ++i)
        u_hat[0](i) = zero;

    value_type rho_0 = one, alpha = zero, omega = one;

    // MR part workspace
    mat::dense2D<value_type> tau(ell + 1, ell + 1);
    vec::dense_vector<value_type> sigma(ell + 1);
    vec::dense_vector<value_type> gamma(ell + 1);
    vec::dense_vector<value_type> gamma_a(ell + 1);
    vec::dense_vector<value_type> gamma_aa(ell + 1);

    // Helper lambda: convert internal x to real solution
    // The algorithm solves (A * M^{-1}) * z = b; internal x tracks z.
    // Real solution = M^{-1} * z + x0.
    auto finalize = [&]() {
        M.solve(y, x);
        for (size_type i = 0; i < n; ++i)
            x(i) = y(i) + x0(i);
    };

    while (!iter.finished(r_hat[0])) {
        ++iter;
        rho_0 = -omega * rho_0;

        // BiCG part
        for (std::size_t j = 0; j < ell; ++j) {
            value_type rho_1 = mtl::dot(r0_tilde, r_hat[j]);
            value_type beta = alpha * rho_1 / rho_0;
            rho_0 = rho_1;

            for (std::size_t i = 0; i <= j; ++i) {
                for (size_type k = 0; k < n; ++k)
                    u_hat[i](k) = r_hat[i](k) - beta * u_hat[i](k);
            }

            // u_hat[j+1] = A * M^{-1} * u_hat[j]
            M.solve(y, u_hat[j]);
            auto Ay = A * y;
            for (size_type k = 0; k < n; ++k)
                u_hat[j + 1](k) = Ay(k);

            value_type gamma_val = mtl::dot(r0_tilde, u_hat[j + 1]);
            if (gamma_val == zero) {
                iter.fail(3, "bicgstab_ell breakdown: gamma == 0");
                finalize();
                return iter;
            }
            alpha = rho_0 / gamma_val;

            for (std::size_t i = 0; i <= j; ++i) {
                for (size_type k = 0; k < n; ++k)
                    r_hat[i](k) -= alpha * u_hat[i + 1](k);
            }

            // Early convergence check (before x += alpha*u_hat[0], matching MTL4)
            if (iter.finished(r_hat[j])) {
                finalize();
                return iter;
            }

            // r_hat[j+1] = A * M^{-1} * r_hat[j]
            M.solve(y, r_hat[j]);
            auto Ar = A * y;
            for (size_type k = 0; k < n; ++k)
                r_hat[j + 1](k) = Ar(k);

            for (size_type k = 0; k < n; ++k)
                x(k) += alpha * u_hat[0](k);
        }

        // Modified Gram-Schmidt (MR part)
        for (std::size_t j = 1; j <= ell; ++j) {
            for (std::size_t i = 1; i < j; ++i) {
                if (sigma(i) == zero) continue;
                tau(i, j) = mtl::dot(r_hat[j], r_hat[i]) / sigma(i);
                for (size_type k = 0; k < n; ++k)
                    r_hat[j](k) -= tau(i, j) * r_hat[i](k);
            }
            sigma(j) = mtl::dot(r_hat[j], r_hat[j]);
            gamma_a(j) = mtl::dot(r_hat[0], r_hat[j]) / sigma(j);
        }

        gamma(ell) = gamma_a(ell);
        omega = gamma(ell);

        if (omega == zero) {
            iter.fail(3, "bicgstab_ell breakdown: omega == 0");
            finalize();
            return iter;
        }

        // Back-solve for gamma
        for (std::size_t jj = 1; jj < ell; ++jj) {
            std::size_t j = ell - jj;
            value_type sum = zero;
            for (std::size_t i = j + 1; i <= ell; ++i)
                sum += tau(j, i) * gamma(i);
            gamma(j) = gamma_a(j) - sum;
        }

        // Compute gamma_aa (double-prime)
        // gamma_aa(j) = gamma(j+1) + sum_{k=j+1}^{ell-1} tau(j,k) * gamma(k+1)
        for (std::size_t j = 1; j < ell; ++j) {
            value_type sum = zero;
            for (std::size_t k = j + 1; k < ell; ++k)
                sum += tau(j, k) * gamma(k + 1);
            gamma_aa(j) = gamma(j + 1) + sum;
        }

        // Update x, r_hat[0], u_hat[0]
        for (size_type k = 0; k < n; ++k)
            x(k) += gamma(1) * r_hat[0](k);
        for (size_type k = 0; k < n; ++k)
            r_hat[0](k) -= gamma_a(ell) * r_hat[ell](k);
        for (size_type k = 0; k < n; ++k)
            u_hat[0](k) -= gamma(ell) * u_hat[ell](k);

        for (std::size_t j = 1; j < ell; ++j) {
            for (size_type k = 0; k < n; ++k)
                u_hat[0](k) -= gamma(j) * u_hat[j](k);
            for (size_type k = 0; k < n; ++k)
                x(k) += gamma_aa(j) * r_hat[j](k);
            for (size_type k = 0; k < n; ++k)
                r_hat[0](k) -= gamma_a(j) * r_hat[j](k);
        }
    }

    // Convert to real solution and undo shift
    finalize();
    return iter;
}

} // namespace mtl::itl
