#pragma once
// MTL5 -- IDR(s) solver (Induced Dimension Reduction)
// Sonneveld & van Gijzen (2008). Modern Krylov solver for non-symmetric systems.
// Parameter s controls shadow space dimension (default 4).
#include <mtl/concepts/vector.hpp>   // FieldVector (#503)
#include <cmath>
#include <random>
#include <numbers>
#include <vector>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/operation/dot.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/operation/mult.hpp>
#include <mtl/itl/krylov/apply_operator.hpp>
#include <mtl/concepts/matrix.hpp>
#include <mtl/math/identity.hpp>

namespace mtl::itl {

/// IDR(s) solver for non-symmetric systems A*x = b.
/// s = shadow space dimension (larger s -> fewer iterations, more memory:
/// 2s + 4 vectors of length n). Right preconditioner M.
///
/// The prototype IDR(s) of Sonneveld & van Gijzen (SIAM J. Sci. Comput. 31,
/// 2008, Alg. 3.1). After s initial minimal-residual steps, the iteration
/// runs in cycles of s + 1 steps. Each step solves the s x s system
/// (P^T dR) c = P^T r, so the new residual is orthogonal to the shadow space
/// P. The first step of a cycle also picks omega, minimizing || v - omega t ||;
/// the other s steps reuse that omega and take dR = -A dX. All s + 1 residuals
/// of a cycle must share one omega for them to lie in the same space G_j --
/// that is the dimension reduction. Before #734 omega was recomputed at every
/// step, which loses it: iteration counts were irregular in s (s = 8 took
/// 932 iterations where s = 4 took 153, on a Poisson problem), and on some
/// shadow spaces the method diverged.
///
/// Every step costs one product with A and one preconditioner solve, and
/// counts as one iteration.
///
/// P is s Gaussian vectors from a fixed seed, so a run is reproducible. Its
/// draw makes little difference: over 20 seeds, the iteration counts on
/// convection-diffusion and Poisson systems spread by 5-15%, and the median
/// falls monotonically with s. Orthonormalizing P, or enlarging a small omega
/// (Sleijpen and van Gijzen's kappa = 0.7), moved those medians by at most
/// about 6% and was left out.
///
/// A is a Matrix or any LinearOperator (A * x). With a matrix-free A the
/// Accumulator reaches only the dot products (apply_operator.hpp, #278).
template <typename Accumulator = void,
          typename LinearOp, typename VecX, typename VecB,
          typename PC, typename Iter>
    requires FieldVector<VecX>
int idr_s(const LinearOp& A, VecX& x, const VecB& b, const PC& M, Iter& iter,
          typename VecX::size_type s = 4) {
    using value_type = typename VecX::value_type;
    using size_type  = typename VecX::size_type;
    using std::abs;
    const size_type n = x.size();

    // Random shadow space P (n x s)
    std::mt19937 gen(42);
    auto next_uniform = [&gen]() {
        return static_cast<double>(gen() - gen.min()) /
               static_cast<double>(gen.max() - gen.min());
    };
    auto next_normal = [&]() {
        double u1 = std::max(next_uniform(), 1e-12);
        double u2 = next_uniform();
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * std::numbers::pi_v<double> * u2);
    };

    std::vector<vec::dense_vector<value_type>> P(s, vec::dense_vector<value_type>(n));
    for (size_type j = 0; j < s; ++j)
        for (size_type i = 0; i < n; ++i)
            P[j](i) = value_type(next_normal());

    // Workspace
    vec::dense_vector<value_type> r(n), v(n), vhat(n), t(n), dxn(n);
    std::vector<vec::dense_vector<value_type>> dR(s, vec::dense_vector<value_type>(n));
    std::vector<vec::dense_vector<value_type>> dX(s, vec::dense_vector<value_type>(n));

    // r = b - A*x
    apply_operator<Accumulator>(A, x, t);
    for (size_type i = 0; i < n; ++i)
        r(i) = b(i) - t(i);

    // Mmat = P^T dR (s x s), m = P^T r, and the solution c of Mmat c = m
    mat::dense2D<value_type> Mmat(s, s), Mcopy(s, s);
    vec::dense_vector<value_type> m(s), f(s), c(s);

    // omega minimizing || w - omega t ||; kept if t = 0.
    value_type omega = value_type(1);
    auto min_residual_omega = [&](const vec::dense_vector<value_type>& w) {
        const value_type tt = mtl::dot<Accumulator, value_type>(t, t);
        if (tt != value_type(0))
            omega = mtl::dot<Accumulator, value_type>(t, w) / tt;
    };

    // s initial minimal-residual steps build dR and dX.
    for (size_type k = 0; k < s; ++k) {
        M.solve(vhat, r);
        apply_operator<Accumulator>(A, vhat, t);
        min_residual_omega(r);
        for (size_type i = 0; i < n; ++i) {
            dX[k](i) = omega * vhat(i);
            dR[k](i) = -omega * t(i);
            x(i) += dX[k](i);
            r(i) += dR[k](i);
        }
        for (size_type j = 0; j < s; ++j)
            Mmat(j, k) = mtl::dot<Accumulator, value_type>(P[j], dR[k]);

        if (iter.finished(r)) return iter;
        ++iter;
    }
    for (size_type j = 0; j < s; ++j)
        m(j) = mtl::dot<Accumulator, value_type>(P[j], r);

    // Solve Mmat c = m by Gaussian elimination with partial pivoting (s is
    // small). Returns false if Mmat is singular.
    auto solve_small = [&]() {
        for (size_type i = 0; i < s; ++i) {
            f(i) = m(i);
            for (size_type j = 0; j < s; ++j)
                Mcopy(i, j) = Mmat(i, j);
        }
        for (size_type k = 0; k < s; ++k) {
            size_type pivot = k;
            value_type maxval = abs(Mcopy(k, k));
            for (size_type i = k + 1; i < s; ++i) {
                if (abs(Mcopy(i, k)) > maxval) {
                    maxval = abs(Mcopy(i, k));
                    pivot = i;
                }
            }
            if (pivot != k) {
                for (size_type j = 0; j < s; ++j)
                    std::swap(Mcopy(k, j), Mcopy(pivot, j));
                std::swap(f(k), f(pivot));
            }
            if (abs(Mcopy(k, k)) < value_type(1e-300))
                return false;
            for (size_type i = k + 1; i < s; ++i) {
                value_type factor = Mcopy(i, k) / Mcopy(k, k);
                for (size_type j = k + 1; j < s; ++j)
                    Mcopy(i, j) -= factor * Mcopy(k, j);
                f(i) -= factor * f(k);
            }
        }
        for (size_type ii = 0; ii < s; ++ii) {
            size_type i = s - 1 - ii;
            value_type sum = math::zero<value_type>();
            for (size_type j = i + 1; j < s; ++j)
                sum += Mcopy(i, j) * c(j);
            c(i) = (f(i) - sum) / Mcopy(i, i);
        }
        return true;
    };

    // Main loop: cycles of s + 1 steps, one omega per cycle.
    size_type oldest = 0;
    while (!iter.finished(r)) {
        for (size_type step = 0; step <= s; ++step) {
            if (!solve_small()) {
                iter.fail(2, "idr_s breakdown: singular M");
                return iter;
            }

            // v = r - dR c, which is orthogonal to P; vhat = M^-1 v.
            for (size_type i = 0; i < n; ++i) {
                v(i) = r(i);
                for (size_type k = 0; k < s; ++k)
                    v(i) -= c(k) * dR[k](i);
            }
            M.solve(vhat, v);

            // The first step of a cycle picks omega from t = A vhat.
            if (step == 0) {
                apply_operator<Accumulator>(A, vhat, t);
                min_residual_omega(v);
            }

            // dX_new = omega vhat - dX c (into dxn: dX[oldest] is read here).
            for (size_type i = 0; i < n; ++i) {
                value_type dx = omega * vhat(i);
                for (size_type k = 0; k < s; ++k)
                    dx -= c(k) * dX[k](i);
                dxn(i) = dx;
            }
            // dR_new = -A dX_new: at the first step that is v - r - omega t
            // (A dX = -dR holds for every stored pair), after it one product.
            if (step == 0) {
                for (size_type i = 0; i < n; ++i)
                    dR[oldest](i) = v(i) - r(i) - omega * t(i);
            } else {
                apply_operator<Accumulator>(A, dxn, t);
                for (size_type i = 0; i < n; ++i)
                    dR[oldest](i) = -t(i);
            }
            for (size_type i = 0; i < n; ++i) {
                dX[oldest](i) = dxn(i);
                x(i) += dxn(i);
                r(i) += dR[oldest](i);
            }

            // Mmat's column `oldest` and m = P^T r, updated by P^T dR_new.
            for (size_type j = 0; j < s; ++j) {
                const value_type dm = mtl::dot<Accumulator, value_type>(P[j], dR[oldest]);
                Mmat(j, oldest) = dm;
                m(j) += dm;
            }
            oldest = (oldest + 1) % s;

            ++iter;
            if (iter.finished(r)) return iter;
        }
    }

    return iter;
}

} // namespace mtl::itl
