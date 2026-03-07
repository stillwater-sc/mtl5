#pragma once
// MTL5 -- IDR(s) solver (Induced Dimension Reduction)
// Sonneveld & van Gijzen (2008). Modern Krylov solver for non-symmetric systems.
// Parameter s controls shadow space dimension (default 4).
#include <cmath>
#include <random>
#include <vector>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/operation/dot.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/math/identity.hpp>

namespace mtl::itl {

/// IDR(s) solver for non-symmetric systems A*x = b.
/// s = shadow space dimension (larger s -> faster convergence, more memory).
template <typename LinearOp, typename VecX, typename VecB,
          typename PC, typename Iter>
int idr_s(const LinearOp& A, VecX& x, const VecB& b, const PC& M, Iter& iter,
          typename VecX::size_type s = 4) {
    using value_type = typename VecX::value_type;
    using size_type  = typename VecX::size_type;
    using std::abs;
    const size_type n = x.size();

    // Random shadow space P (n x s)
    std::mt19937 gen(42);
    std::normal_distribution<value_type> dist(0.0, 1.0);

    std::vector<vec::dense_vector<value_type>> P(s, vec::dense_vector<value_type>(n));
    for (size_type j = 0; j < s; ++j)
        for (size_type i = 0; i < n; ++i)
            P[j](i) = dist(gen);

    // Workspace
    vec::dense_vector<value_type> r(n), v(n), t(n);
    std::vector<vec::dense_vector<value_type>> dR(s, vec::dense_vector<value_type>(n));
    std::vector<vec::dense_vector<value_type>> dX(s, vec::dense_vector<value_type>(n));

    // r = b - A*x
    auto Ax = A * x;
    for (size_type i = 0; i < n; ++i)
        r(i) = b(i) - Ax(i);

    // M matrix (s x s) and helper vectors
    mat::dense2D<value_type> Mmat(s, s);
    vec::dense_vector<value_type> f(s), c(s);

    // Initial dR: solve for s initial directions
    value_type omega = value_type(1);

    for (size_type k = 0; k < s; ++k) {
        // v = M^{-1} r
        M.solve(v, r);
        auto Av = A * v;
        for (size_type i = 0; i < n; ++i)
            t(i) = Av(i);

        // omega = dot(t,r) / dot(t,t)
        value_type tt = mtl::dot(t, t);
        if (tt != value_type(0))
            omega = mtl::dot(t, r) / tt;

        // dX[k] = omega * v
        // dR[k] = -omega * t (= r_new - r_old in effect)
        for (size_type i = 0; i < n; ++i) {
            dX[k](i) = omega * v(i);
            dR[k](i) = r(i) - omega * t(i) - r(i); // = -omega * t
        }

        // But we actually want: dR[k] = r_new - r, so:
        // r_new = r + dR[k]
        // Just update: dR[k] = -omega * A * v, x += omega * v, r -= omega * t
        for (size_type i = 0; i < n; ++i) {
            dR[k](i) = -omega * t(i);
            dX[k](i) = omega * v(i);
            x(i) += dX[k](i);
            r(i) += dR[k](i);
        }

        if (iter.finished(r)) return iter;
        ++iter;
    }

    // Build M = P^T * dR
    for (size_type i = 0; i < s; ++i)
        for (size_type j = 0; j < s; ++j)
            Mmat(i, j) = mtl::dot(P[i], dR[j]);

    // Main IDR(s) loop
    size_type oldest = 0;

    while (!iter.finished(r)) {
        // f = P^T * r
        for (size_type i = 0; i < s; ++i)
            f(i) = mtl::dot(P[i], r);

        // Solve M * c = f via simple Gaussian elimination (s is small)
        // Copy M and f for in-place solve
        mat::dense2D<value_type> Mcopy(s, s);
        vec::dense_vector<value_type> fcopy(s);
        for (size_type i = 0; i < s; ++i) {
            fcopy(i) = f(i);
            for (size_type j = 0; j < s; ++j)
                Mcopy(i, j) = Mmat(i, j);
        }

        // Forward elimination
        for (size_type k = 0; k < s; ++k) {
            // Partial pivot
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
                std::swap(fcopy(k), fcopy(pivot));
            }
            if (abs(Mcopy(k, k)) < value_type(1e-300)) {
                iter.fail(2, "idr_s breakdown: singular M");
                return iter;
            }
            for (size_type i = k + 1; i < s; ++i) {
                value_type factor = Mcopy(i, k) / Mcopy(k, k);
                for (size_type j = k + 1; j < s; ++j)
                    Mcopy(i, j) -= factor * Mcopy(k, j);
                fcopy(i) -= factor * fcopy(k);
            }
        }
        // Back substitution
        for (size_type ii = 0; ii < s; ++ii) {
            size_type i = s - 1 - ii;
            value_type sum = math::zero<value_type>();
            for (size_type j = i + 1; j < s; ++j)
                sum += Mcopy(i, j) * c(j);
            c(i) = (fcopy(i) - sum) / Mcopy(i, i);
        }

        // v = r - sum(c[k] * dR[k])
        for (size_type i = 0; i < n; ++i) {
            v(i) = r(i);
            for (size_type k = 0; k < s; ++k)
                v(i) -= c(k) * dR[k](i);
        }

        // Preconditioned: t = M^{-1} v
        vec::dense_vector<value_type> vhat(n);
        M.solve(vhat, v);

        // t = A * vhat
        auto Avhat = A * vhat;
        for (size_type i = 0; i < n; ++i)
            t(i) = Avhat(i);

        // omega = dot(t, v) / dot(t, t)
        value_type tt = mtl::dot(t, t);
        if (tt != value_type(0))
            omega = mtl::dot(t, v) / tt;

        // dR[oldest] = -sum(c[k]*dR[k]) - omega*t  (= new_r - old_r)
        // dX[oldest] = -sum(c[k]*dX[k]) + omega*vhat
        // But first compute dX update
        vec::dense_vector<value_type> dx_new(n);
        for (size_type i = 0; i < n; ++i) {
            dx_new(i) = omega * vhat(i);
            for (size_type k = 0; k < s; ++k)
                dx_new(i) -= c(k) * dX[k](i);
        }

        vec::dense_vector<value_type> dr_new(n);
        for (size_type i = 0; i < n; ++i)
            dr_new(i) = v(i) - omega * t(i) - r(i);

        // Actually: new_r = v - omega * t = r - sum(c*dR) - omega*t
        // So dR = new_r - r = -sum(c*dR) - omega*t
        for (size_type i = 0; i < n; ++i) {
            dR[oldest](i) = -(r(i) - v(i)) - omega * t(i);
            dX[oldest](i) = dx_new(i);
        }

        // Update x and r
        for (size_type i = 0; i < n; ++i) {
            x(i) += dX[oldest](i);
            r(i) += dR[oldest](i);
        }

        // Update M = P^T * dR (only column 'oldest')
        for (size_type i = 0; i < s; ++i)
            Mmat(i, oldest) = mtl::dot(P[i], dR[oldest]);

        oldest = (oldest + 1) % s;
        ++iter;
    }

    return iter;
}

} // namespace mtl::itl
