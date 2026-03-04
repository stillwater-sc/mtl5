#pragma once
// MTL5 — GMRES (Generalized Minimum Residual) solver with restart
// Left-preconditioned, Modified Gram-Schmidt, Givens rotations for QR
#include <cmath>
#include <vector>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/operation/dot.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/operation/givens.hpp>
#include <mtl/math/identity.hpp>

namespace mtl::itl {

namespace detail {

/// Single GMRES cycle (inner iteration) — up to kmax Arnoldi steps.
/// Returns 0 on convergence, 1 if kmax exhausted (restart needed).
template <typename LinearOp, typename VecX, typename VecB,
          typename PC, typename Iter>
int gmres_inner(const LinearOp& A, VecX& x, const VecB& b,
                const PC& M, Iter& iter, int kmax) {
    using value_type = typename VecX::value_type;
    using size_type  = typename VecX::size_type;
    const size_type n = x.size();

    // r = M^{-1}(b - A*x)
    vec::dense_vector<value_type> r0(n), r(n);
    auto Ax = A * x;
    for (size_type i = 0; i < n; ++i)
        r0(i) = b(i) - Ax(i);
    M.solve(r, r0);

    value_type beta = mtl::two_norm(r);
    if (iter.finished(beta))
        return iter.error_code();

    // Arnoldi basis vectors V[0..kmax]
    std::vector<vec::dense_vector<value_type>> V(kmax + 1, vec::dense_vector<value_type>(n));
    for (size_type i = 0; i < n; ++i)
        V[0](i) = r(i) / beta;

    // Hessenberg matrix H: (kmax+1) x kmax
    mat::dense2D<value_type> H(kmax + 1, kmax);
    for (int i = 0; i <= kmax; ++i)
        for (int j = 0; j < kmax; ++j)
            H(i, j) = math::zero<value_type>();

    // RHS vector g for least-squares, Givens rotation storage
    vec::dense_vector<value_type> g(kmax + 1, math::zero<value_type>());
    g(0) = beta;
    vec::dense_vector<value_type> cs(kmax, math::zero<value_type>());
    vec::dense_vector<value_type> sn(kmax, math::zero<value_type>());

    int k = 0;
    for (; k < kmax; ++k) {
        // w = M^{-1}(A * V[k])
        vec::dense_vector<value_type> w_tmp(n), w(n);
        auto Avk = A * V[k];
        for (size_type i = 0; i < n; ++i)
            w_tmp(i) = Avk(i);
        M.solve(w, w_tmp);

        // Modified Gram-Schmidt
        for (int j = 0; j <= k; ++j) {
            H(j, k) = mtl::dot(V[j], w);
            for (size_type i = 0; i < n; ++i)
                w(i) -= H(j, k) * V[j](i);
        }

        H(k + 1, k) = mtl::two_norm(w);

        // Check for breakdown
        if (H(k + 1, k) == math::zero<value_type>()) {
            // Lucky breakdown: exact solution in Krylov subspace
            // Apply existing stored rotations and solve
            break;
        }

        for (size_type i = 0; i < n; ++i)
            V[k + 1](i) = w(i) / H(k + 1, k);

        // Apply previously stored Givens rotations to column k
        for (int j = 0; j < k; ++j)
            mtl::apply_stored_rotation(H, cs, sn, static_cast<size_type>(j), static_cast<size_type>(k));

        // Compute new Givens rotation and apply
        mtl::apply_givens_rotation(H, g, cs, sn, static_cast<size_type>(k));

        ++iter;

        // Check convergence using |g(k+1)|
        using std::abs;
        if (iter.finished(abs(g(k + 1)))) {
            ++k; // include this column in the solution
            break;
        }
    }

    // Back-substitution: solve upper triangular H*y = g
    int m = k; // number of columns used
    vec::dense_vector<value_type> y(m, math::zero<value_type>());
    for (int i = m - 1; i >= 0; --i) {
        y(i) = g(i);
        for (int j = i + 1; j < m; ++j)
            y(i) -= H(i, j) * y(j);
        y(i) /= H(i, i);
    }

    // Update solution: x += V * y
    for (int j = 0; j < m; ++j)
        for (size_type i = 0; i < n; ++i)
            x(i) += y(j) * V[j](i);

    return iter.error_code();
}

} // namespace detail

/// GMRES with restart.
/// Solves A*x = b with left preconditioner M.
/// restart: maximum Krylov subspace dimension before restart (default 30).
template <typename LinearOp, typename VecX, typename VecB,
          typename PC, typename Iter>
int gmres(const LinearOp& A, VecX& x, const VecB& b,
          const PC& M, Iter& iter, int restart = 30) {
    while (!iter.is_finished()) {
        detail::gmres_inner(A, x, b, M, iter, restart);
    }
    return iter;
}

} // namespace mtl::itl
