#pragma once
// MTL5 -- GMRES (Generalized Minimum Residual) solver with restart
// Left-preconditioned, Modified Gram-Schmidt, Givens rotations for QR
#include <mtl/detail/exceptions.hpp>
#include <mtl/concepts/vector.hpp>   // FieldVector (#503)
#include <cmath>
#include <stdexcept>
#include <vector>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/operation/dot.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/operation/givens.hpp>
#include <mtl/operation/mult.hpp>
#include <mtl/itl/krylov/apply_operator.hpp>
#include <mtl/itl/krylov/scaled_rhs.hpp>
#include <mtl/math/identity.hpp>

namespace mtl::itl {

namespace detail {

/// gmres's vectors, allocated once per solve rather than per restart cycle
/// and per Arnoldi step: kmax + 1 basis vectors were zero-filled on every
/// cycle (about 250 MB at n = 1e6 and restart 30), and two vectors were
/// allocated on every step (issue 924).
template <typename Value>
struct gmres_workspace {
    gmres_workspace(std::size_t n, int kmax)
        : V(static_cast<std::size_t>(kmax) + 1, vec::dense_vector<Value>(n)),
          H(kmax + 1, kmax), g(kmax + 1), cs(kmax), sn(kmax), y(kmax),
          r0(n), r(n), w_tmp(n), w(n) {}

    std::vector<vec::dense_vector<Value>> V;   // Arnoldi basis V[0..kmax]
    mat::dense2D<Value> H;                     // Hessenberg, (kmax+1) x kmax
    vec::dense_vector<Value> g, cs, sn, y;     // least-squares RHS, rotations, solution
    vec::dense_vector<Value> r0, r, w_tmp, w;
};

/// The identity, for a gmres_inner called without a scaled right-hand side.
struct gmres_no_unscale {
    template <typename T>
    T operator()(const T& r) const { return r; }
};

/// One GMRES cycle in the workspace `ws`, for the right-hand side b.
/// `unscale` maps a residual norm to the units of the iteration object
/// (scaled_rhs.hpp). See gmres_inner for the return value.
template <typename Accumulator, typename LinearOp, typename VecX, typename VecB,
          typename PC, typename Iter, typename Value, typename Unscale>
int gmres_cycle(const LinearOp& A, VecX& x, const VecB& b, const PC& M, Iter& iter,
                int kmax, gmres_workspace<Value>& ws, const Unscale& unscale) {
    using value_type = Value;
    using size_type  = typename VecX::size_type;
    const size_type n = x.size();
    const auto zero = math::zero<value_type>();
    auto& V = ws.V;
    auto& H = ws.H;
    auto& g = ws.g;

    // r = M^{-1}(b - A*x); w_tmp holds A x
    apply_operator<Accumulator>(A, x, ws.w_tmp);
    for (size_type i = 0; i < n; ++i)
        ws.r0(i) = b(i) - ws.w_tmp(i);
    M.solve(ws.r, ws.r0);

    // Real for a complex system (#853): the iteration takes a magnitude.
    const auto beta = mtl::two_norm(ws.r);
    if (iter.finished(unscale(beta)))
        return 0;

    for (size_type i = 0; i < n; ++i)
        V[0](i) = ws.r(i) / beta;

    // g(k + 1) is read by the rotation of column k, so g starts at zero; each
    // column of H is cleared before its Arnoldi step writes it.
    for (int i = 0; i <= kmax; ++i)
        g(i) = zero;
    g(0) = beta;

    int k = 0;
    bool breakdown = false;
    for (; k < kmax; ++k) {
        for (int i = 0; i <= kmax; ++i)
            H(i, k) = zero;

        // w = M^{-1}(A * V[k])
        apply_operator<Accumulator>(A, V[k], ws.w_tmp);
        M.solve(ws.w, ws.w_tmp);

        // Modified Gram-Schmidt
        for (int j = 0; j <= k; ++j) {
            H(j, k) = mtl::dot<Accumulator, value_type>(V[j], ws.w);
            for (size_type i = 0; i < n; ++i)
                ws.w(i) -= H(j, k) * V[j](i);
        }

        H(k + 1, k) = mtl::two_norm(ws.w);

        breakdown = (H(k + 1, k) == zero);
        if (!breakdown) {
            for (size_type i = 0; i < n; ++i)
                V[k + 1](i) = ws.w(i) / H(k + 1, k);
        }

        // Apply previously stored Givens rotations to column k
        for (int j = 0; j < k; ++j)
            mtl::apply_stored_rotation(H, ws.cs, ws.sn, static_cast<size_type>(j), static_cast<size_type>(k));

        // Compute new Givens rotation and apply
        mtl::apply_givens_rotation(H, g, ws.cs, ws.sn, static_cast<size_type>(k));

        ++iter;

        if (breakdown) {
            // Lucky breakdown: exact solution lies in the Krylov subspace
            // spanned by V[0..k]. Column k's rotation was just applied
            // above, so it's valid to include in back-substitution --
            // unlike the old code, which broke out before applying the
            // rotation or incrementing iter, leaving x and iter unchanged
            // and causing gmres()'s outer restart loop to spin forever.
            ++k;
            break;
        }

        // Check convergence using |g(k+1)|
        using std::abs;
        if (iter.finished(unscale(abs(g(k + 1))))) {
            ++k; // include this column in the solution
            break;
        }
    }

    // Back-substitution: solve upper triangular H*y = g. A zero H(i,i) is
    // possible only in the last column of a breakdown, where the Krylov space
    // is invariant and A is singular on it: there is nothing left to gain
    // from it. Dividing by it wrote NaN into x, for A = [[1,1],[1,1]] and
    // b = (1,0) (issue 924). The column is left out (its y is 0, a
    // least-squares minimizer), and a cycle that has not converged ends in a
    // breakdown rather than a restart that finds the same space again.
    int m = k; // number of columns used
    bool singular = false;
    if (m > 0 && H(m - 1, m - 1) == zero) {
        --m;
        singular = true;
    }
    auto& y = ws.y;
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

    // |g(m)| is the residual estimate without the dropped column (its rotation
    // had nothing to rotate, so g(m + 1) is 0). A consistent system can still
    // have converged.
    if (singular && !iter.is_finished()) {
        using std::abs;
        if (!iter.finished(unscale(abs(g(m)))))
            iter.fail(2, "gmres breakdown: A is singular on an invariant Krylov space");
    }

    return iter.is_finished() ? 0 : 1;
}

/// Single GMRES cycle (inner iteration) -- up to kmax Arnoldi steps.
/// Returns 0 once `iter` is finished (converged, at its limit, or broken
/// down: its error code says which), 1 if kmax was exhausted first (restart
/// needed). It does not call iter.error_code(), which a cyclic_iteration
/// answers by printing its summary (issue 866).
template <typename Accumulator = void,
          typename LinearOp, typename VecX, typename VecB,
          typename PC, typename Iter>
    requires FieldVector<VecX>
int gmres_inner(const LinearOp& A, VecX& x, const VecB& b,
                const PC& M, Iter& iter, int kmax) {
    gmres_workspace<typename VecX::value_type> ws(x.size(), kmax);
    return gmres_cycle<Accumulator>(A, x, b, M, iter, kmax, ws, gmres_no_unscale{});
}

} // namespace detail

/// GMRES with restart.
/// Solves A*x = b with left preconditioner M.
/// restart: maximum Krylov subspace dimension before restart (default 30).
/// A is a Matrix or any LinearOperator (A * x). With a matrix-free A the
/// Accumulator reaches only the dot products (apply_operator.hpp, #278).
template <typename Accumulator = void,
          typename LinearOp, typename VecX, typename VecB,
          typename PC, typename Iter>
    requires FieldVector<VecX>
int gmres(const LinearOp& A, VecX& x, const VecB& b,
          const PC& M, Iter& iter, int restart = 30) {
    // restart = 0 made no step and the loop below spun forever; a negative
    // one sized the Krylov basis from a negative count (issue 866).
    if (restart < 1)
        ::mtl::detail::throw_or_abort(std::invalid_argument("gmres: restart must be at least 1"));
    // b's own copy, read again on every restart, so x may alias b; scaled
    // near 1 when ||b|| is extreme (scaled_rhs.hpp, issue 924).
    detail::scaled_rhs<VecX, VecB> rhs(x, b, true);
    detail::gmres_workspace<typename VecX::value_type> ws(x.size(), restart);
    auto unscale = [&](const auto& r) { return rhs.unscale(r); };
    // At least one cycle: it tests the residual first, so an iteration object
    // still finished from an earlier solve is not taken as done (issue 866).
    do {
        detail::gmres_cycle<Accumulator>(A, x, rhs.b(), M, iter, restart, ws, unscale);
    } while (!iter.is_finished());
    return iter;
}

} // namespace mtl::itl
