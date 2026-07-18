#pragma once
// MTL5 -- Lanczos iteration for a few extremal eigenpairs of a SYMMETRIC
// linear operator. Matrix-free (only needs A * x). Builds an orthonormal Krylov
// basis and a symmetric tridiagonal projection, whose eigenpairs (via the dense
// eigen_symmetric) give the Ritz pairs. Full reorthogonalization keeps the basis
// orthogonal for a robust reference implementation.
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/operation/dot.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/operation/eigenvalue_symmetric.hpp>
#include <mtl/itl/eigen/eigen_common.hpp>

namespace mtl::itl {

/// Compute `k` extremal eigenpairs of a symmetric operator `A` by the Lanczos
/// method. `v0` is the starting vector; `which` selects the end of the spectrum;
/// `subspace` is the Krylov dimension to build (default: a modest multiple of k,
/// capped at n -- larger is more accurate); `tol` is the Ritz residual
/// tolerance used to flag convergence.
///
/// Returns k Ritz values (ordered per `which`) and their Ritz vectors (columns).
/// A is any LinearOperator; it MUST be symmetric for the tridiagonal projection
/// to be meaningful.
template <typename LinearOp, typename T>
ritz_pairs<T> lanczos(const LinearOp& A, vec::dense_vector<T> v0, std::size_t k,
                      eigen_which which = eigen_which::largest_algebraic,
                      std::size_t subspace = 0, T tol = T(1e-8)) {
    using std::abs;
    using size_type = typename vec::dense_vector<T>::size_type;
    const size_type n = v0.size();

    ritz_pairs<T> out;
    if (n == 0 || k == 0) return out;

    size_type m = subspace ? subspace
                           : std::max<size_type>(2 * k + 20, 20);
    if (m > n) m = n;
    if (k > m) k = m;

    const T breakdown = std::sqrt(std::numeric_limits<T>::epsilon());

    // Orthonormal Lanczos basis (columns) and tridiagonal entries.
    mat::dense2D<T> V(n, m);
    std::vector<T> alpha(m, T(0));
    std::vector<T> beta(m, T(0));   // beta[j] links columns j-1 and j; beta[0] unused

    T nv = mtl::two_norm(v0);
    if (nv == T(0)) { v0(0) = T(1); nv = T(1); }
    vec::dense_vector<T> q(n), q_prev(n, T(0)), w(n);
    for (size_type i = 0; i < n; ++i) q(i) = v0(i) / nv;

    size_type built = 0;
    T beta_next = T(0);   // ||r_m||: residual norm past the last built column
    for (size_type j = 0; j < m; ++j) {
        for (size_type i = 0; i < n; ++i) V(i, j) = q(i);

        w = ev_matvec(A, q);
        if (j > 0)
            for (size_type i = 0; i < n; ++i) w(i) -= beta[j] * q_prev(i);

        T a = mtl::dot(q, w);
        alpha[j] = a;
        for (size_type i = 0; i < n; ++i) w(i) -= a * q(i);

        // Full reorthogonalization (two passes) against all built columns.
        for (int pass = 0; pass < 2; ++pass)
            for (size_type c = 0; c <= j; ++c) {
                T s = T(0);
                for (size_type i = 0; i < n; ++i) s += V(i, c) * w(i);
                for (size_type i = 0; i < n; ++i) w(i) -= s * V(i, c);
            }

        built = j + 1;
        T bn = mtl::two_norm(w);
        beta_next = bn;
        if (bn <= breakdown) break;         // invariant subspace found
        if (j + 1 < m) beta[j + 1] = bn;
        for (size_type i = 0; i < n; ++i) { q_prev(i) = q(i); q(i) = w(i) / bn; }
    }
    m = built;

    // Symmetric tridiagonal projection T_m.
    mat::dense2D<T> Tm(m, m);
    for (size_type i = 0; i < m; ++i)
        for (size_type j = 0; j < m; ++j)
            Tm(i, j) = T(0);
    for (size_type j = 0; j < m; ++j) Tm(j, j) = alpha[j];
    for (size_type j = 1; j < m; ++j) { Tm(j, j - 1) = beta[j]; Tm(j - 1, j) = beta[j]; }

    // Dense symmetric eigensolve of the small tridiagonal (theta ascending).
    auto es = mtl::eigen_symmetric(Tm);
    const auto& theta = es.eigenvalues;    // size m, ascending
    const auto& S     = es.eigenvectors;   // m x m, column i = eigvec of Tm

    // Order the m Ritz values per `which`, then take the first k.
    std::vector<size_type> order(m);
    for (size_type i = 0; i < m; ++i) order[i] = i;
    auto by = [&](eigen_which w2) {
        switch (w2) {
            case eigen_which::largest_algebraic:
                std::sort(order.begin(), order.end(),
                          [&](size_type a2, size_type b2){ return theta(a2) > theta(b2); });
                break;
            case eigen_which::smallest_algebraic:
                std::sort(order.begin(), order.end(),
                          [&](size_type a2, size_type b2){ return theta(a2) < theta(b2); });
                break;
            case eigen_which::largest_magnitude:
                std::sort(order.begin(), order.end(),
                          [&](size_type a2, size_type b2){ return abs(theta(a2)) > abs(theta(b2)); });
                break;
            case eigen_which::smallest_magnitude:
                std::sort(order.begin(), order.end(),
                          [&](size_type a2, size_type b2){ return abs(theta(a2)) < abs(theta(b2)); });
                break;
        }
    };
    by(which);

    out.values.change_dim(k);
    out.vectors = mat::dense2D<T>(n, k);
    out.subspace = static_cast<int>(m);
    bool all_conv = true;
    for (size_type w2 = 0; w2 < k; ++w2) {
        size_type c = order[w2];
        out.values(w2) = theta(c);
        // Ritz vector y = V(:,0..m-1) * S(:,c).
        for (size_type i = 0; i < n; ++i) {
            T acc = T(0);
            for (size_type r = 0; r < m; ++r) acc += V(i, r) * S(r, c);
            out.vectors(i, w2) = acc;
        }
        // Ritz residual estimate: |beta_next| * |last component of S(:,c)|.
        T rest = abs(beta_next) * abs(S(m - 1, c));
        if (rest > tol * (abs(theta(c)) + tol)) all_conv = false;
    }
    out.converged = all_conv;
    return out;
}

} // namespace mtl::itl
