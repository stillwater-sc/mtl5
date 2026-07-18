#pragma once
// MTL5 -- Arnoldi iteration for a few eigenpairs of a GENERAL (non-symmetric)
// linear operator. Matrix-free (only needs A * x). Builds an orthonormal Krylov
// basis and an upper Hessenberg projection, whose eigenpairs (via the dense
// general eigen) give the Ritz pairs. Full reorthogonalization for robustness.
#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <vector>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/operation/dot.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/operation/eigenvalue.hpp>
#include <mtl/itl/eigen/eigen_common.hpp>

namespace mtl::itl {

/// Compute `k` eigenpairs of a general operator `A` by the Arnoldi method.
/// `v0` is the starting vector; `which` selects which Ritz values to keep
/// (magnitude selectors, or algebraic selectors on the real part); `subspace`
/// is the Krylov dimension to build (default: a modest multiple of k, capped at
/// n); `tol` flags convergence via the Ritz residual estimate.
///
/// Returns k complex Ritz values (ordered per `which`) and their complex Ritz
/// vectors (columns). A is any LinearOperator.
template <typename LinearOp, typename T>
ritz_pairs<std::complex<T>> arnoldi(const LinearOp& A, vec::dense_vector<T> v0,
                                    std::size_t k,
                                    eigen_which which = eigen_which::largest_magnitude,
                                    std::size_t subspace = 0, T tol = T(1e-8)) {
    using std::abs;
    using complex_type = std::complex<T>;
    using size_type = typename vec::dense_vector<T>::size_type;
    const size_type n = v0.size();

    ritz_pairs<complex_type> out;
    if (n == 0 || k == 0) return out;

    size_type m = subspace ? subspace
                           : std::max<size_type>(2 * k + 20, 20);
    if (m > n) m = n;
    if (k > m) k = m;

    const T breakdown = std::sqrt(std::numeric_limits<T>::epsilon());

    mat::dense2D<T> V(n, m);
    mat::dense2D<T> H(m, m);            // upper Hessenberg projection
    for (size_type i = 0; i < m; ++i)
        for (size_type j = 0; j < m; ++j)
            H(i, j) = T(0);

    T nv = mtl::two_norm(v0);
    if (nv == T(0)) { v0(0) = T(1); nv = T(1); }
    vec::dense_vector<T> q(n), w(n);
    for (size_type i = 0; i < n; ++i) q(i) = v0(i) / nv;

    size_type built = 0;
    T h_next = T(0);   // ||r_m||: subdiagonal past the last built column
    for (size_type j = 0; j < m; ++j) {
        for (size_type i = 0; i < n; ++i) V(i, j) = q(i);

        w = ev_matvec(A, q);

        // Modified Gram-Schmidt with a second pass for numerical stability.
        for (int pass = 0; pass < 2; ++pass)
            for (size_type c = 0; c <= j; ++c) {
                T s = T(0);
                for (size_type i = 0; i < n; ++i) s += V(i, c) * w(i);
                H(c, j) += s;
                for (size_type i = 0; i < n; ++i) w(i) -= s * V(i, c);
            }

        built = j + 1;
        T hn = mtl::two_norm(w);
        h_next = hn;
        if (hn <= breakdown) break;         // invariant subspace found
        if (j + 1 < m) { H(j + 1, j) = hn; for (size_type i = 0; i < n; ++i) q(i) = w(i) / hn; }
    }
    m = built;

    // Trim H to the built m x m block if we stopped early.
    mat::dense2D<T> Hm(m, m);
    for (size_type i = 0; i < m; ++i)
        for (size_type j = 0; j < m; ++j)
            Hm(i, j) = H(i, j);

    // Dense general eigensolve of the small Hessenberg block.
    auto es = mtl::eigen(Hm);
    const auto& mu = es.eigenvalues;    // size m, complex
    const auto& S  = es.eigenvectors;   // m x m, complex, column i = eigvec of Hm

    // Order the m Ritz values per `which`, then take the first k.
    std::vector<size_type> order(m);
    for (size_type i = 0; i < m; ++i) order[i] = i;
    switch (which) {
        case eigen_which::largest_magnitude:
            std::sort(order.begin(), order.end(),
                      [&](size_type a2, size_type b2){ return std::abs(mu(a2)) > std::abs(mu(b2)); });
            break;
        case eigen_which::smallest_magnitude:
            std::sort(order.begin(), order.end(),
                      [&](size_type a2, size_type b2){ return std::abs(mu(a2)) < std::abs(mu(b2)); });
            break;
        case eigen_which::largest_algebraic:
            std::sort(order.begin(), order.end(),
                      [&](size_type a2, size_type b2){ return mu(a2).real() > mu(b2).real(); });
            break;
        case eigen_which::smallest_algebraic:
            std::sort(order.begin(), order.end(),
                      [&](size_type a2, size_type b2){ return mu(a2).real() < mu(b2).real(); });
            break;
    }

    out.values.change_dim(k);
    out.vectors = mat::dense2D<complex_type>(n, k);
    out.subspace = static_cast<int>(m);
    bool all_conv = true;
    for (size_type w2 = 0; w2 < k; ++w2) {
        size_type c = order[w2];
        out.values(w2) = mu(c);
        // Ritz vector y = V(:,0..m-1) * S(:,c)  (real V times complex S).
        for (size_type i = 0; i < n; ++i) {
            complex_type acc(0);
            for (size_type r = 0; r < m; ++r) acc += complex_type(V(i, r)) * S(r, c);
            out.vectors(i, w2) = acc;
        }
        // Ritz residual estimate: |h_next| * |last component of S(:,c)|.
        T rest = abs(h_next) * std::abs(S(m - 1, c));
        if (rest > tol * (std::abs(mu(c)) + tol)) all_conv = false;
    }
    out.converged = all_conv;
    return out;
}

} // namespace mtl::itl
