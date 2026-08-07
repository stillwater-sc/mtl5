#pragma once
// MTL5 -- Singular Value Decomposition via one-sided Jacobi / iterative QR
// Decomposes A = U * S * V^T where S is diagonal (singular values).
// Optional LAPACK dispatch when MTL5_HAS_LAPACK is defined and types qualify.
#include <cmath>
#include <algorithm>
#include <cassert>
#include <vector>
#include <limits>
#include <mtl/concepts/matrix.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/operation/qr.hpp>
#include <mtl/operation/trans.hpp>
#include <mtl/math/identity.hpp>
#include <mtl/interface/dispatch_traits.hpp>
#ifdef MTL5_HAS_LAPACK
#include <mtl/interface/lapack.hpp>
#endif

namespace mtl {

/// SVD: decompose A (m x n) = U * S * V^T
/// U is m x m orthogonal, S is m x n diagonal (stored as m x n matrix), V is n x n orthogonal.
/// Singular values are non-negative and returned in descending order.
///
/// Uses one-sided Jacobi: the columns are orthogonalized by plane rotations,
/// which are accumulated into V, and the singular values are the resulting
/// column norms.
///
/// `tol` is the relative threshold for treating a column pair as already
/// orthogonal, i.e. the accuracy the singular values are driven to. It
/// defaults to the machine epsilon of the value type; the previous default of
/// 1e-10 would cap accuracy several orders of magnitude short of what the
/// method achieves. Callers wanting the old, looser behaviour can pass 1e-10
/// explicitly, and a larger tolerance trades accuracy for fewer sweeps.
template <Matrix M>
void svd(const M& A,
         mat::dense2D<typename M::value_type>& U,
         mat::dense2D<typename M::value_type>& S,
         mat::dense2D<typename M::value_type>& V,
         typename M::value_type tol =
             std::numeric_limits<typename M::value_type>::epsilon()) {
    using value_type = typename M::value_type;
    using size_type  = typename M::size_type;
    using std::abs;
    using std::sqrt;

    const size_type m = A.num_rows();
    const size_type n = A.num_cols();
    const size_type mn = std::min(m, n);

#ifdef MTL5_HAS_LAPACK
    // No orientation guard: the branch copies A into a column-major buffer via
    // the (i,j) accessor and writes U/S/V back via (i,j), so it is correct for
    // row-major and column-major M alike (see #417).
    if constexpr (interface::BlasDenseMatrix<M>) {
        // LAPACK gesdd: A_copy is overwritten, returns U, S_vec, VT
        U.change_dim(m, m);
        S.change_dim(m, n);
        V.change_dim(n, n);

        // Work on a copy (LAPACK overwrites input)
        std::vector<value_type> A_copy(m * n);
        for (size_type i = 0; i < m; ++i)
            for (size_type j = 0; j < n; ++j)
                A_copy[j * m + i] = A(i, j);  // column-major copy

        std::vector<value_type> S_vec(mn);
        std::vector<value_type> U_data(m * m);
        std::vector<value_type> VT_data(n * n);
        std::vector<int> iwork(8 * mn);

        // Workspace query
        value_type work_opt;
        interface::lapack::gesdd('A', static_cast<int>(m), static_cast<int>(n),
            A_copy.data(), static_cast<int>(m), S_vec.data(),
            U_data.data(), static_cast<int>(m),
            VT_data.data(), static_cast<int>(n),
            &work_opt, -1, iwork.data());
        int lwork = static_cast<int>(work_opt);
        std::vector<value_type> work(lwork);
        interface::lapack::gesdd('A', static_cast<int>(m), static_cast<int>(n),
            A_copy.data(), static_cast<int>(m), S_vec.data(),
            U_data.data(), static_cast<int>(m),
            VT_data.data(), static_cast<int>(n),
            work.data(), lwork, iwork.data());

        // Copy results: U from column-major U_data
        for (size_type i = 0; i < m; ++i)
            for (size_type j = 0; j < m; ++j)
                U(i, j) = U_data[j * m + i];

        // S as diagonal m x n matrix
        for (size_type i = 0; i < m; ++i)
            for (size_type j = 0; j < n; ++j)
                S(i, j) = (i == j && i < mn) ? S_vec[i] : math::zero<value_type>();

        // V from VT (transpose): V(i,j) = VT(j,i) = VT_data[i * n + j]
        for (size_type i = 0; i < n; ++i)
            for (size_type j = 0; j < n; ++j)
                V(i, j) = VT_data[i * n + j];

        return;
    }
#endif

    U.change_dim(m, m);
    V.change_dim(n, n);
    S.change_dim(m, n);

    for (size_type i = 0; i < m; ++i)
        for (size_type j = 0; j < m; ++j)
            U(i, j) = (i == j) ? math::one<value_type>() : math::zero<value_type>();

    for (size_type i = 0; i < n; ++i)
        for (size_type j = 0; j < n; ++j)
            V(i, j) = (i == j) ? math::one<value_type>() : math::zero<value_type>();

    if (mn == 0) return;

    mat::dense2D<value_type> B;

    // ---- One-sided Jacobi -------------------------------------------------
    //
    // Replaces an alternating-QR iteration that was both inaccurate and
    // self-destructive (#337). That scheme converged linearly to ~1e-10 and
    // then CORRUPTED its own answer: on a 10x10 SPD case the worst singular
    // value went 8.4e-09 wrong at iteration 100 to 136% wrong by iteration
    // 1000, while its convergence test -- off-diagonal mass over diagonal mass
    // -- kept shrinking to 1e-27 and so read as perfectly converged. Because
    // max_iter was 100*max(m,n), the shipped result was the corrupted one, and
    // for ~30% of inputs the degenerate reflectors turned it into all-NaN.
    //
    // One-sided Jacobi orthogonalizes the COLUMNS of A by plane rotations:
    // A*V = U*S, so the rotations are accumulated into V and the singular
    // values fall out as the column norms at the end. It converges
    // quadratically, needs no deflation or shifts, and has no degenerate case
    // to guard -- a pair that is already orthogonal is simply skipped.
    B.change_dim(m, n);
    for (size_type i = 0; i < m; ++i)
        for (size_type j = 0; j < n; ++j)
            B(i, j) = A(i, j);

    const value_type eps = std::numeric_limits<value_type>::epsilon();
    const value_type thresh = (tol > math::zero<value_type>()) ? tol : eps;
    const size_type max_sweeps = 60;

    for (size_type sweep = 0; sweep < max_sweeps; ++sweep) {
        bool rotated = false;
        for (size_type p = 0; p + 1 < n; ++p) {
            for (size_type q = p + 1; q < n; ++q) {
                // Form the inner products in COMMON-SCALED variables. Taken
                // raw, alpha and betac overflow to inf for entries around
                // 1e200 and underflow to zero around 1e-200 -- which silently
                // skipped every rotation, so the columns were never
                // orthogonalized (measured: a matrix scaled by 1e-160 came
                // back with |U^T U - I| = 0.87). One scale shared by both
                // columns keeps zeta, and hence the rotation, unchanged.
                value_type cs = math::zero<value_type>();
                for (size_type i = 0; i < m; ++i) {
                    const value_type ap = abs(B(i, p)), aq = abs(B(i, q));
                    if (ap > cs) cs = ap;
                    if (aq > cs) cs = aq;
                }
                if (cs == math::zero<value_type>()) continue;   // both columns zero

                value_type alpha = math::zero<value_type>();   // ||B_p||^2, scaled
                value_type gamma = math::zero<value_type>();   // B_p . B_q,  scaled
                value_type betac = math::zero<value_type>();   // ||B_q||^2, scaled
                for (size_type i = 0; i < m; ++i) {
                    const value_type bp = B(i, p) / cs, bq = B(i, q) / cs;
                    alpha += bp * bp;
                    betac += bq * bq;
                    gamma += bp * bq;
                }
                if (gamma == math::zero<value_type>()) continue;
                const value_type scale = sqrt(alpha * betac);
                // Already orthogonal to working precision: nothing to do. This
                // is the whole degenerate-case handling -- no reflector to
                // build, so nothing can underflow or divide by zero.
                if (scale == math::zero<value_type>() || abs(gamma) <= thresh * scale)
                    continue;

                // Rotation zeroing B_p . B_q. Root of smaller magnitude of
                // t^2 + 2*zeta*t - 1 = 0, chosen in the numerically stable form.
                const value_type zeta = (betac - alpha) / (value_type(2) * gamma);
                const value_type sgn  = (zeta >= math::zero<value_type>())
                                      ? math::one<value_type>() : -math::one<value_type>();
                const value_type t    = sgn / (abs(zeta) + sqrt(math::one<value_type>() + zeta * zeta));
                const value_type c    = math::one<value_type>() / sqrt(math::one<value_type>() + t * t);
                const value_type sn   = c * t;

                for (size_type i = 0; i < m; ++i) {
                    const value_type bp = B(i, p), bq = B(i, q);
                    B(i, p) = c * bp - sn * bq;
                    B(i, q) = sn * bp + c * bq;
                }
                for (size_type i = 0; i < n; ++i) {
                    const value_type vp = V(i, p), vq = V(i, q);
                    V(i, p) = c * vp - sn * vq;
                    V(i, q) = sn * vp + c * vq;
                }
                rotated = true;
            }
        }
        if (!rotated) break;   // all pairs orthogonal: converged
    }

    // Singular values are the column norms of B; U's columns are B normalized.
    std::vector<value_type> sigma(n);
    for (size_type j = 0; j < n; ++j) {
        // Scaled norm, for the same overflow/underflow reason as above.
        value_type cs = math::zero<value_type>();
        for (size_type i = 0; i < m; ++i) {
            const value_type a = abs(B(i, j));
            if (a > cs) cs = a;
        }
        if (cs == math::zero<value_type>()) { sigma[j] = math::zero<value_type>(); continue; }
        value_type s2 = math::zero<value_type>();
        for (size_type i = 0; i < m; ++i) {
            const value_type b = B(i, j) / cs;
            s2 += b * b;
        }
        sigma[j] = cs * sqrt(s2);
    }

    // Order descending, permuting V (and B, which becomes U) to match.
    std::vector<size_type> order(n);
    for (size_type j = 0; j < n; ++j) order[j] = j;
    std::sort(order.begin(), order.end(),
              [&sigma](size_type a, size_type b) { return sigma[a] > sigma[b]; });

    mat::dense2D<value_type> Bs(m, n), Vs(n, n);
    std::vector<value_type> sigma_s(n);
    for (size_type j = 0; j < n; ++j) {
        const size_type src = order[j];
        sigma_s[j] = sigma[src];
        for (size_type i = 0; i < m; ++i) Bs(i, j) = B(i, src);
        for (size_type i = 0; i < n; ++i) Vs(i, j) = V(i, src);
    }
    for (size_type i = 0; i < n; ++i)
        for (size_type j = 0; j < n; ++j)
            V(i, j) = Vs(i, j);

    for (size_type i = 0; i < m; ++i)
        for (size_type j = 0; j < n; ++j)
            S(i, j) = math::zero<value_type>();
    for (size_type j = 0; j < mn; ++j)
        S(j, j) = sigma_s[j];

    // U: normalized columns of B where the singular value is nonzero. The
    // remaining columns (rank deficiency, or m > n) are filled with an
    // orthonormal completion so U stays a genuine m x m orthogonal matrix.
    const value_type utol = (sigma_s.empty() ? math::zero<value_type>() : sigma_s[0])
                          * eps * value_type(m > n ? m : n);
    size_type filled = 0;
    for (size_type j = 0; j < n && filled < m; ++j) {
        if (sigma_s[j] <= utol) continue;
        for (size_type i = 0; i < m; ++i) U(i, filled) = Bs(i, j) / sigma_s[j];
        ++filled;
    }
    // Complete the basis by PIVOTED Gram-Schmidt over the canonical vectors:
    // at each step take the candidate with the largest remaining residual.
    //
    // A fixed acceptance bound does not work here. With d = m - filled
    // directions still missing, the residuals of the m canonical vectors
    // satisfy sum ||P_perp e_k||^2 = d, so the best of them is only guaranteed
    // to reach sqrt(d/m) -- 0.25 for a single missing direction at m = 16.
    // A `> 0.5` test therefore rejects EVERY candidate while the basis is
    // still incomplete, leaving stale identity columns in U. Measured on
    // A = I - v*v^T with v = ones/sqrt(m): |U^T U - I| was 0.87 at m = 4 and
    // 0.97 at m = 16. Taking the maximum each step always makes progress,
    // because that maximum is at least sqrt(d/m) > 0 whenever d > 0.
    if (filled < m) {
        // C holds the canonical vectors, kept orthogonal to the accepted
        // columns so each step costs O(m^2) rather than a full re-projection.
        mat::dense2D<value_type> C(m, m);
        for (size_type i = 0; i < m; ++i)
            for (size_type j = 0; j < m; ++j)
                C(i, j) = (i == j) ? math::one<value_type>() : math::zero<value_type>();

        for (size_type j = 0; j < filled; ++j)
            for (int pass = 0; pass < 2; ++pass)
                for (size_type k = 0; k < m; ++k) {
                    value_type d = math::zero<value_type>();
                    for (size_type i = 0; i < m; ++i) d += U(i, j) * C(i, k);
                    for (size_type i = 0; i < m; ++i) C(i, k) -= d * U(i, j);
                }

        while (filled < m) {
            size_type best = 0;
            value_type best_n2 = -math::one<value_type>();
            for (size_type k = 0; k < m; ++k) {
                value_type n2 = math::zero<value_type>();
                for (size_type i = 0; i < m; ++i) n2 += C(i, k) * C(i, k);
                if (n2 > best_n2) { best_n2 = n2; best = k; }
            }
            if (!(best_n2 > math::zero<value_type>())) break;   // numerically exhausted

            const value_type wn = sqrt(best_n2);
            for (size_type i = 0; i < m; ++i) U(i, filled) = C(i, best) / wn;
            // Re-orthogonalize the accepted column against the earlier ones.
            for (int pass = 0; pass < 2; ++pass)
                for (size_type j = 0; j < filled; ++j) {
                    value_type d = math::zero<value_type>();
                    for (size_type i = 0; i < m; ++i) d += U(i, j) * U(i, filled);
                    for (size_type i = 0; i < m; ++i) U(i, filled) -= d * U(i, j);
                }
            value_type rn = math::zero<value_type>();
            for (size_type i = 0; i < m; ++i) rn += U(i, filled) * U(i, filled);
            rn = sqrt(rn);
            if (!(rn > math::zero<value_type>())) break;
            for (size_type i = 0; i < m; ++i) U(i, filled) /= rn;
            ++filled;

            for (size_type k = 0; k < m; ++k) {
                value_type d = math::zero<value_type>();
                for (size_type i = 0; i < m; ++i) d += U(i, filled - 1) * C(i, k);
                for (size_type i = 0; i < m; ++i) C(i, k) -= d * U(i, filled - 1);
            }
        }
    }
}

/// Convenience overload returning a tuple of (U, S, V).
template <Matrix M>
auto svd(const M& A,
         typename M::value_type tol = std::numeric_limits<typename M::value_type>::epsilon()) {
    using value_type = typename M::value_type;
    mat::dense2D<value_type> U, S, V;
    svd(A, U, S, V, tol);
    return std::make_tuple(std::move(U), std::move(S), std::move(V));
}

} // namespace mtl
