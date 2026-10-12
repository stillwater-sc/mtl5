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

namespace detail {

/// One-sided Jacobi core shared by every svd overload. On entry B (m x n) holds
/// the matrix to decompose and V (n x n) the identity. It rotates B's columns
/// until each pair is orthogonal to `tol` (machine epsilon if tol <= 0),
/// accumulating the rotations into V, so that B_in * V = B_out. It then sorts
/// the column norms -- the singular values -- in descending order into
/// sigma_s (length n) and permutes the columns of B and V to match, so on
/// return B's column j is sigma_s[j] times the j-th left singular vector.
template <typename value_type>
void one_sided_jacobi(mat::dense2D<value_type>& B, mat::dense2D<value_type>& V,
                      std::vector<value_type>& sigma_s, value_type tol) {
    using size_type = typename mat::dense2D<value_type>::size_type;
    using std::abs;
    using std::sqrt;
    const size_type m = B.num_rows();
    const size_type n = B.num_cols();

    const value_type eps = std::numeric_limits<value_type>::epsilon();
    const value_type thresh = (tol > math::zero<value_type>()) ? tol : eps;
    const size_type max_sweeps = 60;

    // Column-major working copies: every loop below runs down a column, which
    // in B's row-major storage would stride by n.
    std::vector<value_type> b(m * n), v(n * n);
    for (size_type j = 0; j < n; ++j) {
        for (size_type i = 0; i < m; ++i) b[j * m + i] = B(i, j);
        for (size_type i = 0; i < n; ++i) v[j * n + i] = V(i, j);
    }

    // The inner products alpha = ||b_p||^2, betac = ||b_q||^2 and gamma =
    // b_p . b_q are first summed raw, which is one pass and no divisions.
    // They are kept when alpha and betac both land in [lo, hi]:
    //  - an overflow anywhere in the sums makes alpha or betac inf (gamma is
    //    bounded by sqrt(alpha * betac)), and inf or NaN fails the test;
    //  - hi = sqrt(max)/2 keeps alpha * betac, under the sqrt below, finite;
    //  - lo = sqrt(min)/eps keeps alpha * betac normal, and puts each column's
    //    largest square above min/eps^2, so every entry within a factor eps
    //    of the column's largest squares to a normal number.
    // Otherwise they are re-formed in variables scaled by the pair's largest
    // magnitude. Taken raw at the extremes they overflow to inf for entries
    // around 1e200 and underflow to zero around 1e-200 -- which silently
    // skipped every rotation, so the columns were never orthogonalized
    // (measured: a matrix scaled by 1e-160 came back with |U^T U - I| = 0.87).
    // One scale shared by both columns keeps zeta, and hence the rotation,
    // unchanged. A type without numeric_limits always takes the scaled path.
    constexpr bool raw_ok = std::numeric_limits<value_type>::is_specialized;
    value_type lo = math::zero<value_type>(), hi = math::zero<value_type>();
    if constexpr (raw_ok) {
        lo = sqrt(std::numeric_limits<value_type>::min()) / eps;
        hi = sqrt(std::numeric_limits<value_type>::max()) / value_type(2);
    }

    for (size_type sweep = 0; sweep < max_sweeps; ++sweep) {
        bool rotated = false;
        for (size_type p = 0; p + 1 < n; ++p) {
            value_type* bp_ = b.data() + p * m;
            for (size_type q = p + 1; q < n; ++q) {
                value_type* bq_ = b.data() + q * m;
                value_type alpha = math::zero<value_type>();
                value_type gamma = math::zero<value_type>();
                value_type betac = math::zero<value_type>();
                bool in_range = false;
                if constexpr (raw_ok) {
                    for (size_type i = 0; i < m; ++i) {
                        const value_type bp = bp_[i], bq = bq_[i];
                        alpha += bp * bp;
                        betac += bq * bq;
                        gamma += bp * bq;
                    }
                    in_range = alpha >= lo && alpha <= hi && betac >= lo && betac <= hi;
                }
                if (!in_range) {
                    value_type cs = math::zero<value_type>();
                    for (size_type i = 0; i < m; ++i) {
                        const value_type ap = abs(bp_[i]), aq = abs(bq_[i]);
                        if (ap > cs) cs = ap;
                        if (aq > cs) cs = aq;
                    }
                    if (cs == math::zero<value_type>()) continue;   // both columns zero

                    alpha = math::zero<value_type>();
                    gamma = math::zero<value_type>();
                    betac = math::zero<value_type>();
                    for (size_type i = 0; i < m; ++i) {
                        const value_type bp = bp_[i] / cs, bq = bq_[i] / cs;
                        alpha += bp * bp;
                        betac += bq * bq;
                        gamma += bp * bq;
                    }
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
                    const value_type bp = bp_[i], bq = bq_[i];
                    bp_[i] = c * bp - sn * bq;
                    bq_[i] = sn * bp + c * bq;
                }
                value_type* vp_ = v.data() + p * n;
                value_type* vq_ = v.data() + q * n;
                for (size_type i = 0; i < n; ++i) {
                    const value_type vp = vp_[i], vq = vq_[i];
                    vp_[i] = c * vp - sn * vq;
                    vq_[i] = sn * vp + c * vq;
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
        const value_type* bj = b.data() + j * m;
        value_type cs = math::zero<value_type>();
        for (size_type i = 0; i < m; ++i) {
            const value_type a = abs(bj[i]);
            if (a > cs) cs = a;
        }
        if (cs == math::zero<value_type>()) { sigma[j] = math::zero<value_type>(); continue; }
        value_type s2 = math::zero<value_type>();
        for (size_type i = 0; i < m; ++i) {
            const value_type x = bj[i] / cs;
            s2 += x * x;
        }
        sigma[j] = cs * sqrt(s2);
    }

    // Order descending, permuting V (and B, which becomes U) to match.
    std::vector<size_type> order(n);
    for (size_type j = 0; j < n; ++j) order[j] = j;
    std::sort(order.begin(), order.end(),
              [&sigma](size_type a, size_type b) { return sigma[a] > sigma[b]; });

    sigma_s.assign(n, math::zero<value_type>());
    for (size_type j = 0; j < n; ++j) {
        const size_type src = order[j];
        sigma_s[j] = sigma[src];
        for (size_type i = 0; i < m; ++i) B(i, j) = b[src * m + i];
        for (size_type i = 0; i < n; ++i) V(i, j) = v[src * n + i];
    }
}

} // namespace detail

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
template <FieldMatrix M>
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
    std::vector<value_type> sigma_s;
    detail::one_sided_jacobi(B, V, sigma_s, tol);
    const mat::dense2D<value_type>& Bs = B;

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

/// SVD without U: the singular values s (length min(m,n), descending) and the
/// n x n right singular vectors V, so that A * V(:,j) = s(j) * u_j for some unit
/// u_j that is never formed (#566). A null-space caller -- the 8-point (2n x 9)
/// or DLT (2n x 12) problem -- needs only V's last column, and the full svd
/// spends most of its time on the m x m U and on rotating m-long columns.
///
/// For a tall A (m > n) the matrix is first reduced by Householder QR, A = QR,
/// and one-sided Jacobi runs on the n x n triangle R (Drmac & Veselic's
/// preconditioner): R has A's singular values and right singular vectors, and
/// the sweeps then cost O(n^3) instead of O(m n^2) each, after an O(m n^2) QR.
/// Householder QR is column-wise backward stable, so a column scaling of A
/// carries over to R and Jacobi's relative accuracy is kept. For m <= n the
/// Jacobi runs on A itself and S and V are bit-identical to svd(A, U, S, V).
///
/// Always the generic C++ path: LAPACK's gesdd, the full svd's backend, has no
/// V-only mode. `tol` is as for svd.
template <FieldMatrix M>
void svd(const M& A,
         vec::dense_vector<typename M::value_type>& s,
         mat::dense2D<typename M::value_type>& V,
         typename M::value_type tol =
             std::numeric_limits<typename M::value_type>::epsilon()) {
    using value_type = typename M::value_type;
    using size_type  = typename M::size_type;

    const size_type m = A.num_rows();
    const size_type n = A.num_cols();
    const size_type mn = std::min(m, n);

    V.change_dim(n, n);
    for (size_type i = 0; i < n; ++i)
        for (size_type j = 0; j < n; ++j)
            V(i, j) = (i == j) ? math::one<value_type>() : math::zero<value_type>();
    s.change_dim(mn);
    if (mn == 0) return;

    mat::dense2D<value_type> B;
    if (m > n) {
        mat::dense2D<value_type> F(m, n);
        for (size_type i = 0; i < m; ++i)
            for (size_type j = 0; j < n; ++j)
                F(i, j) = A(i, j);
        vec::dense_vector<value_type> tau;
        qr_factor(F, tau);
        B.change_dim(n, n);
        for (size_type i = 0; i < n; ++i)
            for (size_type j = 0; j < n; ++j)
                B(i, j) = (i <= j) ? F(i, j) : math::zero<value_type>();
    } else {
        B.change_dim(m, n);
        for (size_type i = 0; i < m; ++i)
            for (size_type j = 0; j < n; ++j)
                B(i, j) = A(i, j);
    }

    std::vector<value_type> sigma;
    detail::one_sided_jacobi(B, V, sigma, tol);
    for (size_type j = 0; j < mn; ++j)
        s(j) = sigma[j];
}

/// A unit vector x minimizing ||A x||: the right singular vector of A's
/// smallest singular value, i.e. the last column of V from the V-only svd
/// (#566). For a wide A (m < n) the null space has dimension at least n - m
/// and x is one vector in it. The sign is arbitrary. Returns an empty vector
/// when A has no columns.
template <FieldMatrix M>
vec::dense_vector<typename M::value_type>
null_vector(const M& A,
            typename M::value_type tol =
                std::numeric_limits<typename M::value_type>::epsilon()) {
    using value_type = typename M::value_type;
    using size_type  = typename M::size_type;
    const size_type n = A.num_cols();
    vec::dense_vector<value_type> x(n);
    if (n == 0) return x;
    vec::dense_vector<value_type> s;
    mat::dense2D<value_type> V;
    svd(A, s, V, tol);
    for (size_type i = 0; i < n; ++i)
        x(i) = V(i, n - 1);
    return x;
}

/// Convenience overload returning a tuple of (U, S, V).
template <FieldMatrix M>
auto svd(const M& A,
         typename M::value_type tol = std::numeric_limits<typename M::value_type>::epsilon()) {
    using value_type = typename M::value_type;
    mat::dense2D<value_type> U, S, V;
    svd(A, U, S, V, tol);
    return std::make_tuple(std::move(U), std::move(S), std::move(V));
}

} // namespace mtl
