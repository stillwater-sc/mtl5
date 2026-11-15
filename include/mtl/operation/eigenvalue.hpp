#pragma once
// MTL5 -- General eigenvalue computation via QR algorithm on Hessenberg form
// Reduce to upper Hessenberg, then apply the Francis implicit double-shift QR.
#include <cmath>
#include <complex>
#include <algorithm>
#include <cassert>
#include <limits>
#include <stdexcept>
#include <vector>
#include <mtl/concepts/matrix.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/operation/hessenberg.hpp>
#include <mtl/detail/francis_qr.hpp>
#include <mtl/detail/unit_roundoff.hpp>
#include <mtl/math/identity.hpp>
#include <mtl/interface/dispatch_traits.hpp>
#ifdef MTL5_HAS_LAPACK
#include <mtl/interface/lapack.hpp>
#endif

namespace mtl {

namespace detail {

#ifdef MTL5_HAS_LAPACK
/// LAPACK geev on a real float/double dense matrix. Fills `eigs` (size n) with
/// the eigenvalues; if `V != nullptr`, also fills right eigenvectors (column k
/// pairs with eigs(k)) into *V, unpacking LAPACK's real/complex-conjugate
/// column packing into complex columns, unit-normalized with a canonical phase
/// (largest-magnitude entry real-positive) to match the in-house path.
template <typename M>
void lapack_geev(const M& A,
                 vec::dense_vector<std::complex<typename M::value_type>>& eigs,
                 mat::dense2D<std::complex<typename M::value_type>>* V) {
    using value_type   = typename M::value_type;
    using complex_type = std::complex<value_type>;
    using size_type    = typename M::size_type;
    using std::abs;
    using std::sqrt;
    const size_type n  = A.num_rows();
    const int ni = static_cast<int>(n);

    // Column-major working copy (LAPACK is Fortran/column-major); built through
    // the (i,j) accessor so it is correct regardless of M's own orientation.
    std::vector<value_type> a(static_cast<std::size_t>(n) * n);
    for (size_type i = 0; i < n; ++i)
        for (size_type j = 0; j < n; ++j)
            a[static_cast<std::size_t>(j) * n + i] = A(i, j);

    std::vector<value_type> wr(n), wi(n);
    std::vector<value_type> vr;
    value_type vl_dummy[1] = {value_type(0)};
    char jobvr = 'N';
    value_type* vrp = vl_dummy;   // unreferenced when jobvr == 'N'
    int ldvr = 1;
    if (V != nullptr) {
        jobvr = 'V';
        vr.resize(static_cast<std::size_t>(n) * n);
        vrp = vr.data();
        ldvr = ni;
    }

    // Workspace query, then compute.
    value_type wq = value_type(0);
    interface::lapack::geev('N', jobvr, ni, a.data(), ni, wr.data(), wi.data(),
                            vl_dummy, 1, vrp, ldvr, &wq, -1);
    int lwork = static_cast<int>(wq);
    if (lwork < 4 * ni) lwork = 4 * ni;   // LAPACK minimum for jobvr='V'
    std::vector<value_type> work(static_cast<std::size_t>(lwork));
    int info = interface::lapack::geev('N', jobvr, ni, a.data(), ni,
                                       wr.data(), wi.data(),
                                       vl_dummy, 1, vrp, ldvr,
                                       work.data(), lwork);
    if (info != 0)
        throw std::runtime_error("mtl::eigenvalue: LAPACK geev failed to converge");

    for (size_type i = 0; i < n; ++i)
        eigs(i) = complex_type(wr[i], wi[i]);

    if (V == nullptr) return;

    // Unpack right eigenvectors from the column-major vr buffer. A real
    // eigenvalue j uses column j; a complex-conjugate pair (j, j+1) with
    // wi[j] > 0 uses vr[:,j] +/- i*vr[:,j+1].
    auto vrc = [&](size_type c, size_type row) -> value_type {
        return vr[static_cast<std::size_t>(c) * n + row];
    };
    for (size_type j = 0; j < n; ) {
        if (wi[j] == value_type(0)) {
            for (size_type i = 0; i < n; ++i)
                (*V)(i, j) = complex_type(vrc(j, i), value_type(0));
            ++j;
        } else {
            for (size_type i = 0; i < n; ++i) {
                value_type re = vrc(j, i), im = vrc(j + 1, i);
                (*V)(i, j)     = complex_type(re,  im);
                (*V)(i, j + 1) = complex_type(re, -im);
            }
            j += 2;
        }
    }
    // Unit-normalize and fix a canonical phase, matching the in-house path.
    for (size_type k = 0; k < n; ++k) {
        value_type nrm = value_type(0);
        for (size_type i = 0; i < n; ++i) nrm += std::norm((*V)(i, k));
        nrm = sqrt(nrm);
        if (nrm > value_type(0))
            for (size_type i = 0; i < n; ++i) (*V)(i, k) /= nrm;
        size_type imax = 0;
        value_type mmax = value_type(0);
        for (size_type i = 0; i < n; ++i) {
            value_type m = abs((*V)(i, k));
            if (m > mmax) { mmax = m; imax = i; }
        }
        if (mmax > value_type(0)) {
            complex_type phase = (*V)(imax, k) / abs((*V)(imax, k));
            for (size_type i = 0; i < n; ++i) (*V)(i, k) /= phase;
        }
    }
}
#endif // MTL5_HAS_LAPACK

/// A right eigenvector y of the real quasi-upper-triangular T (a real Schur
/// form, 2x2 blocks in standard form) for its eigenvalue lambda at index k:
/// (T - lambda I) y = 0, with y zero below lambda's block. Back-substitution
/// over T's diagonal blocks in complex arithmetic, the recurrence of LAPACK's
/// dtrevc. A diagonal block whose shifted value or determinant is below smin
/// -- the same eigenvalue again -- is perturbed to smin, and y is rescaled
/// when an entry passes `limit`. O(n^2) per eigenvector.
template <typename T>
std::vector<std::complex<T>> quasi_triangular_eigenvector(const mat::dense2D<T>& R, std::size_t k,
                                                          std::complex<T> lambda, T smin, T limit) {
    using C = std::complex<T>;
    using std::abs;
    const std::size_t n = R.num_rows();
    const T zero(0);
    std::vector<C> y(n, C(zero));

    // lambda's own block, rows top..bot.
    std::size_t top = k, bot = k;
    if (k + 1 < n && !(R(k + 1, k) == zero)) bot = k + 1;
    else if (k > 0 && !(R(k, k - 1) == zero)) top = k - 1;
    if (top == bot) {
        y[k] = C(T(1));
    } else {
        // (B - lambda I) [u; v] = 0 for the block B: from its first row,
        // u = t12, v = lambda - t11 (t12 != 0 in standard form).
        if (!(R(top, bot) == zero)) { y[top] = C(R(top, bot)); y[bot] = lambda - C(R(top, top)); }
        else                        { y[top] = lambda - C(R(bot, bot)); y[bot] = C(R(bot, top)); }
    }

    for (std::size_t i = top; i > 0; ) {
        const std::size_t b1 = i - 1;
        const std::size_t b0 = (b1 > 0 && !(R(b1, b1 - 1) == zero)) ? b1 - 1 : b1;
        C r0(zero), r1(zero);
        for (std::size_t m = b1 + 1; m <= bot; ++m) {
            r1 -= C(R(b1, m)) * y[m];
            if (b0 < b1) r0 -= C(R(b0, m)) * y[m];
        }
        if (b0 == b1) {
            C d = C(R(b1, b1)) - lambda;
            if (abs(d) < smin) d = C(smin);
            y[b1] = r1 / d;
        } else {
            const C a11 = C(R(b0, b0)) - lambda, a12 = C(R(b0, b1));
            const C a21 = C(R(b1, b0)),          a22 = C(R(b1, b1)) - lambda;
            C det = a11 * a22 - a12 * a21;
            if (abs(det) < smin) det = C(smin);
            y[b0] = (r0 * a22 - a12 * r1) / det;
            y[b1] = (a11 * r1 - a21 * r0) / det;
        }
        T mx = zero;
        for (std::size_t j = b0; j <= b1; ++j)
            if (abs(y[j]) > mx) mx = abs(y[j]);
        if (mx > limit)
            for (std::size_t j = b0; j <= bot; ++j) y[j] /= mx;
        i = b0;
    }
    return y;
}

} // namespace detail

/// Compute eigenvalues of a general (non-symmetric) square matrix via the
/// Francis implicit double-shift QR algorithm on the upper Hessenberg form.
/// Returns eigenvalues as a dense_vector of complex values (real and
/// complex-conjugate pairs read from the 1x1/2x2 blocks of the real Schur form).
/// Throws std::invalid_argument for a non-square A or a NaN or infinite entry,
/// and std::runtime_error if the QR iteration fails to converge. The in-house
/// path scales A by a power of two first, as schur() does (#853).
///
/// Dispatches to LAPACK geev when MTL5_HAS_LAPACK is defined and the type
/// qualifies (dense2D<float/double>, either orientation); otherwise uses the
/// in-house path above, which also serves custom number types (posits, LNS, ...).
///
/// `Accumulator` (leading, default void) is forwarded to the Householder
/// Hessenberg reduction (#536). A non-void accumulator always takes the
/// in-house path, since geev forms its sums its own way. The Francis QR
/// iteration keeps plain arithmetic: its updates are fixed 2- and 3-term
/// expressions, not sums.
template <typename Accumulator = void, FieldMatrix M>
auto eigenvalue(const M& A, typename M::value_type tol = 1e-10,
                typename M::size_type max_iter = 0) {
    using value_type = typename M::value_type;
    using complex_type = std::complex<value_type>;
    using size_type  = typename M::size_type;
    using std::abs;
    using std::sqrt;
    const size_type n = A.num_rows();
    // Square and finite, on every path (#853): a non-square A was only
    // asserted, and a NaN reached the iteration.
    detail::require_square_finite(A, "mtl::eigenvalue");

    vec::dense_vector<complex_type> eigs(n);
    if (n == 0) return eigs;

#ifdef MTL5_HAS_LAPACK
    // Accelerate real float/double dense matrices via LAPACK geev (mirrors the
    // symmetric syev dispatch). lapack_geev builds its column-major copy through
    // the (i,j) accessor, so this is correct for any orientation of M -- no
    // !is_row_major_v guard (see #417). Custom number types fall through to the
    // in-house double-shift QR below.
    if constexpr (interface::BlasDenseMatrix<M> &&
                  interface::accumulator_allows_blas_v<Accumulator>) {
        detail::lapack_geev(A, eigs, static_cast<mat::dense2D<complex_type>*>(nullptr));
        return eigs;
    }
#endif

    // Scale by a power of two to max |a_ij| in [1/2, 1], as schur() does, and
    // scale the eigenvalues back at the end (#853). Unscaled, a matrix with
    // entries near 1e-170 or 1e-200 gave silently wrong eigenvalues and one
    // near 1e+-250 failed to converge, where the LAPACK path was right.
    mat::dense2D<value_type> W(n, n);
    for (size_type i = 0; i < n; ++i)
        for (size_type j = 0; j < n; ++j) W(i, j) = A(i, j);
    const value_type unscale = detail::scale_to_unit_binade(W);

    // Reduce to upper Hessenberg form.
    mat::dense2D<value_type> H = hessenberg<Accumulator>(W);
    if (n == 1) { eigs(0) = complex_type(H(0, 0) * unscale); return eigs; }

    // Francis implicit double-shift QR on the real Hessenberg form (EISPACK
    // hqr, in detail/francis_qr.hpp, shared with schur()). Converges to real
    // Schur form and reads 1x1 (real) and 2x2 (complex conjugate) diagonal
    // blocks. Unlike a single real shift, the double shift uses the trailing
    // 2x2 block's complex-conjugate pair, so complex spectra of strongly
    // non-normal matrices are handled without stalling. Exceptional shifts
    // break stagnation; non-convergence is reported, never returned as a
    // silently-wrong diagonal read.
    detail::francis_options<value_type> opt;
    opt.tol = tol;
    // Per-block iteration cap before an exceptional shift / bailout.
    opt.max_its = (max_iter > 0) ? static_cast<int>(max_iter) : 100;
    vec::dense_vector<value_type> wr, wi;
    detail::francis_qr<detail::francis_output::eigenvalues>(H, static_cast<mat::dense2D<value_type>*>(nullptr), wr, wi, opt);

    for (size_type i = 0; i < n; ++i)
        eigs(i) = complex_type(wr[i] * unscale, wi[i] * unscale);
    return eigs;
}

/// Compute eigenvalues AND right eigenvectors of a general (non-symmetric)
/// square matrix. Returns {eigenvalues, eigenvectors} where:
///   eigenvalues:  dense_vector<complex> of size n
///   eigenvectors: dense2D<complex> of size n x n, column k is the right
///                 eigenvector for eigenvalues(k), i.e. A * v_k = lambda_k * v_k
///
/// In-house path: the real Schur form A = Q T Q^T (Hessenberg reduction, then
/// the Francis double-shift QR accumulating Q, as schur() runs it, on A scaled
/// by a power of two), then each eigenvector of T by back-substitution on its
/// quasi-triangular form, v_k = Q y_k (LAPACK dtrevc's method). That is
/// O(n^3) in all. It used to factor a fresh complex n x n matrix A - lambda I
/// for every eigenvalue, O(n^4): 0.38 s at n = 160 (#853). Eigenvectors are
/// unit 2-norm with a canonical phase (largest entry made real-positive), and
/// pair with the eigenvalues by column. The eigenvalues are eigenvalue()'s, in
/// its order; each vector is the one of T's matching eigenvalue, which agrees
/// with it to rounding at the given tol. A repeated eigenvalue of a diagonalizable A gives independent
/// vectors; for a defective A they are as ill-conditioned as the problem.
///
/// Throws std::invalid_argument for a non-square A or a NaN or infinite
/// entry, and std::runtime_error when the iteration does not converge within
/// max_iter sweeps per eigenvalue (default 100).
///
/// Dispatches to LAPACK geev (eigenvalues + right eigenvectors) when
/// MTL5_HAS_LAPACK is defined and the type qualifies (dense2D<float/double>,
/// either orientation); otherwise uses the in-house path above.
///
/// `Accumulator` (leading, default void) is forwarded to the Hessenberg
/// reduction and its Q (#536), and a non-void accumulator skips geev. The QR
/// iteration and the back-substitution keep plain arithmetic.
template <typename Accumulator = void, FieldMatrix M>
auto eigen(const M& A, typename M::value_type tol = 1e-10,
           typename M::size_type max_iter = 0) {
    using value_type   = typename M::value_type;
    using complex_type = std::complex<value_type>;
    using size_type    = typename M::size_type;
    using std::abs;
    using std::sqrt;
    const size_type n = A.num_rows();
    detail::require_square_finite(A, "mtl::eigen");

    struct EigenResult {
        vec::dense_vector<complex_type> eigenvalues;
        mat::dense2D<complex_type> eigenvectors;
    };

    if (n == 0)
        return EigenResult{vec::dense_vector<complex_type>(0),
                           mat::dense2D<complex_type>(0, 0)};

#ifdef MTL5_HAS_LAPACK
    // LAPACK geev computes eigenvalues and right eigenvectors together for real
    // float/double dense matrices (mirrors the eigenvalue dispatch). lapack_geev
    // copies through the (i,j) accessor, so any orientation of M is correct --
    // no !is_row_major_v guard (see #417).
    if constexpr (interface::BlasDenseMatrix<M> &&
                  interface::accumulator_allows_blas_v<Accumulator>) {
        vec::dense_vector<complex_type> eigs_l(n);
        mat::dense2D<complex_type> V_l(n, n);
        detail::lapack_geev(A, eigs_l, &V_l);
        return EigenResult{eigs_l, V_l};
    }
#endif

    // The real Schur form of A scaled to max |a_ij| in [1/2, 1].
    mat::dense2D<value_type> W(n, n);
    for (size_type i = 0; i < n; ++i)
        for (size_type j = 0; j < n; ++j) W(i, j) = A(i, j);
    const value_type unscale = detail::scale_to_unit_binade(W);
    vec::dense_vector<value_type> tau;
    hessenberg_factor<Accumulator>(W, tau);
    mat::dense2D<value_type> Q = hessenberg_extract_Q<Accumulator>(W, tau);
    mat::dense2D<value_type> T = hessenberg_extract_H(W);

    const value_type eps = detail::machine_epsilon<value_type>();
    detail::francis_options<value_type> opt;
    opt.who = "mtl::eigen";
    opt.tol = tol;
    opt.max_its = (max_iter > 0) ? static_cast<int>(max_iter) : 100;
    opt.lanv2 = detail::make_lanv2_constants<value_type>();
    vec::dense_vector<value_type> wr, wi;
    detail::francis_qr<detail::francis_output::schur_form>(T, &Q, wr, wi, opt);
    for (size_type i = 2; i < n; ++i)
        for (size_type j = 0; j + 1 < i; ++j) T(i, j) = value_type(0);

    // The eigenvalues returned are eigenvalue()'s, in its order, as they have
    // always been: the Schur-form iteration may place them differently. Each
    // is paired with the nearest unused eigenvalue of T, whose vector it gets.
    vec::dense_vector<complex_type> eigs = eigenvalue<Accumulator>(A, tol, max_iter);

    mat::dense2D<complex_type> V(n, n);
    if (n == 1) {
        V(0, 0) = complex_type(value_type(1));
        return EigenResult{eigs, V};
    }
    std::vector<size_type> at(static_cast<std::size_t>(n));   // eigs(k) is T's eigenvalue at(k)
    {
        std::vector<bool> used(static_cast<std::size_t>(n), false);
        for (size_type k = 0; k < n; ++k) {
            size_type best = 0;
            value_type dbest = value_type(-1);
            for (size_type j = 0; j < n; ++j) {
                if (used[static_cast<std::size_t>(j)]) continue;
                const value_type d = abs(eigs(k) / unscale - complex_type(wr[j], wi[j]));
                if (dbest < value_type(0) || d < dbest) { dbest = d; best = j; }
            }
            used[static_cast<std::size_t>(best)] = true;
            at[static_cast<std::size_t>(k)] = best;
        }
    }

    // T's scale is about 1, so eps * max|t_ij| is the perturbation floor for a
    // repeated eigenvalue, and 1 / eps a safe bound on the growth of y.
    value_type tmax = value_type(0);
    for (size_type i = 0; i < n; ++i)
        for (size_type j = 0; j < n; ++j)
            if (abs(T(i, j)) > tmax) tmax = abs(T(i, j));
    value_type smin = eps * tmax;
    if (!(smin > value_type(0))) smin = eps;
    const value_type limit = value_type(1) / eps;

    const std::size_t nn = static_cast<std::size_t>(n);
    std::vector<complex_type> x(nn);
    for (size_type k = 0; k < n; ++k) {
        const size_type j = at[static_cast<std::size_t>(k)];
        const std::vector<complex_type> y = detail::quasi_triangular_eigenvector(
            T, static_cast<std::size_t>(j), complex_type(wr[j], wi[j]), smin, limit);
        // y is zero below lambda's block: x = Q(:, 0..bot) y(0..bot).
        std::size_t bot = nn;
        while (bot > 0 && y[bot - 1] == complex_type(value_type(0))) --bot;
        for (std::size_t i = 0; i < nn; ++i) {
            complex_type xi(value_type(0));
            for (std::size_t j = 0; j < bot; ++j)
                xi += complex_type(Q(static_cast<size_type>(i), static_cast<size_type>(j))) * y[j];
            x[i] = xi;
        }

        // Unit 2-norm, formed from the entries scaled by the largest, and the
        // canonical phase: the largest-magnitude entry real-positive.
        std::size_t imax = 0;
        value_type mmax = value_type(0);
        for (std::size_t i = 0; i < nn; ++i) {
            const value_type m = abs(x[i]);
            if (m > mmax) { mmax = m; imax = i; }
        }
        if (mmax > value_type(0)) {
            value_type s2 = value_type(0);
            for (std::size_t i = 0; i < nn; ++i) {
                x[i] /= mmax;
                s2 += std::norm(x[i]);
            }
            const value_type nrm = sqrt(s2);
            const complex_type phase = x[imax] / abs(x[imax]);
            for (std::size_t i = 0; i < nn; ++i) x[i] /= (nrm * phase);
        }
        for (size_type i = 0; i < n; ++i)
            V(i, k) = x[static_cast<std::size_t>(i)];
    }

    return EigenResult{eigs, V};
}

} // namespace mtl
