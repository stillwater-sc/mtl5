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

/// In-place LU factorization of an n x n complex matrix (row-major, `M[i*n+j]`)
/// with partial pivoting. Tiny pivots are floored to `pivot_floor` so that a
/// (near-)singular system stays solvable -- exactly the regime that inverse
/// iteration relies on, since A - lambda*I is singular at a true eigenvalue.
/// `piv[i]` holds the original row now living in row i after pivoting.
template <typename Complex, typename Real>
void eig_lu_factor(std::vector<Complex>& M, std::vector<std::size_t>& piv,
                   std::size_t n, Real pivot_floor) {
    using std::abs;
    for (std::size_t i = 0; i < n; ++i) piv[i] = i;
    for (std::size_t k = 0; k < n; ++k) {
        // Partial pivot: largest-magnitude entry in column k at or below k.
        std::size_t p = k;
        Real best = abs(M[k * n + k]);
        for (std::size_t i = k + 1; i < n; ++i) {
            Real v = abs(M[i * n + k]);
            if (v > best) { best = v; p = i; }
        }
        if (p != k) {
            for (std::size_t j = 0; j < n; ++j)
                std::swap(M[k * n + j], M[p * n + j]);
            std::swap(piv[k], piv[p]);
        }
        Complex akk = M[k * n + k];
        if (abs(akk) < pivot_floor) { akk = Complex(pivot_floor); M[k * n + k] = akk; }
        for (std::size_t i = k + 1; i < n; ++i) {
            Complex f = M[i * n + k] / akk;
            M[i * n + k] = f;
            for (std::size_t j = k + 1; j < n; ++j)
                M[i * n + j] -= f * M[k * n + j];
        }
    }
}

/// Solve M x = b in place given the LU factors from eig_lu_factor.
template <typename Complex>
void eig_lu_solve(const std::vector<Complex>& M, const std::vector<std::size_t>& piv,
                  std::size_t n, std::vector<Complex>& b) {
    std::vector<Complex> x(n);
    for (std::size_t i = 0; i < n; ++i) x[i] = b[piv[i]];
    // Forward substitution (unit lower triangle).
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < i; ++j)
            x[i] -= M[i * n + j] * x[j];
    // Back substitution (upper triangle).
    for (std::size_t i = n; i-- > 0; ) {
        for (std::size_t j = i + 1; j < n; ++j)
            x[i] -= M[i * n + j] * x[j];
        x[i] /= M[i * n + i];
    }
    b = std::move(x);
}

} // namespace detail

/// Compute eigenvalues of a general (non-symmetric) square matrix via the
/// Francis implicit double-shift QR algorithm on the upper Hessenberg form.
/// Returns eigenvalues as a dense_vector of complex values (real and
/// complex-conjugate pairs read from the 1x1/2x2 blocks of the real Schur form).
/// Throws std::runtime_error if the QR iteration fails to converge.
///
/// Dispatches to LAPACK geev when MTL5_HAS_LAPACK is defined and the type
/// qualifies (column-major dense2D<float/double>); otherwise uses the in-house
/// path above, which also serves custom number types (posits, LNS, ...).
template <Matrix M>
auto eigenvalue(const M& A, typename M::value_type tol = 1e-10,
                typename M::size_type max_iter = 0) {
    using value_type = typename M::value_type;
    using complex_type = std::complex<value_type>;
    using size_type  = typename M::size_type;
    using std::abs;
    using std::sqrt;
    const size_type n = A.num_rows();
    assert(n == A.num_cols());

    vec::dense_vector<complex_type> eigs(n);
    if (n == 0) return eigs;

#ifdef MTL5_HAS_LAPACK
    // Accelerate real float/double column-major dense matrices via LAPACK geev
    // (mirrors the symmetric syev dispatch). Custom number types and other
    // orientations fall through to the in-house double-shift QR below.
    if constexpr (interface::BlasDenseMatrix<M> && !interface::is_row_major_v<M>) {
        detail::lapack_geev(A, eigs, static_cast<mat::dense2D<complex_type>*>(nullptr));
        return eigs;
    }
#endif

    // Reduce to upper Hessenberg form.
    mat::dense2D<value_type> H = hessenberg(A);
    if (n == 1) { eigs(0) = complex_type(H(0, 0)); return eigs; }

    // Francis implicit double-shift QR on the real Hessenberg form (EISPACK
    // hqr). Converges to real Schur form and reads 1x1 (real) and 2x2 (complex
    // conjugate) diagonal blocks. Unlike a single real shift, the double shift
    // uses the trailing 2x2 block's complex-conjugate pair, so complex spectra
    // of strongly non-normal matrices are handled without stalling. Exceptional
    // shifts break stagnation; non-convergence is reported, never returned as a
    // silently-wrong diagonal read.
    using idx = long long;
    // Per-block iteration cap before an exceptional shift / bailout.
    const int max_its = (max_iter > 0) ? static_cast<int>(max_iter) : 100;

    // sign(a,b): magnitude of a with the sign of b.
    auto sign = [](value_type a, value_type b) {
        return (b >= value_type(0)) ? abs(a) : -abs(a);
    };

    std::vector<value_type> wr(n), wi(n);

    // Norm of the Hessenberg matrix (fallback scale for the deflation test).
    value_type anorm = value_type(0);
    for (size_type i = 0; i < n; ++i)
        for (size_type j = (i >= 1 ? i - 1 : size_type(0)); j < n; ++j)
            anorm += abs(H(i, j));

    idx nn = static_cast<idx>(n) - 1;  // bottom index of the active block
    value_type t = value_type(0);      // accumulated exceptional shift

    while (nn >= 0) {
        int its = 0;
        idx l;
        do {
            // Look for a single small subdiagonal element.
            for (l = nn; l >= 1; --l) {
                value_type s = abs(H(l - 1, l - 1)) + abs(H(l, l));
                if (s == value_type(0)) s = anorm;
                if (abs(H(l, l - 1)) <= tol * s) { H(l, l - 1) = value_type(0); break; }
            }
            if (l < 0) l = 0;

            value_type x = H(nn, nn);
            if (l == nn) {
                // One real root.
                wr[static_cast<size_type>(nn)] = x + t;
                wi[static_cast<size_type>(nn)] = value_type(0);
                nn -= 1;
            } else {
                value_type y = H(nn - 1, nn - 1);
                value_type w = H(nn, nn - 1) * H(nn - 1, nn);
                if (l == nn - 1) {
                    // Two roots: eigenvalues of the trailing 2x2 block.
                    value_type p = value_type(0.5) * (y - x);
                    value_type q = p * p + w;
                    value_type z = sqrt(abs(q));
                    x += t;
                    if (q >= value_type(0)) {
                        z = p + sign(z, p);
                        wr[static_cast<size_type>(nn - 1)] = wr[static_cast<size_type>(nn)] = x + z;
                        if (z != value_type(0)) wr[static_cast<size_type>(nn)] = x - w / z;
                        wi[static_cast<size_type>(nn - 1)] = wi[static_cast<size_type>(nn)] = value_type(0);
                    } else {
                        wr[static_cast<size_type>(nn - 1)] = wr[static_cast<size_type>(nn)] = x + p;
                        wi[static_cast<size_type>(nn)]     = z;
                        wi[static_cast<size_type>(nn - 1)] = -z;
                    }
                    nn -= 2;
                } else {
                    // No root yet: perform a Francis double-shift QR sweep.
                    if (its == max_its)
                        throw std::runtime_error(
                            "mtl::eigenvalue: QR iteration failed to converge");
                    value_type p = value_type(0), q = value_type(0), r = value_type(0);
                    value_type z = value_type(0);
                    if (its == 10 || its == 20) {
                        // Exceptional (ad-hoc Wilkinson) shift to break stagnation.
                        t += x;
                        for (idx i = 0; i <= nn; ++i) H(i, i) -= x;
                        value_type s = abs(H(nn, nn - 1)) + abs(H(nn - 1, nn - 2));
                        y = x = value_type(0.75) * s;
                        w = value_type(-0.4375) * s * s;
                    }
                    ++its;

                    // Look for two consecutive small subdiagonal elements.
                    idx m;
                    for (m = nn - 2; m >= l; --m) {
                        z = H(m, m);
                        r = x - z;
                        value_type s = y - z;
                        p = (r * s - w) / H(m + 1, m) + H(m, m + 1);
                        q = H(m + 1, m + 1) - z - r - s;
                        r = H(m + 2, m + 1);
                        s = abs(p) + abs(q) + abs(r);
                        p /= s; q /= s; r /= s;
                        if (m == l) break;
                        value_type u = abs(H(m, m - 1)) * (abs(q) + abs(r));
                        value_type v = abs(p) * (abs(H(m - 1, m - 1)) + abs(z) + abs(H(m + 1, m + 1)));
                        if (u <= tol * v) break;
                    }

                    for (idx i = m + 2; i <= nn; ++i) {
                        H(i, i - 2) = value_type(0);
                        if (i != m + 2) H(i, i - 3) = value_type(0);
                    }

                    // Chase the bulge with Householder(3)/(2) transforms.
                    for (idx k = m; k <= nn - 1; ++k) {
                        if (k != m) {
                            p = H(k, k - 1);
                            q = H(k + 1, k - 1);
                            r = value_type(0);
                            if (k != nn - 1) r = H(k + 2, k - 1);
                            x = abs(p) + abs(q) + abs(r);
                            if (x != value_type(0)) { p /= x; q /= x; r /= x; }
                        }
                        value_type s = sign(sqrt(p * p + q * q + r * r), p);
                        if (s != value_type(0)) {
                            if (k == m) {
                                if (l != m) H(k, k - 1) = -H(k, k - 1);
                            } else {
                                H(k, k - 1) = -s * x;
                            }
                            p += s;
                            x = p / s; y = q / s; z = r / s;
                            q /= p; r /= p;
                            // Row transformation.
                            for (idx j = k; j <= nn; ++j) {
                                p = H(k, j) + q * H(k + 1, j);
                                if (k != nn - 1) {
                                    p += r * H(k + 2, j);
                                    H(k + 2, j) -= p * z;
                                }
                                H(k + 1, j) -= p * y;
                                H(k, j)     -= p * x;
                            }
                            idx mmin = (nn < k + 3) ? nn : k + 3;
                            // Column transformation.
                            for (idx i = l; i <= mmin; ++i) {
                                p = x * H(i, k) + y * H(i, k + 1);
                                if (k != nn - 1) {
                                    p += z * H(i, k + 2);
                                    H(i, k + 2) -= p * r;
                                }
                                H(i, k + 1) -= p * q;
                                H(i, k)     -= p;
                            }
                        }
                    }
                }
            }
        } while (l < nn - 1);
    }

    for (size_type i = 0; i < n; ++i)
        eigs(i) = complex_type(wr[i], wi[i]);
    return eigs;
}

/// Compute eigenvalues AND right eigenvectors of a general (non-symmetric)
/// square matrix. Returns {eigenvalues, eigenvectors} where:
///   eigenvalues:  dense_vector<complex> of size n
///   eigenvectors: dense2D<complex> of size n x n, column k is the right
///                 eigenvector for eigenvalues(k), i.e. A * v_k = lambda_k * v_k
///
/// Eigenvalues come from the tested general QR path (`eigenvalue`); each
/// eigenvector is then recovered by inverse iteration on A - lambda_k*I. This
/// handles real and complex-conjugate eigenpairs uniformly in complex
/// arithmetic. Eigenvectors are unit 2-norm with a canonical phase (largest
/// entry made real-positive). Eigenvalue/eigenvector pairing matches by column.
///
/// Accuracy is bounded by the eigenvalues from `eigenvalue`: inverse iteration
/// recovers the eigenvector of the true eigenvalue nearest each computed
/// lambda_k. Since `eigenvalue` now uses a Francis double-shift QR, complex
/// spectra of strongly non-normal matrices are resolved accurately here too.
///
/// Dispatches to LAPACK geev (eigenvalues + right eigenvectors) when
/// MTL5_HAS_LAPACK is defined and the type qualifies (column-major
/// dense2D<float/double>); otherwise uses the in-house path above.
template <Matrix M>
auto eigen(const M& A, typename M::value_type tol = 1e-10,
           typename M::size_type max_iter = 0) {
    using value_type   = typename M::value_type;
    using complex_type = std::complex<value_type>;
    using size_type    = typename M::size_type;
    using std::abs;
    using std::sqrt;
    const size_type n = A.num_rows();
    assert(n == A.num_cols());

    struct EigenResult {
        vec::dense_vector<complex_type> eigenvalues;
        mat::dense2D<complex_type> eigenvectors;
    };

    if (n == 0)
        return EigenResult{vec::dense_vector<complex_type>(0),
                           mat::dense2D<complex_type>(0, 0)};

#ifdef MTL5_HAS_LAPACK
    // LAPACK geev computes eigenvalues and right eigenvectors together for real
    // float/double column-major dense matrices (mirrors the eigenvalue dispatch).
    if constexpr (interface::BlasDenseMatrix<M> && !interface::is_row_major_v<M>) {
        vec::dense_vector<complex_type> eigs_l(n);
        mat::dense2D<complex_type> V_l(n, n);
        detail::lapack_geev(A, eigs_l, &V_l);
        return EigenResult{eigs_l, V_l};
    }
#endif

    // Eigenvalues via the general QR iteration.
    vec::dense_vector<complex_type> eigs = eigenvalue(A, tol, max_iter);

    mat::dense2D<complex_type> V(n, n);
    if (n == 1) {
        V(0, 0) = complex_type(value_type(1));
        return EigenResult{eigs, V};
    }

    // Frobenius norm of A -- sets the scale for the pivot floor.
    value_type anorm = value_type(0);
    for (size_type i = 0; i < n; ++i)
        for (size_type j = 0; j < n; ++j)
            anorm += A(i, j) * A(i, j);
    anorm = sqrt(anorm);
    if (anorm == value_type(0)) anorm = value_type(1);
    const value_type eps = std::numeric_limits<value_type>::epsilon();
    const value_type pivot_floor = eps * anorm;

    const std::size_t nn = static_cast<std::size_t>(n);
    const int max_inv_iter = 5;

    // Two eigenvalues are treated as one cluster (a repeated eigenvalue) when
    // they agree to within this scale. Kept tiny so genuinely distinct
    // eigenvalues -- including complex-conjugate partners -- are never merged.
    const value_type cluster_scale = sqrt(eps);

    for (size_type k = 0; k < n; ++k) {
        const complex_type lambda = eigs(k);
        const value_type cluster_tol = cluster_scale * (anorm + std::abs(lambda));

        // Build M = A - lambda*I (row-major complex) and factor once.
        std::vector<complex_type> Mc(nn * nn);
        for (size_type i = 0; i < n; ++i)
            for (size_type j = 0; j < n; ++j)
                Mc[static_cast<std::size_t>(i) * nn + static_cast<std::size_t>(j)] =
                    complex_type(A(i, j)) - (i == j ? lambda : complex_type(0));

        std::vector<std::size_t> piv(nn);
        detail::eig_lu_factor(Mc, piv, nn, pivot_floor);

        // Modified Gram-Schmidt deflation of v against previously computed
        // eigenvectors that share this eigenvalue. For a repeated eigenvalue the
        // whole eigenspace is (near-)null for M, so plain inverse iteration would
        // return the same vector each time; deflating keeps successive iterates
        // in the orthogonal complement of what is already found, yielding an
        // independent basis of the eigenspace. (Orthonormal combinations within
        // one eigenspace are still eigenvectors.)
        auto deflate = [&](std::vector<complex_type>& v) {
            for (size_type m = 0; m < k; ++m) {
                if (std::abs(eigs(m) - lambda) > cluster_tol) continue;
                complex_type proj(0);
                for (size_type i = 0; i < n; ++i)
                    proj += std::conj(V(i, m)) * v[static_cast<std::size_t>(i)];
                for (size_type i = 0; i < n; ++i)
                    v[static_cast<std::size_t>(i)] -= proj * V(i, m);
            }
        };
        auto norm2 = [&](const std::vector<complex_type>& v) {
            value_type s = value_type(0);
            for (std::size_t i = 0; i < nn; ++i) s += std::norm(v[i]);
            return sqrt(s);
        };

        // Seed: index-varied base plus a k-dependent emphasis, so repeated
        // eigenvalues start from non-parallel vectors. Reseed if a seed happens
        // to lie in the span of the already-found cluster vectors.
        std::vector<complex_type> x(nn);
        for (int attempt = 0; attempt <= static_cast<int>(nn); ++attempt) {
            for (size_type i = 0; i < n; ++i)
                x[static_cast<std::size_t>(i)] =
                    complex_type(value_type(1) + value_type(i) / value_type(n));
            x[(static_cast<std::size_t>(k) + static_cast<std::size_t>(attempt)) % nn]
                += complex_type(value_type(1));
            deflate(x);
            value_type s = norm2(x);
            if (s > pivot_floor) {
                for (std::size_t i = 0; i < nn; ++i) x[i] /= s;
                break;
            }
        }

        for (int it = 0; it < max_inv_iter; ++it) {
            detail::eig_lu_solve(Mc, piv, nn, x);
            deflate(x);
            value_type nrm = norm2(x);
            if (nrm == value_type(0)) break;
            for (std::size_t i = 0; i < nn; ++i) x[i] /= nrm;

            // Residual ||A x - lambda x|| against the (unperturbed) eigenvalue.
            value_type res = value_type(0);
            for (size_type i = 0; i < n; ++i) {
                complex_type axi(0);
                for (size_type j = 0; j < n; ++j)
                    axi += complex_type(A(i, j)) * x[static_cast<std::size_t>(j)];
                axi -= lambda * x[static_cast<std::size_t>(i)];
                res += std::norm(axi);
            }
            if (sqrt(res) <= tol * anorm) break;
        }

        // Canonical phase: rotate so the largest-magnitude entry is real-positive.
        std::size_t imax = 0;
        value_type mmax = value_type(0);
        for (std::size_t i = 0; i < nn; ++i) {
            value_type m = abs(x[i]);
            if (m > mmax) { mmax = m; imax = i; }
        }
        if (mmax > value_type(0)) {
            complex_type phase = x[imax] / abs(x[imax]);
            for (std::size_t i = 0; i < nn; ++i) x[i] /= phase;
        }

        for (size_type i = 0; i < n; ++i)
            V(i, k) = x[static_cast<std::size_t>(i)];
    }

    return EigenResult{eigs, V};
}

} // namespace mtl
