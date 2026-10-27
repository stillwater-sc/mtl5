#pragma once
// MTL5 -- Real Schur decomposition A = Q T Q^T, and its complex form (#650)
//
// schur(A) reduces A to upper Hessenberg form, forms the orthogonal factor of
// that reduction, and runs the Francis double-shift QR iteration on the whole
// matrix (EISPACK hqr2), accumulating its transformations into Q. T is
// quasi-upper-triangular: 1x1 blocks hold the real eigenvalues, and 2x2 blocks
// in LAPACK's standard form (equal diagonal, off-diagonals of opposite sign)
// hold the complex-conjugate pairs.
//
// The in-house path is generic in the element type: it needs + - * /,
// comparisons, and abs and sqrt by ADL, so it runs for posits, cfloats, LNS,
// double-double and the other user-defined real types. dense2D<float/double>
// dispatch to LAPACK gees when MTL5_HAS_LAPACK is defined.
#include <cmath>
#include <complex>
#include <concepts>
#include <limits>
#include <stdexcept>
#include <vector>
#include <mtl/concepts/matrix.hpp>
#include <mtl/concepts/scalar.hpp>
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

/// The factors of a Schur decomposition A = Q T Q^H.
template <typename Value>
struct schur_result {
    mat::dense2D<Value> Q;  ///< orthogonal (unitary) Schur vectors
    mat::dense2D<Value> T;  ///< quasi-upper-triangular (upper triangular if complex)
};

namespace detail {

#ifdef MTL5_HAS_LAPACK
/// LAPACK gees on a real float/double dense matrix, with Schur vectors and no
/// reordering. Its 2x2 blocks are in the same standard form as the in-house
/// path's.
template <typename M>
schur_result<typename M::value_type> lapack_gees(const M& A) {
    using value_type = typename M::value_type;
    using size_type  = typename M::size_type;
    const size_type n = A.num_rows();
    const int ni = static_cast<int>(n);
    const std::size_t nn = static_cast<std::size_t>(n);

    // Column-major copy through the (i,j) accessor, so any orientation works.
    std::vector<value_type> a(nn * nn), vs(nn * nn), wr(nn), wi(nn);
    for (size_type i = 0; i < n; ++i)
        for (size_type j = 0; j < n; ++j)
            a[static_cast<std::size_t>(j) * nn + i] = A(i, j);
    std::vector<int> bwork(nn);   // not referenced when unsorted

    value_type wq = value_type(0);
    interface::lapack::gees('V', ni, a.data(), ni, wr.data(), wi.data(),
                            vs.data(), ni, &wq, -1, bwork.data());
    int lwork = static_cast<int>(wq);
    if (lwork < 3 * ni) lwork = 3 * ni;   // LAPACK minimum
    std::vector<value_type> work(static_cast<std::size_t>(lwork));
    const int info = interface::lapack::gees('V', ni, a.data(), ni, wr.data(), wi.data(),
                                             vs.data(), ni, work.data(), lwork, bwork.data());
    if (info != 0)
        throw std::runtime_error("mtl::schur: LAPACK gees failed to converge");

    schur_result<value_type> r{mat::dense2D<value_type>(n, n), mat::dense2D<value_type>(n, n)};
    for (size_type i = 0; i < n; ++i)
        for (size_type j = 0; j < n; ++j) {
            const std::size_t at = static_cast<std::size_t>(j) * nn + i;
            r.Q(i, j) = vs[at];
            r.T(i, j) = (i > j + 1) ? value_type(0) : a[at];
        }
    return r;
}
#endif // MTL5_HAS_LAPACK

} // namespace detail

/// Real Schur decomposition of a square matrix: A = Q T Q^T with Q orthogonal
/// and T quasi-upper-triangular. T's 1x1 diagonal blocks are the real
/// eigenvalues; each 2x2 block [a b; c a] with b*c < 0 holds the complex pair
/// a +- i sqrt(-b*c). Use as `auto [Q, T] = mtl::schur(A);`.
///
/// Throws std::invalid_argument if A is not square or has a NaN or infinite
/// entry, and std::runtime_error if the QR iteration does not converge.
///
/// In-house path, for every real element type:
///   - The deflation test is relative to the type's own epsilon (measured for
///     a type without numeric_limits), with LAPACK's safe-minimum floor when
///     numeric_limits gives one that is meaningful.
///   - A is first scaled by a power of two to max |a_ij| in [1/2, 1], and T is
///     scaled back at the end. The EISPACK iteration forms products of two
///     matrix entries, which overflow or underflow for an IEEE matrix with
///     entries near 1e200 or 1e-200; scaling makes them safe, and is exact in a
///     binary floating-point type. For a posit, whose precision is greatest
///     near 1, it also keeps the deflation test reachable.
///   - A fixed-point type has no dynamic range: the products in the iteration
///     must stay representable, so A should be scaled well inside the range.
///
/// Dispatches to LAPACK gees when MTL5_HAS_LAPACK is defined and the type
/// qualifies (dense2D<float/double>, either orientation). `Accumulator`
/// (leading, default void) is forwarded to the Hessenberg reduction and the
/// formation of its Q; a non-void accumulator takes the in-house path. The QR
/// iteration's updates are fixed 2- and 3-term expressions and keep plain
/// arithmetic.
template <typename Accumulator = void, FieldMatrix M>
auto schur(const M& A) {
    using value_type = typename M::value_type;
    using size_type  = typename M::size_type;
    using std::abs;
    static_assert(!is_complex_v<value_type>,
                  "mtl::schur: the complex Schur form of a complex matrix is not "
                  "available yet (#650); schur() takes a real element type");
    const size_type n = A.num_rows();
    if (A.num_cols() != n)
        throw std::invalid_argument("mtl::schur: matrix must be square");

    for (size_type i = 0; i < n; ++i)
        for (size_type j = 0; j < n; ++j) {
            const value_type a = A(i, j);
            if (!(a - a == value_type(0)))
                throw std::invalid_argument("mtl::schur: matrix has a NaN or infinite entry");
        }

    if (n == 0)
        return schur_result<value_type>{mat::dense2D<value_type>(0, 0), mat::dense2D<value_type>(0, 0)};

#ifdef MTL5_HAS_LAPACK
    if constexpr (interface::BlasDenseMatrix<M> &&
                  interface::accumulator_allows_blas_v<Accumulator>) {
        return detail::lapack_gees(A);
    }
#endif

    mat::dense2D<value_type> W(n, n);
    for (size_type i = 0; i < n; ++i)
        for (size_type j = 0; j < n; ++j) W(i, j) = A(i, j);

    // Scale by a power of two to max |a_ij| in [1/2, 1].
    value_type unscale = value_type(1);
    {
        value_type amax = value_type(0);
        for (size_type i = 0; i < n; ++i)
            for (size_type j = 0; j < n; ++j)
                if (amax < abs(W(i, j))) amax = abs(W(i, j));
        if (amax > value_type(0)) {
            value_type scale = value_type(1);
            const value_type two(2), half(0.5);
            // Bounded: each step moves amax * scale one binade.
            for (int k = 0; k < 100000 && amax * scale > value_type(1); ++k) {
                scale = scale * half; unscale = unscale * two;
            }
            for (int k = 0; k < 100000 && amax * scale < half; ++k) {
                scale = scale * two; unscale = unscale * half;
            }
            if (!(scale == value_type(1)))
                for (size_type i = 0; i < n; ++i)
                    for (size_type j = 0; j < n; ++j) W(i, j) = W(i, j) * scale;
        }
    }

    vec::dense_vector<value_type> tau;
    hessenberg_factor<Accumulator>(W, tau);
    schur_result<value_type> r{hessenberg_extract_Q<Accumulator>(W, tau), hessenberg_extract_H(W)};

    detail::francis_options<value_type> opt;
    opt.who = "mtl::schur";
    opt.tol = detail::machine_epsilon<value_type>();
    opt.lanv2 = detail::make_lanv2_constants<value_type>();
    opt.deflation_floor = value_type(0);
    if constexpr (std::numeric_limits<value_type>::is_specialized) {
        // LAPACK dlahqr's smlnum = safmin * (n / ulp). A fixed-point type's
        // smallest value is not a safe minimum in that sense and gives a floor
        // of order 1, so the floor is used only when it is below epsilon.
        const value_type floor =
            std::numeric_limits<value_type>::min() * (value_type(double(n)) / opt.tol);
        if (floor > value_type(0) && floor < opt.tol) opt.deflation_floor = floor;
    }
    std::vector<value_type> wr, wi;
    detail::francis_qr<true>(r.T, &r.Q, wr, wi, opt);

    // The bulge chase leaves stale entries below the subdiagonal.
    for (size_type i = 2; i < n; ++i)
        for (size_type j = 0; j + 1 < i; ++j) r.T(i, j) = value_type(0);

    if (!(unscale == value_type(1)))
        for (size_type i = 0; i < n; ++i)
            for (size_type j = 0; j < n; ++j) r.T(i, j) = r.T(i, j) * unscale;
    return r;
}

/// Convert a real Schur decomposition A = Q T Q^T into the complex one,
/// A = U R U^H with U unitary and R upper triangular, as MATLAB's rsf2csf does.
/// Each 2x2 block of T is split by a complex rotation; R's diagonal then holds
/// the eigenvalues, with the one of positive imaginary part first in each pair.
/// Element types with a std::complex (float, double, long double) for now; a
/// customization point for Universal's complex type follows (#650).
template <typename Value>
    requires std::floating_point<Value>
schur_result<std::complex<Value>> rsf2csf(const mat::dense2D<Value>& Q, const mat::dense2D<Value>& T) {
    using C = std::complex<Value>;
    using size_type = typename mat::dense2D<Value>::size_type;
    const size_type n = T.num_rows();
    if (T.num_cols() != n || Q.num_rows() != n || Q.num_cols() != n)
        throw std::invalid_argument("mtl::rsf2csf: Q and T must be square and of the same size");

    schur_result<C> r{mat::dense2D<C>(n, n), mat::dense2D<C>(n, n)};
    for (size_type i = 0; i < n; ++i)
        for (size_type j = 0; j < n; ++j) {
            r.Q(i, j) = C(Q(i, j));
            r.T(i, j) = C(T(i, j));
        }

    for (size_type m = n; m-- > 1; ) {
        if (r.T(m, m - 1) == C(0)) continue;
        // mu = lambda - T(m, m) for the block's eigenvalue lambda of positive
        // imaginary part, which the rotation moves to position m - 1.
        const C a = r.T(m - 1, m - 1), b = r.T(m - 1, m), c = r.T(m, m - 1), d = r.T(m, m);
        const C half_diff = (a - d) / Value(2);
        C disc = std::sqrt(half_diff * half_diff + b * c);
        if (disc.imag() < Value(0)) disc = -disc;
        const C mu = half_diff + disc;
        const Value rr = detail::lapy2(std::abs(mu), std::abs(c));
        const C cs = mu / rr;
        const C sn = c / rr;
        // [cs; sn] is the unit eigenvector [lambda - d; c] / rr of the block,
        // so G = [conj(cs) conj(sn); -sn cs] has it as G^H's first column and
        // G T G^H zeroes T(m, m-1). Rows m-1, m of T, then columns m-1, m of T
        // and of U (U = U G^H).
        for (size_type j = m - 1; j < n; ++j) {
            const C x = r.T(m - 1, j), y = r.T(m, j);
            r.T(m - 1, j) = std::conj(cs) * x + std::conj(sn) * y;
            r.T(m, j)     = -sn * x + cs * y;
        }
        for (size_type i = 0; i <= m; ++i) {
            const C x = r.T(i, m - 1), y = r.T(i, m);
            r.T(i, m - 1) = x * cs + y * sn;
            r.T(i, m)     = -x * std::conj(sn) + y * std::conj(cs);
        }
        for (size_type i = 0; i < n; ++i) {
            const C x = r.Q(i, m - 1), y = r.Q(i, m);
            r.Q(i, m - 1) = x * cs + y * sn;
            r.Q(i, m)     = -x * std::conj(sn) + y * std::conj(cs);
        }
        r.T(m, m - 1) = C(0);
    }
    return r;
}

} // namespace mtl
