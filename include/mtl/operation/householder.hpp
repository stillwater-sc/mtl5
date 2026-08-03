#pragma once
// MTL5 -- Householder reflections for QR factorization
//
// The reflector is H = I - tau * v * v^H, with v(0) == 1 implicit, chosen so
// that H*x = beta*e_1. Writing it with v^H rather than v^T is what makes the
// complex case work, and costs the real case nothing: conjugation is the
// identity there, so H = I - tau*v*v^T exactly as before (#353).
//
// H is UNITARY but, for complex tau, NOT Hermitian -- so H^-1 = H^H, not H.
// That distinction is invisible in the real case and is the whole reason the
// consumers need conj(tau) in some places: see qr_extract_Q / lq_extract_Q.
#include <algorithm>
#include <cmath>
#include <cassert>
#include <cstddef>
#include <type_traits>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/concepts/matrix.hpp>
#include <mtl/concepts/magnitude.hpp>
#include <mtl/concepts/scalar.hpp>
#include <mtl/math/identity.hpp>
#include <mtl/functor/scalar/conj.hpp>
#include <mtl/functor/scalar/real.hpp>
#include <mtl/functor/scalar/imag.hpp>
#include <mtl/detail/thread_pool.hpp>

namespace mtl {

/// Compute Householder vector v and scalar tau for a column vector x, such
/// that H = I - tau*v*v^H satisfies H*x = beta*e_1. v(0) is always 1
/// (implicit). Returns {v, tau}. For complex x, beta is real by construction.
template <typename T>
std::pair<vec::dense_vector<T>, T> householder(const vec::dense_vector<T>& x) {
    using std::sqrt;
    using std::abs;
    using size_type = typename vec::dense_vector<T>::size_type;
    const size_type n = x.size();

    vec::dense_vector<T> v(n);
    if (n == 0) return {v, math::zero<T>()};   // no element at index 0 to write

    // Scale by the largest magnitude before squaring. Forming sum(x(i)^2)
    // directly underflows to zero once the entries are small -- and the old
    // `sigma == 0` test only caught EXACT zero, so a merely tiny sigma fell
    // through to `beta = 2*v0*v0 / (sigma + v0*v0)` with both terms flushed to
    // zero, i.e. 0/0 = NaN, plus `v(i) /= v0` overflowing to infinity.
    //
    // The SVD's alternating-QR iteration drives exactly that: as it converges
    // the trailing columns shrink toward zero, so the reflectors underflow and
    // poison an already-correct answer with NaN (#337). Scaling keeps the
    // squares O(1), so the only way sigma vanishes now is that x(1:) really is
    // negligible against x(0) -- which is the genuine "already along e_1" case.
    // Magnitudes live in magnitude_t<T>, not T. For a real T the two are the
    // same type and this is a no-op; for a complex T it is the difference
    // between compiling and not -- `axi > scale` on two complex values was one
    // of the errors reported in #353, and the fix is a type, not a cast.
    using mag_t = magnitude_t<T>;
    using conj_t = functor::scalar::conj<T>;

    mag_t scale = mag_t(0);
    for (size_type i = 0; i < n; ++i) {
        const mag_t axi = abs(x(i));
        if (axi > scale) scale = axi;
    }

    for (size_type i = 0; i < n; ++i)
        v(i) = x(i);
    v(0) = math::one<T>();

    if (scale == mag_t(0))
        return {v, math::zero<T>()};   // x is the zero vector

    // sigma = sum |x(i)/scale|^2 for i >= 1, in the scaled variables.
    // |z|^2, NOT z^2: the squared modulus is what a norm needs, and for real T
    // the two coincide, which is why the unconjugated form survived this long.
    mag_t sigma = mag_t(0);
    for (size_type i = 1; i < n; ++i) {
        const T xs = x(i) / scale;
        sigma += functor::scalar::real<T>::apply(T(xs * conj_t::apply(xs)));
    }

    // x is already along e_1. For a real T that IS the identity reflection.
    // For a complex T it is not, unless x(0) happens to be real: this function
    // promises a REAL beta, and a complex x(0) still has a phase to remove.
    // Returning tau = 0 here would leave H*x = x(0)*e_1 with a complex leading
    // entry, which propagates straight into a complex R diagonal from qr_factor
    // and a complex L diagonal from lq_factor -- silent, because Q stays
    // unitary and the reconstruction still holds. LAPACK zlarfg draws the line
    // in the same place: tau = 0 only when the tail vanishes AND alpha is real.
    //
    // Falling through is correct and needs no special case: with sigma == 0 the
    // complex branch gets nrm = |x0|, a well-conditioned d = x0 - beta of
    // magnitude 2|x0| (beta takes the opposite sign of Re(x0)), and a tail loop
    // that writes only zeros. x0 cannot be zero here -- scale > 0 and sigma == 0
    // force |x0| == scale.
    if (sigma == mag_t(0)) {
        if constexpr (!is_complex_v<T>) {
            return {v, math::zero<T>()};
        } else if (functor::scalar::imag<T>::apply(x(0)) == mag_t(0)) {
            return {v, math::zero<T>()};
        }
    }

    const T x0 = x(0) / scale;

    if constexpr (!is_complex_v<T>) {
        // Real path, arithmetic UNCHANGED. Kept verbatim rather than folded
        // into the general formulation so that the existing results stay
        // bit-identical -- see the agreement test in test_householder.cpp.
        const T norm_x = sqrt(x0 * x0 + sigma);
        T v0;
        if (x0 <= math::zero<T>())
            v0 = x0 - norm_x;
        else
            v0 = -sigma / (x0 + norm_x);

        const T beta = T(2) * v0 * v0 / (sigma + v0 * v0);

        // beta == 0 means the reflection is the identity, so v carries no
        // information -- and computing it would divide by a v0 small enough to
        // overflow. Return a clean unit v: qr_factor STORES v below the diagonal,
        // so letting huge or infinite entries through would poison the factor and
        // every later use of it.
        if (beta == math::zero<T>())
            return {v, math::zero<T>()};

        // Normalize so that v(0) = 1. Both numerator and denominator are in the
        // scaled variables, so the ratio is the same as before the scaling.
        for (size_type i = 1; i < n; ++i)
            v(i) = (x(i) / scale) / v0;
        v(0) = math::one<T>();

        return {v, beta};
    } else {
        // Complex path (LAPACK zlarfg's construction).
        //
        // With v = [1; u], x = [a; y] and H = I - tau*v*v^H, requiring
        // H*x = beta*e_1 with beta REAL gives, after substituting
        // |a|^2 + ||y||^2 = beta^2,
        //
        //     u   = y / (a - beta)
        //     tau = (beta - conj(a)) / beta
        //
        // beta's SIGN is the free choice, and it is chosen to keep a - beta
        // away from cancellation -- the complex analogue of the real branch's
        // `x0 <= 0` test. Picking the opposite sign of Re(a) makes |a - beta|
        // >= beta, so the division below never amplifies rounding.
        const mag_t ax0  = abs(x0);
        const mag_t nrm  = sqrt(ax0 * ax0 + sigma);
        const mag_t beta = (functor::scalar::real<T>::apply(x0) > mag_t(0)) ? -nrm : nrm;
        const T     d    = x0 - T(beta);          // a - beta, no cancellation
        const T     tau  = (T(beta) - conj_t::apply(x0)) / T(beta);

        // Identity reflection -- same reasoning as the real branch.
        if (tau == math::zero<T>())
            return {v, math::zero<T>()};

        for (size_type i = 1; i < n; ++i)
            v(i) = (x(i) / scale) / d;
        v(0) = math::one<T>();

        return {v, tau};
    }
}

/// Apply Householder reflection H = I - beta*v*v^H on the LEFT (A := H*A)
/// to columns col..ncols-1
/// of matrix A, rows row..nrows-1. Modifies A in-place.
template <Matrix M, typename T>
void apply_householder_left(M& A, const vec::dense_vector<T>& v, T beta,
                            typename M::size_type row, typename M::size_type col) {
    using size_type = typename M::size_type;
    const size_type n = A.num_cols();
    const size_type vlen = v.size();

    // Each column j is updated independently (reads the shared reflector v and
    // only column j of A, writes only column j), so partitioning the columns
    // across the thread pool is bit-identical to serial. No-op at
    // MTL5_NUM_THREADS=1.
    // Empty or beyond-end target column: no work (matches the serial loop, and
    // avoids an unsigned underflow in n - col before deriving the work range).
    if (col >= n) return;
    // beta == 0 is the identity reflection. Returning early is not just an
    // optimisation: `beta * v(i) * w` would evaluate 0 * inf = NaN if any
    // reflector entry had overflowed.
    if (beta == math::zero<T>()) return;
    const size_type ncols = n - col;
    const std::size_t grain = std::max<std::size_t>(
        std::size_t{1}, std::size_t{65536} / (vlen ? static_cast<std::size_t>(vlen) : std::size_t{1}));
    detail::thread_pool::instance().parallel_for(
        static_cast<std::size_t>(ncols), grain,
        [&](std::size_t b, std::size_t e) {
            for (std::size_t t = b; t < e; ++t) {
                const size_type j = col + static_cast<size_type>(t);
                // w = v^H * A(:,j).  conj on v, not on A: for a real T this
                // is the identity and the arithmetic is unchanged.
                T w = math::zero<T>();
                for (size_type i = 0; i < vlen; ++i)
                    w += functor::scalar::conj<T>::apply(v(i)) * A(row + i, j);
                // A(:,j) -= beta * v * w
                for (size_type i = 0; i < vlen; ++i)
                    A(row + i, j) -= beta * v(i) * w;
            }
        });
}

/// Apply Householder reflection on the right: A := A * (I - beta*v*v^H).
/// NOTE this is A*H, not A*H^H -- for complex tau those differ, and a caller
/// wanting the inverse/adjoint must pass conj(beta) itself.
/// Modifies columns col..col+vlen-1 of rows row..nrows-1.
template <Matrix M, typename T>
void apply_householder_right(M& A, const vec::dense_vector<T>& v, T beta,
                             typename M::size_type row, typename M::size_type col) {
    using size_type = typename M::size_type;
    const size_type m = A.num_rows();
    const size_type vlen = v.size();

    // Each row i is updated independently (reads the shared reflector v and only
    // its own vlen entries of A, writes only those), so partitioning the rows
    // across the thread pool is bit-identical to serial. No-op at
    // MTL5_NUM_THREADS=1.
    // Empty or beyond-end target row: no work (matches the serial loop, and
    // avoids an unsigned underflow in m - row before deriving the work range).
    if (row >= m) return;
    // Identity reflection -- see the note in apply_householder_left.
    if (beta == math::zero<T>()) return;
    const size_type nrows = m - row;
    const std::size_t grain = std::max<std::size_t>(
        std::size_t{1}, std::size_t{65536} / (vlen ? static_cast<std::size_t>(vlen) : std::size_t{1}));
    detail::thread_pool::instance().parallel_for(
        static_cast<std::size_t>(nrows), grain,
        [&](std::size_t b, std::size_t e) {
            for (std::size_t t = b; t < e; ++t) {
                const size_type i = row + static_cast<size_type>(t);
                // w = A(i,:) * v
                T w = math::zero<T>();
                for (size_type j = 0; j < vlen; ++j)
                    w += A(i, col + j) * v(j);
                // A(i,:) -= beta * w * v^H.  A*H = A - beta*(A*v)*v^H, so the
                // conjugate lands on the OUTER v here, mirroring the inner one
                // in apply_householder_left. Identity for a real T.
                for (size_type j = 0; j < vlen; ++j)
                    A(i, col + j) -= beta * w * functor::scalar::conj<T>::apply(v(j));
            }
        });
}

} // namespace mtl
