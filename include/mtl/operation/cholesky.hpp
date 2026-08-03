#pragma once
// MTL5 -- Cholesky factorization for positive definite matrices
//
// LL^T vs LL^H (#353). This file provides BOTH, and they are not
// interchangeable for complex elements:
//
//   cholesky_factor / cholesky_solve      A = L*L^T   -- REAL symmetric
//                                         positive definite only
//   cholesky_h_factor / cholesky_h_solve  A = L*L^H   -- real symmetric OR
//                                         HERMITIAN positive definite
//
// For real elements conjugation is the identity, so the two agree. For complex
// elements only the LL^H form exists: "positive definite" is a statement about
// an ordering, and a complex SYMMETRIC matrix has no real diagonal to order.
// That is why cholesky_factor is restricted to real element types rather than
// given a complex-symmetric variant the way ldlt_factor has one -- see the
// static_assert below.
//
// In-place: lower triangle of A is overwritten with L.
// Optional LAPACK dispatch when MTL5_HAS_LAPACK is defined and types qualify.
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <type_traits>
#include <mtl/concepts/matrix.hpp>
#include <mtl/concepts/vector.hpp>
#include <mtl/concepts/magnitude.hpp>
#include <mtl/concepts/scalar.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/math/identity.hpp>
#include <mtl/functor/scalar/conj.hpp>
#include <mtl/functor/scalar/real.hpp>
#include <mtl/functor/scalar/imag.hpp>
#include <mtl/detail/structure_tol.hpp>
#include <mtl/interface/dispatch_traits.hpp>
#include <mtl/detail/thread_pool.hpp>
#ifdef MTL5_HAS_LAPACK
#include <mtl/interface/lapack.hpp>
#endif

namespace mtl {

/// Returned by cholesky_h_factor when the input is complex and has a non-real
/// diagonal, which a Hermitian matrix cannot have. Negative so it cannot
/// collide with the k+1 not-positive-definite codes.
inline constexpr int CHOLESKY_NOT_HERMITIAN = -2;

/// Cholesky factorization: A = L * L^T, for a REAL symmetric positive definite A.
/// The lower triangle of A is overwritten with L. Upper triangle is untouched.
/// Returns 0 on success, k+1 if A(k,k) <= 0 (not SPD).
///
/// Complex element types are rejected at compile time; use cholesky_h_factor.
template <Matrix M>
int cholesky_factor(M& A) {
    using value_type = typename M::value_type;
    using size_type  = typename M::size_type;
    using std::sqrt;
    const size_type n = A.num_rows();
    assert(A.num_cols() == n);

    // This routine computes A = L*L^T and tests the diagonal for positive
    // definiteness. Neither half survives a complex element type:
    //
    //   - the test `diag <= 0` is an ORDERING, and complex is not ordered. It
    //     used to fail here with an opaque "no match for operator<=" (#353).
    //   - even with the comparison repaired, the unconjugated inner product
    //     `sum += A(j,k) * A(j,k)` computes L*L^T, which is not the
    //     factorization a Hermitian matrix has.
    //
    // Repairing only the comparison would therefore convert a compile error
    // into a silently wrong answer -- exactly what happened to ldlt in #352.
    // Reject instead, and point at the routine that is correct for that input.
    static_assert(!is_complex_v<value_type>,
        "cholesky_factor computes A = L*L^T and orders the diagonal to test positive "
        "definiteness, neither of which is meaningful for a complex element type. For a "
        "HERMITIAN positive definite matrix use cholesky_h_factor / cholesky_h_solve, "
        "which compute A = L*L^H (#353).");

#ifdef MTL5_HAS_LAPACK
    if constexpr (interface::BlasDenseMatrix<M> && !interface::is_row_major_v<M>) {
        int n_int = static_cast<int>(n);
        int info = interface::lapack::potrf('L', n_int,
                       const_cast<value_type*>(A.data()), n_int);
        return (info > 0) ? info : 0;
    }
#endif

    for (size_type j = 0; j < n; ++j) {
        // Compute L(j,j) = sqrt(A(j,j) - sum(L(j,k)^2 for k < j))
        auto sum = math::zero<value_type>();
        for (size_type k = 0; k < j; ++k)
            sum += A(j, k) * A(j, k);
        auto diag = A(j, j) - sum;
        if (diag <= math::zero<value_type>())
            return static_cast<int>(j + 1);
        A(j, j) = sqrt(diag);

        // Compute L(i,j) for i > j -- parallelize over rows i (each A(i,j) is
        // written by exactly one chunk and reads only already-finalized columns
        // k < j and the shared column j, so contiguous chunking is bit-identical
        // to the serial path). No-op team at MTL5_NUM_THREADS=1.
        const size_type rows = n - j - 1;
        if (rows > 0) {
            const std::size_t inner = static_cast<std::size_t>(j == 0 ? 1 : j);
            const std::size_t grain =
                std::max<std::size_t>(std::size_t{1}, std::size_t{65536} / inner);
            detail::thread_pool::instance().parallel_for(
                static_cast<std::size_t>(rows), grain,
                [&](std::size_t b, std::size_t e) {
                    for (std::size_t t = b; t < e; ++t) {
                        const size_type i = j + 1 + static_cast<size_type>(t);
                        auto s = math::zero<value_type>();
                        for (size_type k = 0; k < j; ++k)
                            s += A(i, k) * A(j, k);
                        A(i, j) = (A(i, j) - s) / A(j, j);
                    }
                });
        }
    }
    return 0;
}

/// Solve A*x = b using precomputed Cholesky factor L (stored in lower triangle of A).
/// Solves L*y = b (forward), then L^T*x = y (backward).
template <Matrix M, Vector VecX, Vector VecB>
void cholesky_solve(const M& L, VecX& x, const VecB& b) {
    using value_type = typename VecX::value_type;
    using size_type  = typename M::size_type;
    const size_type n = L.num_rows();
    assert(L.num_cols() == n && x.size() == n && b.size() == n);

    // Forward substitution: L*y = b
    for (size_type i = 0; i < n; ++i) {
        auto sum = math::zero<value_type>();
        for (size_type j = 0; j < i; ++j)
            sum += L(i, j) * x(j);
        x(i) = (b(i) - sum) / L(i, i);
    }

    // Back substitution: L^T*x = y
    for (size_type ii = 0; ii < n; ++ii) {
        size_type i = n - 1 - ii;
        auto sum = math::zero<value_type>();
        for (size_type j = i + 1; j < n; ++j)
            sum += L(j, i) * x(j);  // L^T(i,j) = L(j,i)
        x(i) = (x(i) - sum) / L(i, i);
    }
}


/// Cholesky factorization: A = L * L^H, for a HERMITIAN positive definite A.
///
/// The lower triangle of A is overwritten with L, whose diagonal is REAL for a
/// Hermitian A and is stored in the element type with a zero imaginary part.
/// Only the lower triangle is read, so the caller's upper triangle is
/// irrelevant and need not be populated consistently.
///
/// Returns 0 on success, k+1 if the k-th pivot is not positive (not HPD), or
/// CHOLESKY_NOT_HERMITIAN if the diagonal is not real -- a Hermitian matrix
/// cannot have one, so that input does not belong here.
///
/// For real element types conjugation is the identity and this is exactly
/// cholesky_factor -- see the note at the top of this file (#353).
template <Matrix M>
int cholesky_h_factor(M& A) {
    using value_type = typename M::value_type;
    using size_type  = typename M::size_type;
    using mag_t      = magnitude_t<value_type>;
    using conj_t     = functor::scalar::conj<value_type>;
    using real_t     = functor::scalar::real<value_type>;
    using std::sqrt;
    const size_type n = A.num_rows();
    assert(A.num_cols() == n);

    // A Hermitian matrix has a REAL diagonal. If a complex-symmetric matrix
    // arrives here by mistake, taking the real part below would silently
    // discard its imaginary diagonal and factor a DIFFERENT matrix while
    // returning 0 -- the mirror of the ldlt bug in #352. The check is O(n),
    // reads only the diagonal (preserving the lower-triangle-only contract),
    // and is scale-relative because a Hermitian matrix assembled in floating
    // point carries a rounding-sized imaginary residue.
    if constexpr (is_complex_v<value_type>) {
        mag_t dscale(0), dimag(0);
        for (size_type j = 0; j < n; ++j) {
            const mag_t a = detail::cabs1(A(j, j));
            if (a > dscale) dscale = a;
            using std::abs;
            const mag_t im = abs(functor::scalar::imag<value_type>::apply(A(j, j)));
            if (im > dimag) dimag = im;
        }
        if (dimag > detail::structure_tol(static_cast<std::size_t>(n), dscale))
            return CHOLESKY_NOT_HERMITIAN;
    }

    // For j = 0..n-1:
    //   L(j,j) = sqrt( Re( A(j,j) ) - sum_{k<j} |L(j,k)|^2 )      -- real
    //   L(i,j) = ( A(i,j) - sum_{k<j} L(i,k) * conj(L(j,k)) ) / L(j,j)
    //
    // The pivot is real by construction: |L(j,k)|^2 is real and A(j,j) is real
    // for a Hermitian A. Accumulating it in the MAGNITUDE type rather than the
    // element type is what makes the positivity test well-formed -- it is the
    // ordering that does not exist for the element type itself.
    for (size_type j = 0; j < n; ++j) {
        mag_t s(0);
        for (size_type k = 0; k < j; ++k)
            // |L(j,k)|^2: the product is real by construction, so take its real
            // part rather than abs() (no sqrt) or cabs1() (which would add the
            // rounding-sized imaginary residue instead of discarding it).
            s += real_t::apply(value_type(A(j, k) * conj_t::apply(A(j, k))));
        const mag_t diag = real_t::apply(A(j, j)) - s;
        if (!(diag > mag_t(0)))
            return static_cast<int>(j + 1);
        const mag_t ljj = sqrt(diag);
        A(j, j) = value_type(ljj);

        // Rows i > j are independent: each writes only A(i,j) and reads already
        // finalized columns k < j plus the shared column j -- so chunking is
        // bit-identical to the serial path, as in cholesky_factor above.
        const size_type rows = n - j - 1;
        if (rows > 0) {
            const std::size_t inner = static_cast<std::size_t>(j == 0 ? 1 : j);
            const std::size_t grain =
                std::max<std::size_t>(std::size_t{1}, std::size_t{65536} / inner);
            detail::thread_pool::instance().parallel_for(
                static_cast<std::size_t>(rows), grain,
                [&](std::size_t b, std::size_t e) {
                    for (std::size_t t = b; t < e; ++t) {
                        const size_type i = j + 1 + static_cast<size_type>(t);
                        auto sum = math::zero<value_type>();
                        for (size_type k = 0; k < j; ++k)
                            sum += A(i, k) * conj_t::apply(A(j, k));
                        A(i, j) = (A(i, j) - sum) / value_type(ljj);
                    }
                });
        }
    }
    return 0;
}

/// Solve A*x = b using precomputed Cholesky factors of a HERMITIAN A, i.e. the
/// L of A = L*L^H stored in the lower triangle by cholesky_h_factor.
/// Solves L*y = b (forward), then L^H*x = y (backward).
template <Matrix M, Vector VecX, Vector VecB>
void cholesky_h_solve(const M& L, VecX& x, const VecB& b) {
    using value_type = typename VecX::value_type;
    using size_type  = typename M::size_type;
    using conj_t     = functor::scalar::conj<value_type>;
    const size_type n = L.num_rows();
    assert(L.num_cols() == n && x.size() == n && b.size() == n);

    // Forward substitution: L*y = b
    for (size_type i = 0; i < n; ++i) {
        auto sum = math::zero<value_type>();
        for (size_type j = 0; j < i; ++j)
            sum += L(i, j) * x(j);
        x(i) = (b(i) - sum) / L(i, i);
    }

    // Back substitution: L^H*x = y, where L^H(i,j) = conj(L(j,i)).
    for (size_type ii = 0; ii < n; ++ii) {
        size_type i = n - 1 - ii;
        auto sum = math::zero<value_type>();
        for (size_type j = i + 1; j < n; ++j)
            sum += conj_t::apply(L(j, i)) * x(j);
        x(i) = (x(i) - sum) / L(i, i);
    }
}

} // namespace mtl
