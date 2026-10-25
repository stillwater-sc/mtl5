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
//
// Mixed precision (#585): the factor and solve routines take a leading
// `Accumulator` (default void, see math/accumulator_traits.hpp); void is
// byte-identical to the routines before #585. A non-void accumulator never
// dispatches to LAPACK. In the factorizations it forms each entry as ONE
// reduction, A(i,j) - sum_k L(i,k) conj(L(j,k)), rounded once, as the sparse
// Cholesky does (#531); there is then no summation order to choose, so the
// tag::subtract_* overloads stay void-only. In the solves it forms each row's
// b(i) - sum_j L(i,j) x(j) the same way, rounded once before the division, as
// the sparse level-scheduled solves do (#535).
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <type_traits>
#include <vector>
#include <mtl/concepts/matrix.hpp>
#include <mtl/concepts/vector.hpp>
#include <mtl/concepts/magnitude.hpp>
#include <mtl/concepts/scalar.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/math/identity.hpp>
#include <mtl/math/accumulator_traits.hpp>
#include <mtl/functor/scalar/conj.hpp>
#include <mtl/functor/scalar/real.hpp>
#include <mtl/functor/scalar/imag.hpp>
#include <mtl/detail/structure_tol.hpp>
#include <mtl/interface/dispatch_traits.hpp>
#include <mtl/detail/thread_pool.hpp>
#include <mtl/tag/summation_order.hpp>
#ifdef MTL5_HAS_LAPACK
#include <mtl/interface/lapack.hpp>
#endif

namespace mtl {

/// Returned by cholesky_h_factor when the input is complex and has a non-real
/// diagonal, which a Hermitian matrix cannot have. Negative so it cannot
/// collide with the k+1 not-positive-definite codes.
inline constexpr int CHOLESKY_NOT_HERMITIAN = -2;

namespace detail {

/// The generic (non-LAPACK) A = L*L^T loop, in either summation order (#556).
///
///   tag::subtract_sum            L(i,j) = (A(i,j) - sum_k L(i,k) L(j,k)) / L(j,j)
///   tag::subtract_term_by_term   s = A(i,j); s = s - L(i,k) L(j,k), k = 0..j-1;
///                                L(i,j) = s / L(j,j)
///
/// and the same two forms for the pivot. The products and the order of k are
/// the same in both; only where the matrix entry enters the sum differs, so the
/// two round differently and neither is more accurate across the board
/// (docs/numerical/summation-order-in-cholesky.md).
template <typename Ordering, typename M>
int cholesky_factor_generic(M& A) {
    using value_type = typename M::value_type;
    using size_type  = typename M::size_type;
    using std::sqrt;
    constexpr bool term_by_term = std::is_same_v<Ordering, tag::subtract_term_by_term>;
    static_assert(term_by_term || std::is_same_v<Ordering, tag::subtract_sum>,
                  "cholesky_factor: Ordering must be tag::subtract_sum or tag::subtract_term_by_term");

    // This routine computes A = L*L^T and tests the diagonal for positive
    // definiteness. Neither half survives a complex element type:
    //
    //   - the pivot test is an ORDERING, and complex is not ordered. It used to
    //     fail here with an opaque "no match for operator<=" (#353).
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

    const size_type n = A.num_rows();
    for (size_type j = 0; j < n; ++j) {
        // Pivot: A(j,j) - sum(L(j,k)^2 for k < j), in the requested order.
        const value_type diag = [&] {
            if constexpr (term_by_term) {
                value_type d = A(j, j);
                for (size_type k = 0; k < j; ++k)
                    d = d - A(j, k) * A(j, k);
                return d;
            } else {
                auto sum = math::zero<value_type>();
                for (size_type k = 0; k < j; ++k)
                    sum += A(j, k) * A(j, k);
                return value_type(A(j, j) - sum);
            }
        }();
        // !(diag > 0) rather than diag <= 0: a NaN pivot fails every comparison,
        // so `diag <= 0` let it through and returned 0 with NaNs in L. This is
        // the test cholesky_h_factor already uses (#556).
        if (!(diag > math::zero<value_type>()))
            return static_cast<int>(j + 1);
        A(j, j) = sqrt(diag);

        // Compute L(i,j) for i > j -- parallelize over rows i (each A(i,j) is
        // written by exactly one chunk and reads only already-finalized columns
        // k < j and the shared column j, so contiguous chunking is bit-identical
        // to the serial path, in either order). No-op team at MTL5_NUM_THREADS=1.
        const size_type rows = n - j - 1;
        if (rows > 0) {
            const std::size_t inner = static_cast<std::size_t>(j == 0 ? 1 : j);
            const std::size_t grain =
                std::max<std::size_t>(std::size_t{1}, std::size_t{65536} / inner);
            ::mtl::parallel::pool().parallel_for(
                static_cast<std::size_t>(rows), grain,
                [&](std::size_t b, std::size_t e) {
                    for (std::size_t t = b; t < e; ++t) {
                        const size_type i = j + 1 + static_cast<size_type>(t);
                        if constexpr (term_by_term) {
                            value_type s = A(i, j);
                            for (size_type k = 0; k < j; ++k)
                                s = s - A(i, k) * A(j, k);
                            A(i, j) = s / A(j, j);
                        } else {
                            auto s = math::zero<value_type>();
                            for (size_type k = 0; k < j; ++k)
                                s += A(i, k) * A(j, k);
                            A(i, j) = (A(i, j) - s) / A(j, j);
                        }
                    }
                });
        }
    }
    return 0;
}

/// The accumulator A = L*L^H loop (#585): each entry is assign(A(i,j)), then
/// add_product(-L(i,k), conj(L(j,k))) for k < j, rounded once. For a real
/// element type conj is the identity and this is A = L*L^T. The pivot is the
/// real part of its round-out, tested for positivity before the square root.
/// Rows i > j are split over the pool as on the void path, bit-identically.
template <typename Accumulator, typename M>
int cholesky_factor_acc(M& A) {
    using value_type = typename M::value_type;
    using size_type  = typename M::size_type;
    using mag_t      = magnitude_t<value_type>;
    using conj_t     = functor::scalar::conj<value_type>;
    using real_t     = functor::scalar::real<value_type>;
    using AT         = math::accumulator_traits<Accumulator, value_type>;
    using std::sqrt;
    const size_type n = A.num_rows();

    for (size_type j = 0; j < n; ++j) {
        Accumulator dacc{};
        AT::assign(dacc, A(j, j));
        for (size_type k = 0; k < j; ++k)
            AT::add_product(dacc, value_type(-A(j, k)), conj_t::apply(A(j, k)));
        const mag_t diag = real_t::apply(AT::template value<value_type>(dacc));
        if (!(diag > mag_t(0)))
            return static_cast<int>(j + 1);
        const mag_t ljj = sqrt(diag);
        A(j, j) = value_type(ljj);

        const size_type rows = n - j - 1;
        if (rows > 0) {
            const std::size_t inner = static_cast<std::size_t>(j == 0 ? 1 : j);
            const std::size_t grain =
                std::max<std::size_t>(std::size_t{1}, std::size_t{65536} / inner);
            ::mtl::parallel::pool().parallel_for(
                static_cast<std::size_t>(rows), grain,
                [&](std::size_t b, std::size_t e) {
                    for (std::size_t t = b; t < e; ++t) {
                        const size_type i = j + 1 + static_cast<size_type>(t);
                        Accumulator acc{};
                        AT::assign(acc, A(i, j));
                        for (size_type k = 0; k < j; ++k)
                            AT::add_product(acc, value_type(-A(i, k)), conj_t::apply(A(j, k)));
                        A(i, j) = AT::template value<value_type>(acc) / value_type(ljj);
                    }
                });
        }
    }
    return 0;
}

/// The two substitutions of an L*L^H (Conj = true) or L*L^T (false) solve, with
/// each row's b(i) - sum of products formed in Accumulator and rounded once.
template <typename Accumulator, bool Conj, typename M, typename VecX, typename VecB>
void cholesky_solve_acc(const M& L, VecX& x, const VecB& b) {
    using value_type = typename VecX::value_type;
    using size_type  = typename M::size_type;
    using conj_t     = functor::scalar::conj<value_type>;
    using AT         = math::accumulator_traits<Accumulator, value_type>;
    const size_type n = L.num_rows();

    for (size_type i = 0; i < n; ++i) {
        Accumulator acc{};
        AT::assign(acc, value_type(b(i)));
        for (size_type j = 0; j < i; ++j)
            AT::add_product(acc, value_type(-L(i, j)), value_type(x(j)));
        x(i) = AT::template value<value_type>(acc) / L(i, i);
    }
    for (size_type ii = 0; ii < n; ++ii) {
        const size_type i = n - 1 - ii;
        Accumulator acc{};
        AT::assign(acc, value_type(x(i)));
        for (size_type j = i + 1; j < n; ++j) {
            const value_type lji = L(j, i);
            AT::add_product(acc, value_type(-(Conj ? conj_t::apply(lji) : lji)), value_type(x(j)));
        }
        x(i) = AT::template value<value_type>(acc) / L(i, i);
    }
}

} // namespace detail

/// Cholesky factorization: A = L * L^T, for a REAL symmetric positive definite A.
/// The lower triangle of A is overwritten with L. Upper triangle is untouched.
/// Returns 0 on success, k+1 if the k-th pivot is not positive (not SPD), which
/// includes a NaN pivot.
///
/// Each entry is formed as `a - (sum of products)`: the products are summed from
/// zero and subtracted once (tag::subtract_sum). Dispatches to LAPACK potrf when
/// MTL5_HAS_LAPACK is defined and the matrix qualifies.
///
/// Complex element types are rejected at compile time; use cholesky_h_factor.
///
/// A non-void `Accumulator` forms each entry as one reduction (see the top of
/// this file) and never dispatches to LAPACK.
template <typename Accumulator = void, FieldMatrix M>
int cholesky_factor(M& A) {
    assert(A.num_cols() == A.num_rows());

    if constexpr (!interface::accumulator_allows_blas_v<Accumulator>) {
        static_assert(!is_complex_v<typename M::value_type>,
            "cholesky_factor computes A = L*L^T for a REAL symmetric positive definite "
            "matrix; for a HERMITIAN one use cholesky_h_factor (#353).");
        return detail::cholesky_factor_acc<Accumulator>(A);
    } else {

#ifdef MTL5_HAS_LAPACK
    if constexpr (interface::BlasDenseMatrix<M>) {
        using value_type = typename M::value_type;
        using size_type  = typename M::size_type;
        const size_type n = A.num_rows();
        // potrf('L') factors the lower triangle in place into L (A = L*L^T). A
        // column-major dense2D is a valid column-major buffer; a row-major one is
        // factored via a column-major scratch built through the (i,j) accessor
        // and copied back through (i,j), so cholesky_solve reads L(i,j) exactly
        // as on the column-major path. For a symmetric A the copied-in lower
        // triangle is the same matrix either way (#417).
        // n == 0 would make lda == 0, which LAPACK rejects (XERBLA); an empty
        // matrix is trivially factored, so skip the call and fall through.
        if (n > 0) {
            int n_int = static_cast<int>(n);
            int info;
            if constexpr (interface::is_row_major_v<M>) {
                std::vector<value_type> buf(static_cast<std::size_t>(n) * static_cast<std::size_t>(n));
                for (size_type i = 0; i < n; ++i)
                    for (size_type j = 0; j < n; ++j)
                        buf[static_cast<std::size_t>(j) * n + i] = A(i, j);
                info = interface::lapack::potrf('L', n_int, buf.data(), n_int);
                for (size_type i = 0; i < n; ++i)
                    for (size_type j = 0; j < n; ++j)
                        A(i, j) = buf[static_cast<std::size_t>(j) * n + i];
            } else {
                info = interface::lapack::potrf('L', n_int,
                           const_cast<value_type*>(A.data()), n_int);
            }
            return (info > 0) ? info : 0;
        }
    }
#endif

    return detail::cholesky_factor_generic<tag::subtract_sum>(A);
    }
}

/// cholesky_factor(A), with the default summation order named explicitly.
template <FieldMatrix M>
int cholesky_factor(M& A, tag::subtract_sum) {
    return cholesky_factor(A);
}

/// Cholesky factorization A = L * L^T with each entry formed TERM BY TERM: start
/// from the matrix entry and subtract each product in turn (#556),
///
///   s = A(i,j);  s = s - L(i,k) * L(j,k)  for k = 0 .. j-1;  L(i,j) = s / L(j,j)
///
/// which is the form many hand-written factorizations use. It rounds differently
/// from the default, so a caller whose results are pinned to that form can
/// reproduce them bit for bit (under the same FP contraction setting). This
/// overload never dispatches to LAPACK, whose summation order is its own.
/// Same return codes as cholesky_factor(A).
template <FieldMatrix M>
int cholesky_factor(M& A, tag::subtract_term_by_term) {
    assert(A.num_cols() == A.num_rows());
    return detail::cholesky_factor_generic<tag::subtract_term_by_term>(A);
}

/// Solve A*x = b using precomputed Cholesky factor L (stored in lower triangle of A).
/// Solves L*y = b (forward), then L^T*x = y (backward). `Accumulator` forms
/// each row's sum of products.
template <typename Accumulator = void, FieldMatrix M, Vector VecX, Vector VecB>
void cholesky_solve(const M& L, VecX& x, const VecB& b) {
    using value_type = typename VecX::value_type;
    using size_type  = typename M::size_type;
    const size_type n = L.num_rows();
    assert(L.num_cols() == n && x.size() == n && b.size() == n);

    if constexpr (!std::is_void_v<Accumulator>) {
        detail::cholesky_solve_acc<Accumulator, false>(L, x, b);
        return;
    }

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
///
/// A non-void `Accumulator` forms each entry as one reduction and never
/// dispatches to LAPACK; for a real element type it forwards to
/// cholesky_factor<Accumulator>, so the two stay bit-identical.
template <typename Accumulator = void, FieldMatrix M>
int cholesky_h_factor(M& A) {
    using value_type = typename M::value_type;
    using size_type  = typename M::size_type;
    using mag_t      = magnitude_t<value_type>;
    using conj_t     = functor::scalar::conj<value_type>;
    using real_t     = functor::scalar::real<value_type>;
    using std::sqrt;
    const size_type n = A.num_rows();
    assert(A.num_cols() == n);

#ifdef MTL5_HAS_LAPACK
    // For a REAL element type A = L*L^H is exactly A = L*L^T, so reuse
    // cholesky_factor. This keeps the two entry points BIT-IDENTICAL -- the
    // invariant test_cholesky.cpp pins with == -- now that cholesky_factor
    // dispatches to LAPACK potrf (#417); factoring here independently would
    // diverge from it in the low bits. It also gives cholesky_h_factor the same
    // dispatch (row- and column-major). Complex Hermitian types are not
    // BlasDenseMatrix (float/double only), so they skip this and take the
    // generic L*L^H below -- the only correct path for them.
    if constexpr (interface::BlasDenseMatrix<M> &&
                  interface::accumulator_allows_blas_v<Accumulator>) {
        return cholesky_factor(A);
    }
#endif
    if constexpr (!std::is_void_v<Accumulator> && !is_complex_v<value_type>)
        return cholesky_factor<Accumulator>(A);

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

    if constexpr (!std::is_void_v<Accumulator>) {
        return detail::cholesky_factor_acc<Accumulator>(A);
    } else {
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
            ::mtl::parallel::pool().parallel_for(
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
}

/// Solve A*x = b using precomputed Cholesky factors of a HERMITIAN A, i.e. the
/// L of A = L*L^H stored in the lower triangle by cholesky_h_factor.
/// Solves L*y = b (forward), then L^H*x = y (backward). `Accumulator` forms
/// each row's sum of products.
template <typename Accumulator = void, FieldMatrix M, Vector VecX, Vector VecB>
void cholesky_h_solve(const M& L, VecX& x, const VecB& b) {
    using value_type = typename VecX::value_type;
    using size_type  = typename M::size_type;
    using conj_t     = functor::scalar::conj<value_type>;
    const size_type n = L.num_rows();
    assert(L.num_cols() == n && x.size() == n && b.size() == n);

    if constexpr (!std::is_void_v<Accumulator>) {
        detail::cholesky_solve_acc<Accumulator, true>(L, x, b);
        return;
    }

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
