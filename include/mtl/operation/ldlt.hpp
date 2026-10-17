#pragma once
// MTL5 -- LDL^T factorization for symmetric matrices (square-root-free Cholesky)
// A = L*D*L^T where L is unit lower triangular, D is diagonal.
// In-place: lower triangle of A is overwritten with L (unit diagonal implicit),
// diagonal of A is overwritten with D.
//
// Key advantages over LL^T Cholesky:
//   - No square roots - avoids precision loss for ill-conditioned matrices
//   - Works for symmetric indefinite matrices (D can have negative entries)
//   - Same O(n^3/3) cost as Cholesky
//
// LDL^T vs LDL^H (#352). This file provides BOTH, and they are not
// interchangeable for complex elements:
//
//   ldlt_factor / ldlt_solve      A = L*D*L^T   -- correct for a REAL symmetric
//                                 matrix and for a COMPLEX SYMMETRIC one (A == A^T)
//   ldlt_h_factor / ldlt_h_solve  A = L*D*L^H   -- correct for a REAL symmetric
//                                 matrix and for a HERMITIAN one (A == A^H),
//                                 with D real
//
// For real elements conjugation is the identity, so the two agree. For complex
// elements they diverge, and feeding a Hermitian matrix to the LDL^T form used
// to run to completion and return a plausible but wrong answer under info == 0.
// Each routine now rejects the other's input rather than computing the wrong
// factorization:
//
//   ldlt_factor    -> LDLT_NOT_SYMMETRIC  on Hermitian, non-symmetric input
//   ldlt_h_factor  -> LDLT_NOT_HERMITIAN  on input with a non-real diagonal
//
// Both tests are scale-relative, not exact: matrices assembled in floating
// point carry their structure only to rounding, and an exact test lets a
// one-ULP perturbation through into the silently-wrong-answer path.
//
// Reference: Golub & Van Loan, "Matrix Computations", Section 4.1.2
//
// Mixed precision (#585): every routine takes a leading `Accumulator` (default
// void, see math/accumulator_traits.hpp); void is byte-identical to the
// routines before #585. In the factorizations a non-void accumulator forms each
// entry as ONE reduction, A(i,j) - sum_k L(i,k) v(k) with v(k) = conj(L(j,k))
// D(k) rounded to the element type, then rounds once -- the sparse LDL^T's form
// (#531). In the solves it forms each row's b(i) - sum_j L(i,j) x(j) the same
// way, rounded once, as the sparse level-scheduled solves do (#535).

#include <cassert>
#include <cstddef>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include <mtl/concepts/matrix.hpp>
#include <mtl/concepts/vector.hpp>
#include <mtl/concepts/magnitude.hpp>
#include <mtl/math/identity.hpp>
#include <mtl/math/accumulator_traits.hpp>
#include <mtl/functor/scalar/conj.hpp>
#include <mtl/functor/scalar/real.hpp>
#include <mtl/functor/scalar/imag.hpp>
#include <mtl/detail/structure_tol.hpp>

namespace mtl {

/// Returned by ldlt_factor when the input is complex and Hermitian but not
/// symmetric -- i.e. the caller wanted LDL^H and would otherwise have received
/// a silently wrong LDL^T. Negative so it cannot collide with the k+1 zero-pivot
/// codes. Use ldlt_h_factor for that input.
inline constexpr int LDLT_NOT_SYMMETRIC = -1;

/// Returned by ldlt_h_factor when the input is complex and has a non-real
/// diagonal, which a Hermitian matrix cannot have -- i.e. the caller wanted
/// LDL^T and would otherwise have received a silently wrong LDL^H. Use
/// ldlt_factor for that input.
inline constexpr int LDLT_NOT_HERMITIAN = -2;


namespace detail {

/// The accumulator loop of ldlt_factor (Conj = false, A = L D L^T) and
/// ldlt_h_factor (Conj = true, A = L D L^H, D real). Column j first forms
/// v(k) = conj?(L(j,k)) * D(k), once, in the element type; then
///
///   D(j)   = A(j,j) - sum_k L(j,k) v(k)            (real part when Conj)
///   L(i,j) = (A(i,j) - sum_k L(i,k) v(k)) / D(j)
///
/// each one accumulator, assigned the matrix entry and rounded once.
template <typename Accumulator, bool Conj, typename M>
int ldlt_factor_acc(M& A) {
    using value_type = typename M::value_type;
    using conj_t     = functor::scalar::conj<value_type>;
    using AT         = math::accumulator_traits<Accumulator, value_type>;
    const std::size_t n = A.num_rows();
    std::vector<value_type> v(n);

    auto reduce = [&](std::size_t i, std::size_t j) {
        Accumulator acc{};
        AT::assign(acc, A(i, j));
        for (std::size_t k = 0; k < j; ++k)
            AT::add_product(acc, value_type(-A(i, k)), v[k]);
        return AT::template value<value_type>(acc);
    };

    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t k = 0; k < j; ++k) {
            const value_type ljk = A(j, k);
            v[k] = (Conj ? conj_t::apply(ljk) : ljk) * A(k, k);   // A(k,k) holds D(k)
        }
        value_type dj = reduce(j, j);
        if constexpr (Conj)
            dj = value_type(functor::scalar::real<value_type>::apply(dj));
        if (dj == math::zero<value_type>())
            return static_cast<int>(j + 1);
        A(j, j) = dj;
        for (std::size_t i = j + 1; i < n; ++i)
            A(i, j) = reduce(i, j) / dj;
    }
    return 0;
}

/// The three phases of an LDL^T (Conj = false) or LDL^H (true) solve, with each
/// row's b(i) - sum of products formed in Accumulator and rounded once.
template <typename Accumulator, bool Conj, typename M, typename VecX, typename VecB>
void ldlt_solve_acc(const M& A, VecX& x, const VecB& b, const char* who) {
    using value_type = typename VecX::value_type;
    using conj_t     = functor::scalar::conj<value_type>;
    using AT         = math::accumulator_traits<Accumulator, value_type>;
    const std::size_t n = A.num_rows();

    for (std::size_t i = 0; i < n; ++i) {
        Accumulator acc{};
        AT::assign(acc, value_type(b(i)));
        for (std::size_t j = 0; j < i; ++j)
            AT::add_product(acc, value_type(-A(i, j)), value_type(x(j)));
        x(i) = AT::template value<value_type>(acc);
    }
    for (std::size_t i = 0; i < n; ++i) {
        if (A(i, i) == math::zero<value_type>())
            throw std::domain_error(std::string(who) + ": zero diagonal pivot in D");
        x(i) /= A(i, i);
    }
    for (std::size_t ii = 0; ii < n; ++ii) {
        const std::size_t i = n - 1 - ii;
        Accumulator acc{};
        AT::assign(acc, value_type(x(i)));
        for (std::size_t j = i + 1; j < n; ++j) {
            const value_type lji = A(j, i);
            AT::add_product(acc, value_type(-(Conj ? conj_t::apply(lji) : lji)), value_type(x(j)));
        }
        x(i) = AT::template value<value_type>(acc);
    }
}

} // namespace detail

/// LDL^T factorization: A = L * D * L^T.
/// The strictly lower triangle of A is overwritten with L (unit diagonal implicit).
/// The diagonal of A is overwritten with D.
/// Returns 0 on success, k+1 if D(k,k) == 0 (zero pivot).
/// A non-void `Accumulator` forms each entry as one reduction (top of file).
template <typename Accumulator = void, FieldMatrix M>
int ldlt_factor(M& A) {
    using value_type = typename M::value_type;
    const std::size_t n = A.num_rows();
    assert(A.num_cols() == n);

    // Reject complex input that is Hermitian but not symmetric. This routine
    // computes A = L*D*L^T, which is the wrong factorization for a Hermitian
    // matrix; it used to run to completion and return a wrong answer under
    // info == 0 (#352). Callers with Hermitian input want ldlt_h_factor.
    //
    // The test is scale-relative, NOT exact. A matrix assembled in floating
    // point -- from an FEM pass, or from a blocked B^H*B whose (i,j) and (j,i)
    // sums accumulate in different orders -- is Hermitian to rounding but not
    // bit-exactly so. An exact test misses those: one ULP in one entry is
    // enough to slip past it and get the wrong answer under info == 0, which is
    // precisely the bug being fixed. So compare the symmetry and Hermitian
    // residuals against a threshold set by the data.
    //
    // Refuse only "Hermitian within tolerance AND not symmetric within
    // tolerance". Complex SYMMETRIC input -- which this routine handles
    // correctly -- has a zero symmetry residual and is untouched, as is a
    // matrix that is neither (the caller is then relying on the documented
    // lower-triangle convention, and the upper triangle is not ours to judge).
    if constexpr (is_complex_v<value_type>) {
        using mag_t  = magnitude_t<value_type>;
        using conj_t = functor::scalar::conj<value_type>;
        mag_t scale(0), sym_err(0), herm_err(0);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = i; j < n; ++j) {   // each pair once; j == i covers
                                                    // the diagonal, which must be
                                                    // real for a Hermitian A
                const mag_t a = detail::cabs1(A(i, j));
                if (a > scale) scale = a;
                const mag_t s = detail::cabs1(value_type(A(i, j) - A(j, i)));
                if (s > sym_err) sym_err = s;
                const mag_t h = detail::cabs1(
                    value_type(A(i, j) - conj_t::apply(A(j, i))));
                if (h > herm_err) herm_err = h;
            }
        const mag_t tol = detail::structure_tol(n, scale);
        if (herm_err <= tol && sym_err > tol) return LDLT_NOT_SYMMETRIC;
    }

    if constexpr (!std::is_void_v<Accumulator>)
        return detail::ldlt_factor_acc<Accumulator, false>(A);

    // Algorithm: column-outer LDL^T (Golub & Van Loan, Algorithm 4.1.2)
    //
    // For j = 0..n-1:
    //   v(k) = L(j,k) * D(k)  for k = 0..j-1
    //   D(j) = A(j,j) - sum_{k<j} L(j,k) * v(k)
    //   L(i,j) = (A(i,j) - sum_{k<j} L(i,k) * v(k)) / D(j)  for i > j

    for (std::size_t j = 0; j < n; ++j) {
        // Compute D(j) = A(j,j) - sum_{k<j} L(j,k)^2 * D(k)
        auto dj = A(j, j);
        for (std::size_t k = 0; k < j; ++k) {
            auto ljk = A(j, k);
            dj -= ljk * ljk * A(k, k);  // A(k,k) holds D(k)
        }
        if (dj == math::zero<value_type>())
            return static_cast<int>(j + 1);
        A(j, j) = dj;  // Store D(j) on diagonal

        // Compute L(i,j) for i > j
        for (std::size_t i = j + 1; i < n; ++i) {
            auto sum = math::zero<value_type>();
            for (std::size_t k = 0; k < j; ++k)
                sum += A(i, k) * A(j, k) * A(k, k);  // L(i,k) * L(j,k) * D(k)
            A(i, j) = (A(i, j) - sum) / dj;
        }
    }
    return 0;
}

/// Solve A*x = b using precomputed LDL^T factors stored in A.
/// Lower triangle of A contains L (unit diagonal implicit), diagonal contains D.
/// Three phases: L*y = b (forward), D*z = y (diagonal), L^T*x = z (backward).
/// `Accumulator` forms each row's sum of products.
template <typename Accumulator = void, FieldMatrix M, Vector VecX, Vector VecB>
void ldlt_solve(const M& A, VecX& x, const VecB& b) {
    using value_type = typename VecX::value_type;
    const std::size_t n = A.num_rows();
    assert(A.num_cols() == n && x.size() == n && b.size() == n);

    if constexpr (!std::is_void_v<Accumulator>) {
        detail::ldlt_solve_acc<Accumulator, false>(A, x, b, "ldlt_solve");
        return;
    }

    // Forward substitution: L*y = b (L has unit diagonal)
    for (std::size_t i = 0; i < n; ++i) {
        auto sum = math::zero<value_type>();
        for (std::size_t j = 0; j < i; ++j)
            sum += A(i, j) * x(j);
        x(i) = b(i) - sum;
    }

    // Diagonal solve: D*z = y
    for (std::size_t i = 0; i < n; ++i) {
        if (A(i, i) == math::zero<value_type>())
            throw std::domain_error("ldlt_solve: zero diagonal pivot in D");
        x(i) /= A(i, i);
    }

    // Back substitution: L^T*x = z (L has unit diagonal)
    for (std::size_t ii = 0; ii < n; ++ii) {
        std::size_t i = n - 1 - ii;
        auto sum = math::zero<value_type>();
        for (std::size_t j = i + 1; j < n; ++j)
            sum += A(j, i) * x(j);  // L^T(i,j) = L(j,i)
        x(i) = x(i) - sum;
    }
}


/// LDL^H factorization: A = L * D * L^H, for a HERMITIAN matrix.
///
/// The strictly lower triangle of A is overwritten with L (unit diagonal
/// implicit); the diagonal is overwritten with D, which is REAL for a Hermitian
/// A and is stored in the element type with a zero imaginary part.
///
/// Only the lower triangle of A is read, so the caller's upper triangle is
/// irrelevant and need not be populated consistently.
///
/// Returns 0 on success, k+1 if D(k) is zero (zero pivot), or
/// LDLT_NOT_HERMITIAN if the diagonal is not real (see below) -- a Hermitian
/// matrix cannot have one, so that input belongs to ldlt_factor.
///
/// For real element types conjugation is the identity and this is exactly
/// ldlt_factor. For complex elements it is the factorization a Hermitian matrix
/// actually has -- see the note at the top of this file (#352).
/// A non-void `Accumulator` forms each entry as one reduction (top of file).
template <typename Accumulator = void, FieldMatrix M>
int ldlt_h_factor(M& A) {
    using value_type = typename M::value_type;
    using mag_t      = magnitude_t<value_type>;
    using conj_t     = functor::scalar::conj<value_type>;
    const std::size_t n = A.num_rows();
    assert(A.num_cols() == n);

    // A Hermitian matrix has a REAL diagonal. If the caller passed a complex-
    // symmetric matrix here by mistake, the Re() below would silently discard
    // its imaginary diagonal and return a wrong factorization under info == 0 --
    // the exact mirror of the ldlt_factor bug this file fixes (#352).
    //
    // This routine reads only the lower triangle, so the full Hermitian check
    // done in ldlt_factor is not available without reading the upper triangle
    // and breaking that contract. The diagonal alone is O(n), touches nothing
    // outside the documented region, and catches the case that actually loses
    // information here.
    if constexpr (is_complex_v<value_type>) {
        mag_t dscale(0), dimag(0);
        for (std::size_t j = 0; j < n; ++j) {
            const mag_t a = detail::cabs1(A(j, j));
            if (a > dscale) dscale = a;
            using std::abs;
            const mag_t im = abs(functor::scalar::imag<value_type>::apply(A(j, j)));
            if (im > dimag) dimag = im;
        }
        if (dimag > detail::structure_tol(n, dscale)) return LDLT_NOT_HERMITIAN;
    }

    if constexpr (!std::is_void_v<Accumulator>)
        return detail::ldlt_factor_acc<Accumulator, true>(A);

    // For j = 0..n-1:
    //   D(j)   = Re( A(j,j) - sum_{k<j} |L(j,k)|^2 * D(k) )
    //   L(i,j) = ( A(i,j) - sum_{k<j} L(i,k) * conj(L(j,k)) * D(k) ) / D(j)
    //
    // D is real by construction: |L(j,k)|^2 is real and A(j,j) is real for a
    // Hermitian A. Taking the real part explicitly keeps a rounding-sized
    // imaginary residue from accumulating down the diagonal.
    for (std::size_t j = 0; j < n; ++j) {
        value_type djv = A(j, j);
        for (std::size_t k = 0; k < j; ++k) {
            const value_type ljk = A(j, k);
            const value_type dk  = A(k, k);
            djv -= ljk * conj_t::apply(ljk) * dk;
        }
        const mag_t dj_real = functor::scalar::real<value_type>::apply(djv);
        if (dj_real == mag_t(0))
            return static_cast<int>(j + 1);
        const value_type dj = value_type(dj_real);
        A(j, j) = dj;

        for (std::size_t i = j + 1; i < n; ++i) {
            value_type sum = math::zero<value_type>();
            for (std::size_t k = 0; k < j; ++k)
                sum += A(i, k) * conj_t::apply(A(j, k)) * A(k, k);
            A(i, j) = (A(i, j) - sum) / dj;
        }
    }
    return 0;
}

/// Solve A*x = b using precomputed LDL^H factors stored in A.
/// Three phases: L*y = b (forward), D*z = y (diagonal), L^H*x = z (backward).
/// `Accumulator` forms each row's sum of products.
template <typename Accumulator = void, FieldMatrix M, Vector VecX, Vector VecB>
void ldlt_h_solve(const M& A, VecX& x, const VecB& b) {
    using value_type = typename VecX::value_type;
    using conj_t     = functor::scalar::conj<value_type>;
    const std::size_t n = A.num_rows();
    assert(A.num_cols() == n && x.size() == n && b.size() == n);

    if constexpr (!std::is_void_v<Accumulator>) {
        detail::ldlt_solve_acc<Accumulator, true>(A, x, b, "ldlt_h_solve");
        return;
    }

    // Forward: L*y = b (unit diagonal)
    for (std::size_t i = 0; i < n; ++i) {
        value_type sum = math::zero<value_type>();
        for (std::size_t j = 0; j < i; ++j)
            sum += A(i, j) * x(j);
        x(i) = b(i) - sum;
    }

    // Diagonal: D*z = y
    for (std::size_t i = 0; i < n; ++i) {
        if (A(i, i) == math::zero<value_type>())
            throw std::domain_error("ldlt_h_solve: zero diagonal pivot in D");
        x(i) /= A(i, i);
    }

    // Backward: L^H*x = z. The (i,j) entry of L^H is conj(L(j,i)) -- the
    // conjugation is the whole difference from the LDL^T solve.
    for (std::size_t ii = 0; ii < n; ++ii) {
        const std::size_t i = n - 1 - ii;
        value_type sum = math::zero<value_type>();
        for (std::size_t j = i + 1; j < n; ++j)
            sum += conj_t::apply(A(j, i)) * x(j);
        x(i) = x(i) - sum;
    }
}

} // namespace mtl
