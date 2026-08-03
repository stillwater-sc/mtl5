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
// ldlt_factor now rejects that input rather than computing the wrong
// factorization; see LDLT_NOT_SYMMETRIC.
//
// Reference: Golub & Van Loan, "Matrix Computations", Section 4.1.2

#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <mtl/concepts/matrix.hpp>
#include <mtl/concepts/vector.hpp>
#include <mtl/concepts/magnitude.hpp>
#include <mtl/math/identity.hpp>
#include <mtl/functor/scalar/conj.hpp>
#include <mtl/functor/scalar/real.hpp>

namespace mtl {

/// Returned by ldlt_factor when the input is complex and Hermitian but not
/// symmetric -- i.e. the caller wanted LDL^H and would otherwise have received
/// a silently wrong LDL^T. Negative so it cannot collide with the k+1 zero-pivot
/// codes. Use ldlt_h_factor for that input.
inline constexpr int LDLT_NOT_SYMMETRIC = -1;

/// LDL^T factorization: A = L * D * L^T.
/// The strictly lower triangle of A is overwritten with L (unit diagonal implicit).
/// The diagonal of A is overwritten with D.
/// Returns 0 on success, k+1 if D(k,k) == 0 (zero pivot).
template <Matrix M>
int ldlt_factor(M& A) {
    using value_type = typename M::value_type;
    const std::size_t n = A.num_rows();
    assert(A.num_cols() == n);

    // Reject complex input that is Hermitian but not symmetric. This routine
    // computes A = L*D*L^T, which is the wrong factorization for a Hermitian
    // matrix; it used to run to completion and return a wrong answer under
    // info == 0 (#352). Callers with Hermitian input want ldlt_h_factor.
    //
    // The test is exact (no tolerance): A(i,j) == conj(A(j,i)) for some pair
    // while A(i,j) != A(j,i). Only a matrix that is genuinely Hermitian and
    // genuinely not symmetric is refused, so complex SYMMETRIC input -- which
    // this routine handles correctly -- is untouched.
    if constexpr (is_complex_v<value_type>) {
        bool asym = false, herm = true;
        for (std::size_t i = 0; i < n && herm; ++i)
            for (std::size_t j = 0; j < n; ++j) {
                if (A(i, j) != A(j, i)) asym = true;
                if (A(i, j) != functor::scalar::conj<value_type>::apply(A(j, i))) {
                    herm = false;
                    break;
                }
            }
        if (asym && herm) return LDLT_NOT_SYMMETRIC;
    }

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
template <Matrix M, Vector VecX, Vector VecB>
void ldlt_solve(const M& A, VecX& x, const VecB& b) {
    using value_type = typename VecX::value_type;
    const std::size_t n = A.num_rows();
    assert(A.num_cols() == n && x.size() == n && b.size() == n);

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
/// Returns 0 on success, k+1 if D(k) is zero (zero pivot).
///
/// For real element types conjugation is the identity and this is exactly
/// ldlt_factor. For complex elements it is the factorization a Hermitian matrix
/// actually has -- see the note at the top of this file (#352).
template <Matrix M>
int ldlt_h_factor(M& A) {
    using value_type = typename M::value_type;
    using mag_t      = magnitude_t<value_type>;
    using conj_t     = functor::scalar::conj<value_type>;
    const std::size_t n = A.num_rows();
    assert(A.num_cols() == n);

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
template <Matrix M, Vector VecX, Vector VecB>
void ldlt_h_solve(const M& A, VecX& x, const VecB& b) {
    using value_type = typename VecX::value_type;
    using conj_t     = functor::scalar::conj<value_type>;
    const std::size_t n = A.num_rows();
    assert(A.num_cols() == n && x.size() == n && b.size() == n);

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
