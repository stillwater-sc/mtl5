#pragma once
// MTL5 -- LDL^T factorization for symmetric matrices (square-root-free Cholesky)
// A = L*D*L^T where L is unit lower triangular, D is diagonal.
// In-place: lower triangle of A is overwritten with L (unit diagonal implicit),
// diagonal of A is overwritten with D.
//
// Key advantages over LL^T Cholesky:
//   - No square roots — avoids precision loss for ill-conditioned matrices
//   - Works for symmetric indefinite matrices (D can have negative entries)
//   - Same O(n^3/3) cost as Cholesky
//
// Reference: Golub & Van Loan, "Matrix Computations", Section 4.1.2

#include <cassert>
#include <mtl/concepts/matrix.hpp>
#include <mtl/concepts/vector.hpp>
#include <mtl/math/identity.hpp>

namespace mtl {

/// LDL^T factorization: A = L * D * L^T.
/// The strictly lower triangle of A is overwritten with L (unit diagonal implicit).
/// The diagonal of A is overwritten with D.
/// Returns 0 on success, k+1 if D(k,k) == 0 (zero pivot).
template <Matrix M>
int ldlt_factor(M& A) {
    using value_type = typename M::value_type;
    using size_type  = typename M::size_type;
    const size_type n = A.num_rows();
    assert(A.num_cols() == n);

    // Algorithm: column-outer LDL^T (Golub & Van Loan, Algorithm 4.1.2)
    //
    // For j = 0..n-1:
    //   v(k) = L(j,k) * D(k)  for k = 0..j-1
    //   D(j) = A(j,j) - sum_{k<j} L(j,k) * v(k)
    //   L(i,j) = (A(i,j) - sum_{k<j} L(i,k) * v(k)) / D(j)  for i > j

    for (size_type j = 0; j < n; ++j) {
        // Compute D(j) = A(j,j) - sum_{k<j} L(j,k)^2 * D(k)
        auto dj = A(j, j);
        for (size_type k = 0; k < j; ++k) {
            auto ljk = A(j, k);
            dj -= ljk * ljk * A(k, k);  // A(k,k) holds D(k)
        }
        if (dj == math::zero<value_type>())
            return static_cast<int>(j + 1);
        A(j, j) = dj;  // Store D(j) on diagonal

        // Compute L(i,j) for i > j
        for (size_type i = j + 1; i < n; ++i) {
            auto sum = math::zero<value_type>();
            for (size_type k = 0; k < j; ++k)
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
    using size_type  = typename M::size_type;
    const size_type n = A.num_rows();
    assert(A.num_cols() == n && x.size() == n && b.size() == n);

    // Forward substitution: L*y = b (L has unit diagonal)
    for (size_type i = 0; i < n; ++i) {
        auto sum = math::zero<value_type>();
        for (size_type j = 0; j < i; ++j)
            sum += A(i, j) * x(j);
        x(i) = b(i) - sum;
    }

    // Diagonal solve: D*z = y
    for (size_type i = 0; i < n; ++i)
        x(i) /= A(i, i);

    // Back substitution: L^T*x = z (L has unit diagonal)
    for (size_type ii = 0; ii < n; ++ii) {
        size_type i = n - 1 - ii;
        auto sum = math::zero<value_type>();
        for (size_type j = i + 1; j < n; ++j)
            sum += A(j, i) * x(j);  // L^T(i,j) = L(j,i)
        x(i) = x(i) - sum;
    }
}

} // namespace mtl
