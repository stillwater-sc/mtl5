#pragma once
// MTL5 — LU factorization with partial pivoting
// In-place: A is overwritten with L\U (unit lower, upper).
// Pivot vector records row swaps.
#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>
#include <mtl/concepts/matrix.hpp>
#include <mtl/concepts/vector.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/math/identity.hpp>
#include <mtl/operation/lower_trisolve.hpp>
#include <mtl/operation/upper_trisolve.hpp>

namespace mtl {

/// LU factorization with partial pivoting.
/// A is overwritten: L (unit lower, stored below diagonal) + U (stored on+above diagonal).
/// pivot[k] = row index swapped with row k at step k.
/// Returns 0 on success, k+1 if U(k,k) is zero (singular).
template <Matrix M>
int lu_factor(M& A, std::vector<typename M::size_type>& pivot) {
    using value_type = typename M::value_type;
    using size_type  = typename M::size_type;
    const size_type n = A.num_rows();
    assert(A.num_cols() == n);
    pivot.resize(n);

    for (size_type k = 0; k < n; ++k) {
        // Find pivot: row with max |A(i,k)| for i >= k
        size_type max_row = k;
        using std::abs;
        auto max_val = abs(A(k, k));
        for (size_type i = k + 1; i < n; ++i) {
            auto v = abs(A(i, k));
            if (v > max_val) {
                max_val = v;
                max_row = i;
            }
        }
        pivot[k] = max_row;

        // Swap rows k and max_row
        if (max_row != k) {
            for (size_type j = 0; j < n; ++j) {
                auto tmp = A(k, j);
                A(k, j) = A(max_row, j);
                A(max_row, j) = tmp;
            }
        }

        // Check for singularity
        if (A(k, k) == math::zero<value_type>())
            return static_cast<int>(k + 1);

        // Eliminate below diagonal
        for (size_type i = k + 1; i < n; ++i) {
            A(i, k) /= A(k, k);  // L multiplier
            for (size_type j = k + 1; j < n; ++j) {
                A(i, j) -= A(i, k) * A(k, j);
            }
        }
    }
    return 0;
}

/// Solve A*x = b using precomputed LU factorization.
/// LU contains both L (below diagonal, unit) and U (on+above diagonal).
/// Applies pivot permutation, then forward/back substitution.
template <Matrix M, Vector VecX, Vector VecB>
void lu_solve(const M& LU, const std::vector<typename M::size_type>& pivot,
              VecX& x, const VecB& b) {
    using size_type = typename M::size_type;
    const size_type n = LU.num_rows();
    assert(LU.num_cols() == n && x.size() == n && b.size() == n);

    // Copy b into x, applying pivot permutation
    for (size_type i = 0; i < n; ++i)
        x(i) = b(i);
    for (size_type i = 0; i < n; ++i) {
        if (pivot[i] != i) {
            auto tmp = x(i);
            x(i) = x(pivot[i]);
            x(pivot[i]) = tmp;
        }
    }

    // Forward substitution: L*y = Pb (unit diagonal)
    lower_trisolve(LU, x, /*unit_diag=*/true);

    // Back substitution: U*x = y
    upper_trisolve(LU, x, /*unit_diag=*/false);
}

/// Convenience: factor and solve A*x = b in one call.
/// A is modified in place. Returns 0 on success.
template <Matrix M, Vector VecX, Vector VecB>
int lu_apply(M& A, VecX& x, const VecB& b) {
    std::vector<typename M::size_type> pivot;
    int info = lu_factor(A, pivot);
    if (info != 0) return info;
    lu_solve(A, pivot, x, b);
    return 0;
}

} // namespace mtl
