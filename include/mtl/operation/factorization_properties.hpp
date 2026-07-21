#pragma once
// MTL5 -- Factorization-backed matrix property queries (#244, batch 2):
// is_spd / is_positive_definite, is_singular / is_nonsingular / is_invertible,
// determinant.
//
// These wrap the existing dense factorizations (cholesky_factor, lu_factor).
// They are O(n^3): each runs a factorization on a COPY of A (the factor
// routines are destructive), so the caller's matrix is left unchanged. Intended
// for dense, square matrices with a value type usable by those factorizations.
#include <cmath>
#include <vector>

#include <mtl/concepts/matrix.hpp>
#include <mtl/concepts/magnitude.hpp>
#include <mtl/math/identity.hpp>
#include <mtl/operation/cholesky.hpp>
#include <mtl/operation/lu.hpp>
#include <mtl/operation/matrix_properties.hpp>   // is_symmetric

namespace mtl {

/// Symmetric positive definite: A is symmetric (within sym_tol, default exact)
/// and its Cholesky factorization exists (every pivot strictly positive).
/// Non-square matrices are never SPD. Runs Cholesky on a copy; O(n^3).
template <Matrix M>
bool is_spd(const M& A, magnitude_t<typename M::value_type> sym_tol = 0) {
    if (A.num_rows() != A.num_cols()) return false;
    if (!is_symmetric(A, sym_tol)) return false;
    auto work = A;                      // Cholesky is destructive
    return cholesky_factor(work) == 0;  // 0 => SPD, k+1 => A(k,k) <= 0
}

/// Alias for is_spd.
template <Matrix M>
bool is_positive_definite(const M& A, magnitude_t<typename M::value_type> sym_tol = 0) {
    return is_spd(A, sym_tol);
}

/// Singular within tol: the LU factorization has a zero pivot, or the smallest
/// |U(k,k)| is <= tol (default 0, i.e. an exact zero pivot). Requires a square
/// matrix. Runs LU on a copy; O(n^3).
template <Matrix M>
bool is_singular(const M& A, magnitude_t<typename M::value_type> tol = 0) {
    using value_type = typename M::value_type;
    using size_type  = typename M::size_type;
    using std::abs;
    const size_type n = A.num_rows();
    if (A.num_cols() != n) return true;   // non-square: no inverse
    if (n == 0) return false;             // empty matrix is (vacuously) nonsingular
    auto work = A;                        // LU is destructive
    std::vector<size_type> pivot;
    if (lu_factor(work, pivot) != 0) return true;  // exact zero pivot
    // No exact zero pivot: flag near-singularity via the smallest |U(k,k)|.
    auto min_diag = abs(work(0, 0));
    for (size_type k = 1; k < n; ++k) {
        auto d = abs(work(k, k));
        if (d < min_diag) min_diag = d;
    }
    return !(min_diag > tol);
}

/// Complement of is_singular.
template <Matrix M>
bool is_nonsingular(const M& A, magnitude_t<typename M::value_type> tol = 0) {
    return !is_singular(A, tol);
}

/// Alias for is_nonsingular.
template <Matrix M>
bool is_invertible(const M& A, magnitude_t<typename M::value_type> tol = 0) {
    return is_nonsingular(A, tol);
}

/// Determinant via LU: sign(permutation) * product of U(k,k). Returns exactly
/// zero when LU reports a zero pivot (singular). The empty (0x0) determinant is
/// one by convention. Requires a square matrix. Runs LU on a copy; O(n^3).
template <Matrix M>
typename M::value_type determinant(const M& A) {
    using value_type = typename M::value_type;
    using size_type  = typename M::size_type;
    const size_type n = A.num_rows();
    if (A.num_cols() != n)
        return math::zero<value_type>();   // non-square: determinant undefined -> 0
    auto work = A;                          // LU is destructive
    std::vector<size_type> pivot;
    if (lu_factor(work, pivot) != 0)
        return math::zero<value_type>();    // zero pivot => singular
    value_type det = math::one<value_type>();
    for (size_type k = 0; k < n; ++k) {
        det = det * work(k, k);
        if (pivot[k] != k) det = -det;      // each actual row swap flips the sign
    }
    return det;
}

} // namespace mtl
