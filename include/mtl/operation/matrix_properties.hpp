#pragma once
// MTL5 -- Matrix structural property predicates (#244, batch 1).
// is_square, is_empty, is_symmetric, is_hermitian, is_upper/lower/is_triangular,
// is_diagonal, is_banded, is_diagonally_dominant.
//
// `tol` is an ABSOLUTE threshold on the relevant deviation; the default 0
// requires exact structure (as constructed). Pass tol > 0 for an approximate
// check on a computed matrix. Works for any Matrix that supports A(i,j) (dense
// or sparse) and any element type with abs() (real, complex, custom).
//
// The deviation tests are written as !(dev <= tol) rather than dev > tol so a
// NaN entry (dev is NaN, all comparisons unordered) fails the predicate instead
// of being silently accepted.
#include <cmath>
#include <cstddef>

#include <mtl/concepts/matrix.hpp>
#include <mtl/concepts/magnitude.hpp>
#include <mtl/functor/scalar/conj.hpp>

namespace mtl {

/// A.num_rows() == A.num_cols().
template <Matrix M>
bool is_square(const M& A) { return A.num_rows() == A.num_cols(); }

/// A has no rows or no columns.
template <Matrix M>
bool is_empty(const M& A) { return A.num_rows() == 0 || A.num_cols() == 0; }

/// A == A^T (within tol). Non-square matrices are never symmetric.
template <Matrix M>
bool is_symmetric(const M& A, magnitude_t<typename M::value_type> tol = 0) {
    using std::abs;
    using size_type = typename M::size_type;
    if (A.num_rows() != A.num_cols()) return false;
    const size_type n = A.num_rows();
    for (size_type i = 0; i < n; ++i)
        for (size_type j = i + 1; j < n; ++j)
            if (!(abs(A(i, j) - A(j, i)) <= tol)) return false;
    return true;
}

/// A == A^H (within tol): A(i,j) == conj(A(j,i)), and the diagonal is real.
/// For real element types this coincides with is_symmetric.
template <Matrix M>
bool is_hermitian(const M& A, magnitude_t<typename M::value_type> tol = 0) {
    using std::abs;
    using T = typename M::value_type;
    using size_type = typename M::size_type;
    if (A.num_rows() != A.num_cols()) return false;
    const size_type n = A.num_rows();
    for (size_type i = 0; i < n; ++i)
        for (size_type j = i; j < n; ++j)   // include the diagonal: Im(A(i,i)) ~ 0
            if (!(abs(A(i, j) - functor::scalar::conj<T>::apply(A(j, i))) <= tol)) return false;
    return true;
}

/// Upper triangular: everything strictly below the main diagonal is ~0.
template <Matrix M>
bool is_upper_triangular(const M& A, magnitude_t<typename M::value_type> tol = 0) {
    using std::abs;
    using size_type = typename M::size_type;
    const size_type m = A.num_rows(), n = A.num_cols();
    for (size_type i = 0; i < m; ++i)
        for (size_type j = 0; j < i && j < n; ++j)
            if (!(abs(A(i, j)) <= tol)) return false;
    return true;
}

/// Lower triangular: everything strictly above the main diagonal is ~0.
template <Matrix M>
bool is_lower_triangular(const M& A, magnitude_t<typename M::value_type> tol = 0) {
    using std::abs;
    using size_type = typename M::size_type;
    const size_type m = A.num_rows(), n = A.num_cols();
    for (size_type i = 0; i < m; ++i)
        for (size_type j = i + 1; j < n; ++j)
            if (!(abs(A(i, j)) <= tol)) return false;
    return true;
}

/// Upper or lower triangular.
template <Matrix M>
bool is_triangular(const M& A, magnitude_t<typename M::value_type> tol = 0) {
    return is_upper_triangular(A, tol) || is_lower_triangular(A, tol);
}

/// Diagonal: every off-diagonal entry is ~0.
template <Matrix M>
bool is_diagonal(const M& A, magnitude_t<typename M::value_type> tol = 0) {
    using std::abs;
    using size_type = typename M::size_type;
    const size_type m = A.num_rows(), n = A.num_cols();
    for (size_type i = 0; i < m; ++i)
        for (size_type j = 0; j < n; ++j)
            if (i != j && !(abs(A(i, j)) <= tol)) return false;
    return true;
}

/// Banded with lower bandwidth kl and upper bandwidth ku: an entry more than kl
/// below (i - j > kl) or more than ku above (j - i > ku) the diagonal must be ~0.
template <Matrix M>
bool is_banded(const M& A, std::size_t kl, std::size_t ku,
               magnitude_t<typename M::value_type> tol = 0) {
    using std::abs;
    using size_type = typename M::size_type;
    const size_type m = A.num_rows(), n = A.num_cols();
    for (size_type i = 0; i < m; ++i)
        for (size_type j = 0; j < n; ++j) {
            const bool below = (i > j) && (static_cast<std::size_t>(i - j) > kl);
            const bool above = (j > i) && (static_cast<std::size_t>(j - i) > ku);
            if ((below || above) && !(abs(A(i, j)) <= tol)) return false;
        }
    return true;
}

/// Row diagonally dominant: |A(i,i)| >= sum_{j != i} |A(i,j)| for every row
/// (strict: strictly greater). Non-square matrices are never dominant.
template <Matrix M>
bool is_diagonally_dominant(const M& A, bool strict = false) {
    using std::abs;
    using mag_t = magnitude_t<typename M::value_type>;
    using size_type = typename M::size_type;
    if (A.num_rows() != A.num_cols()) return false;
    const size_type n = A.num_rows();
    for (size_type i = 0; i < n; ++i) {
        mag_t off = mag_t(0);
        for (size_type j = 0; j < n; ++j)
            if (j != i) off += abs(A(i, j));
        const mag_t d = abs(A(i, i));
        if (strict ? !(d > off) : !(d >= off)) return false;
    }
    return true;
}

} // namespace mtl
