#pragma once
// MTL5 -- LU factorization with partial pivoting
// In-place: A is overwritten with L\U (unit lower, upper).
// Pivot vector records row swaps.
// Optional LAPACK dispatch when MTL5_HAS_LAPACK is defined and types qualify.
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>
#include <mtl/concepts/matrix.hpp>
#include <mtl/concepts/vector.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/math/identity.hpp>
#include <mtl/functor/scalar/conj.hpp>
#include <mtl/operation/lower_trisolve.hpp>
#include <mtl/operation/upper_trisolve.hpp>
#include <mtl/interface/dispatch_traits.hpp>
#include <mtl/detail/thread_pool.hpp>
#ifdef MTL5_HAS_LAPACK
#include <mtl/interface/lapack.hpp>
#endif

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

#ifdef MTL5_HAS_LAPACK
    if constexpr (interface::BlasDenseMatrix<M>) {
        // LAPACK getrf expects column-major. For column-major dense2D, dispatch directly.
        // For row-major, factoring A_row is equivalent to factoring A_col^T = U^T * L^T,
        // which gives an LU of the transpose. We dispatch only for column-major.
        if constexpr (!interface::is_row_major_v<M>) {
            int m_int = static_cast<int>(n);
            std::vector<int> ipiv(n);
            int info = interface::lapack::getrf(m_int, m_int,
                           const_cast<value_type*>(A.data()), m_int, ipiv.data());
            // Convert 1-based Fortran pivots to 0-based size_type pivots
            for (size_type i = 0; i < n; ++i)
                pivot[i] = static_cast<size_type>(ipiv[i] - 1);
            return (info > 0) ? info : 0;
        }
    }
#endif

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

        // Eliminate below diagonal -- parallelize the trailing-submatrix update
        // over rows i (each row is written by exactly one chunk and reads only
        // the shared pivot row k, so contiguous chunking is bit-identical to the
        // serial path). No-op team at MTL5_NUM_THREADS=1.
        const size_type trailing = n - k - 1;   // rows i in (k, n) and inner cols
        if (trailing > 0) {
            const std::size_t grain =
                std::max<std::size_t>(std::size_t{1},
                                      std::size_t{65536} / static_cast<std::size_t>(trailing));
            detail::thread_pool::instance().parallel_for(
                static_cast<std::size_t>(trailing), grain,
                [&](std::size_t b, std::size_t e) {
                    for (std::size_t t = b; t < e; ++t) {
                        const size_type i = k + 1 + static_cast<size_type>(t);
                        A(i, k) /= A(k, k);  // L multiplier
                        for (size_type j = k + 1; j < n; ++j) {
                            A(i, j) -= A(i, k) * A(k, j);
                        }
                    }
                });
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

/// Solve A^H * x = b using the precomputed LU factorization of A.
///
/// lu_factor produces P*A = L*U, so A = P^-1*L*U and
///
///     A^H = U^H * L^H * P^-H = U^H * L^H * P
///
/// because P is a real permutation, hence P^-H = P. The solve is therefore the
/// mirror image of lu_solve: the triangular factors are applied conjugated, in
/// the opposite order, and the permutation moves to the END and is inverted.
///
/// The permutation in lu_solve is a forward sequence of row interchanges, so
/// its inverse is the same interchanges applied in reverse order.
///
/// Needed by bicg and qmr, the only solvers that ask a preconditioner for
/// M^-H (#394).
template <Matrix M, Vector VecX, Vector VecB>
void lu_adjoint_solve(const M& LU, const std::vector<typename M::size_type>& pivot,
                      VecX& x, const VecB& b) {
    using size_type  = typename M::size_type;
    using value_type = typename M::value_type;
    using conj_t     = functor::scalar::conj<value_type>;
    const size_type n = LU.num_rows();
    assert(LU.num_cols() == n && x.size() == n && b.size() == n);

    for (size_type i = 0; i < n; ++i)
        x(i) = b(i);

    // U^H z = b. U^H is LOWER triangular with diagonal conj(U(i,i)), and
    // (U^H)(i,j) = conj(U(j,i)) = conj(LU(j,i)) for j < i.
    for (size_type i = 0; i < n; ++i) {
        auto sum = math::zero<value_type>();
        for (size_type j = 0; j < i; ++j)
            sum += conj_t::apply(LU(j, i)) * x(j);
        x(i) = (x(i) - sum) / conj_t::apply(LU(i, i));
    }

    // L^H w = z. L^H is UPPER triangular with UNIT diagonal, and
    // (L^H)(i,j) = conj(L(j,i)) = conj(LU(j,i)) for j > i.
    for (size_type ii = 0; ii < n; ++ii) {
        const size_type i = n - 1 - ii;
        auto sum = math::zero<value_type>();
        for (size_type j = i + 1; j < n; ++j)
            sum += conj_t::apply(LU(j, i)) * x(j);
        x(i) = x(i) - sum;
    }

    // x = P^-1 w: the interchanges of lu_solve, in reverse.
    for (size_type ii = 0; ii < n; ++ii) {
        const size_type i = n - 1 - ii;
        if (pivot[i] != i) {
            auto tmp = x(i);
            x(i) = x(pivot[i]);
            x(pivot[i]) = tmp;
        }
    }
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
