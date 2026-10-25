#pragma once
// MTL5 -- LU factorization with partial pivoting
// In-place: A is overwritten with L\U (unit lower, upper).
// Pivot vector records row swaps.
// Optional LAPACK dispatch when MTL5_HAS_LAPACK is defined and types qualify.
//
// Mixed precision (#585): every routine here takes a leading `Accumulator`
// (default void, see math/accumulator_traits.hpp). void is byte-identical to
// the routines before #585. A non-void accumulator never dispatches to LAPACK,
// and lu_factor then runs a LEFT-looking (Doolittle) elimination instead of the
// right-looking one: right-looking updates each entry once per step, so there
// is no sum for an accumulator to hold, while left-looking forms each entry of
// L and U as ONE reduction, A(i,j) - sum_p L(i,p) U(p,j), rounded once. Same
// pivots, same storage, same pivot convention, so lu_solve reads either. The
// solves then form each row's b(i) - sum_j T(i,j) x(j) in the accumulator,
// rounded once before the division, as the sparse level-scheduled solves do
// (#535). (lower_trisolve / upper_trisolve round the sum first; lu_solve does
// not use them on this path.)
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include <mtl/concepts/matrix.hpp>
#include <mtl/concepts/vector.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/math/identity.hpp>
#include <mtl/math/accumulator_traits.hpp>
#include <mtl/functor/scalar/conj.hpp>
#include <mtl/operation/lower_trisolve.hpp>
#include <mtl/operation/upper_trisolve.hpp>
#include <mtl/interface/dispatch_traits.hpp>
#include <mtl/detail/thread_pool.hpp>
#ifdef MTL5_HAS_LAPACK
#include <mtl/interface/lapack.hpp>
#endif

namespace mtl {

namespace detail {

/// lu_factor's accumulator path: left-looking LU with partial pivoting. At step
/// k it forms column k of the Schur complement (rows i >= k), pivots on it,
/// forms row k of U (columns j > k), then scales column k into L. Every entry
/// is one accumulator: assign the matrix entry, add each -L(i,p) * U(p,j),
/// round once -- the form the sparse LU uses (#530).
///
/// The entries of a column, and of a row, are independent, so each loop is
/// split over the pool; the result is bit-identical to serial.
template <typename Accumulator, typename M>
int lu_factor_acc(M& A, std::vector<typename M::size_type>& pivot) {
    using value_type = typename M::value_type;
    using size_type  = typename M::size_type;
    using AT = math::accumulator_traits<Accumulator, value_type>;
    using std::abs;
    const size_type n = A.num_rows();

    for (size_type k = 0; k < n; ++k) {
        const std::size_t grain =
            std::max<std::size_t>(std::size_t{1},
                                  std::size_t{65536} / static_cast<std::size_t>(k == 0 ? 1 : k));
        // Column k, rows i >= k: A(i,k) - sum_{p<k} L(i,p) U(p,k).
        ::mtl::parallel::pool().parallel_for(
            static_cast<std::size_t>(n - k), grain,
            [&](std::size_t b, std::size_t e) {
                for (std::size_t t = b; t < e; ++t) {
                    const size_type i = k + static_cast<size_type>(t);
                    Accumulator acc{};
                    AT::assign(acc, A(i, k));
                    for (size_type p = 0; p < k; ++p)
                        AT::add_product(acc, value_type(-A(i, p)), A(p, k));
                    A(i, k) = AT::template value<value_type>(acc);
                }
            });

        // Pivot: row with max |A(i,k)| for i >= k, as on the void path.
        size_type max_row = k;
        auto max_val = abs(A(k, k));
        for (size_type i = k + 1; i < n; ++i) {
            auto v = abs(A(i, k));
            if (v > max_val) {
                max_val = v;
                max_row = i;
            }
        }
        pivot[k] = max_row;
        // Whole rows: the finished L part, column k, and the untouched rest.
        if (max_row != k) {
            for (size_type j = 0; j < n; ++j) {
                auto tmp = A(k, j);
                A(k, j) = A(max_row, j);
                A(max_row, j) = tmp;
            }
        }
        if (A(k, k) == math::zero<value_type>())
            return static_cast<int>(k + 1);

        // Row k of U, columns j > k: A(k,j) - sum_{p<k} L(k,p) U(p,j).
        ::mtl::parallel::pool().parallel_for(
            static_cast<std::size_t>(n - k - 1), grain,
            [&](std::size_t b, std::size_t e) {
                for (std::size_t t = b; t < e; ++t) {
                    const size_type j = k + 1 + static_cast<size_type>(t);
                    Accumulator acc{};
                    AT::assign(acc, A(k, j));
                    for (size_type p = 0; p < k; ++p)
                        AT::add_product(acc, value_type(-A(k, p)), A(p, j));
                    A(k, j) = AT::template value<value_type>(acc);
                }
            });

        // Column k of L.
        for (size_type i = k + 1; i < n; ++i)
            A(i, k) /= A(k, k);
    }
    return 0;
}

} // namespace detail

/// LU factorization with partial pivoting.
/// A is overwritten: L (unit lower, stored below diagonal) + U (stored on+above diagonal).
/// pivot[k] = row index swapped with row k at step k.
/// Returns 0 on success, k+1 if U(k,k) is zero (singular).
///
/// A non-void `Accumulator` forms each entry of L and U as one reduction
/// (left-looking, see the top of this file) and never dispatches to LAPACK.
template <typename Accumulator = void, FieldMatrix M>
int lu_factor(M& A, std::vector<typename M::size_type>& pivot) {
    using value_type = typename M::value_type;
    using size_type  = typename M::size_type;
    const size_type n = A.num_rows();
    assert(A.num_cols() == n);
    pivot.resize(n);

    if constexpr (!interface::accumulator_allows_blas_v<Accumulator>) {
        return detail::lu_factor_acc<Accumulator>(A, pivot);
    } else {

#ifdef MTL5_HAS_LAPACK
    if constexpr (interface::BlasDenseMatrix<M>) {
        // getrf works in-place on a column-major buffer, overwriting it with L\U
        // and returning a forward interchange sequence in ipiv (the same pivot
        // convention lu_solve expects, once shifted 1->0 based). A column-major
        // dense2D IS that buffer; a row-major one is factored via a column-major
        // scratch built through the (i,j) accessor, then copied back through
        // (i,j) so the accessor-based lu_solve / lu_adjoint_solve read it exactly
        // as they read the column-major path (#417). Direct-dispatch on the raw
        // row-major buffer would factor A^T instead -- hence the copy.
        // n == 0 would make lda == 0, which LAPACK rejects (XERBLA); an empty
        // matrix is trivially factored, so skip the call and fall through.
        if (n > 0) {
            int m_int = static_cast<int>(n);
            std::vector<int> ipiv(n);
            int info;
            if constexpr (interface::is_row_major_v<M>) {
                std::vector<value_type> buf(static_cast<std::size_t>(n) * static_cast<std::size_t>(n));
                for (size_type i = 0; i < n; ++i)
                    for (size_type j = 0; j < n; ++j)
                        buf[static_cast<std::size_t>(j) * n + i] = A(i, j);
                info = interface::lapack::getrf(m_int, m_int, buf.data(), m_int, ipiv.data());
                for (size_type i = 0; i < n; ++i)
                    for (size_type j = 0; j < n; ++j)
                        A(i, j) = buf[static_cast<std::size_t>(j) * n + i];
            } else {
                info = interface::lapack::getrf(m_int, m_int,
                           const_cast<value_type*>(A.data()), m_int, ipiv.data());
            }
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
            ::mtl::parallel::pool().parallel_for(
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
}

/// Solve A*x = b using precomputed LU factorization.
/// LU contains both L (below diagonal, unit) and U (on+above diagonal).
/// Applies pivot permutation, then forward/back substitution.
/// A non-void `Accumulator` forms each row's b(i) - sum in one reduction.
template <typename Accumulator = void, FieldMatrix M, Vector VecX, Vector VecB>
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

    if constexpr (!std::is_void_v<Accumulator>) {
        using value_type = typename M::value_type;
        using AT = math::accumulator_traits<Accumulator, value_type>;
        for (size_type i = 0; i < n; ++i) {            // L y = Pb, unit diagonal
            Accumulator acc{};
            AT::assign(acc, value_type(x(i)));
            for (size_type j = 0; j < i; ++j)
                AT::add_product(acc, value_type(-LU(i, j)), value_type(x(j)));
            x(i) = AT::template value<value_type>(acc);
        }
        for (size_type ii = 0; ii < n; ++ii) {         // U x = y
            const size_type i = n - 1 - ii;
            Accumulator acc{};
            AT::assign(acc, value_type(x(i)));
            for (size_type j = i + 1; j < n; ++j)
                AT::add_product(acc, value_type(-LU(i, j)), value_type(x(j)));
            x(i) = AT::template value<value_type>(acc) / LU(i, i);
        }
        return;
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
///
/// `Accumulator` forms both substitutions' sums, as in lu_solve.
template <typename Accumulator = void, FieldMatrix M, Vector VecX, Vector VecB>
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
    // x(i) - sum_{j0 <= j < j1} conj(LU(j,i)) x(j). void: the loops as before;
    // otherwise one reduction in Accumulator, rounded once.
    auto residual = [&](size_type i, size_type j0, size_type j1) {
        if constexpr (std::is_void_v<Accumulator>) {
            auto sum = math::zero<value_type>();
            for (size_type j = j0; j < j1; ++j)
                sum += conj_t::apply(LU(j, i)) * x(j);
            return x(i) - sum;
        } else {
            using AT = math::accumulator_traits<Accumulator, value_type>;
            Accumulator acc{};
            AT::assign(acc, value_type(x(i)));
            for (size_type j = j0; j < j1; ++j)
                AT::add_product(acc, value_type(-conj_t::apply(LU(j, i))), value_type(x(j)));
            return AT::template value<value_type>(acc);
        }
    };

    for (size_type i = 0; i < n; ++i)
        x(i) = residual(i, 0, i) / conj_t::apply(LU(i, i));

    // L^H w = z. L^H is UPPER triangular with UNIT diagonal, and
    // (L^H)(i,j) = conj(L(j,i)) = conj(LU(j,i)) for j > i.
    for (size_type ii = 0; ii < n; ++ii) {
        const size_type i = n - 1 - ii;
        x(i) = residual(i, i + 1, n);
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
template <typename Accumulator = void, FieldMatrix M, Vector VecX, Vector VecB>
int lu_apply(M& A, VecX& x, const VecB& b) {
    std::vector<typename M::size_type> pivot;
    int info = lu_factor<Accumulator>(A, pivot);
    if (info != 0) return info;
    lu_solve<Accumulator>(A, pivot, x, b);
    return 0;
}

} // namespace mtl
