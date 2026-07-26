#pragma once
// MTL5 -- Cholesky factorization for symmetric positive definite matrices
// A = L*L^T. In-place: lower triangle of A is overwritten with L.
// Optional LAPACK dispatch when MTL5_HAS_LAPACK is defined and types qualify.
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <mtl/concepts/matrix.hpp>
#include <mtl/concepts/vector.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/math/identity.hpp>
#include <mtl/interface/dispatch_traits.hpp>
#include <mtl/detail/thread_pool.hpp>
#ifdef MTL5_HAS_LAPACK
#include <mtl/interface/lapack.hpp>
#endif

namespace mtl {

/// Cholesky factorization: A = L * L^T.
/// The lower triangle of A is overwritten with L. Upper triangle is untouched.
/// Returns 0 on success, k+1 if A(k,k) <= 0 (not SPD).
template <Matrix M>
int cholesky_factor(M& A) {
    using value_type = typename M::value_type;
    using size_type  = typename M::size_type;
    using std::sqrt;
    const size_type n = A.num_rows();
    assert(A.num_cols() == n);

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

} // namespace mtl
