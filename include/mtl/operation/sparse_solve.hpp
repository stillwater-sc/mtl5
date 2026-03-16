#pragma once
// MTL5 -- Unified sparse/dense solve dispatch
//
// Provides top-level solve() free functions that automatically select the
// best available backend based on matrix type and library availability:
//
//   Dense float/double + LAPACK  → LAPACK (existing in lu.hpp/cholesky.hpp/qr.hpp)
//   Sparse float/double + UMFPACK → UMFPACK (for LU)
//   Sparse float/double + CHOLMOD → CHOLMOD (for Cholesky)
//   Sparse float/double + SPQR   → SPQR (for QR)
//   Sparse (any type)             → native sparse solvers
//   Dense (any type)              → dense generic solvers
//
// The dispatch is entirely at compile time via if constexpr + traits.
//
// Usage:
//   mtl::solve(A, x, b);                    // auto-dispatch
//   mtl::cholesky_solve_dispatch(A, x, b);  // Cholesky-specific dispatch
//   mtl::lu_solve_dispatch(A, x, b);        // LU-specific dispatch
//   mtl::qr_solve_dispatch(A, x, b);        // QR-specific dispatch

#include <cstddef>
#include <vector>

#include <mtl/concepts/matrix.hpp>
#include <mtl/concepts/vector.hpp>
#include <mtl/traits/category.hpp>
#include <mtl/tag/sparsity.hpp>
#include <mtl/interface/dispatch_traits.hpp>

// Dense solvers (existing)
#include <mtl/operation/lu.hpp>
#include <mtl/operation/cholesky.hpp>
#include <mtl/operation/qr.hpp>

// Native sparse solvers
#include <mtl/sparse/factorization/sparse_cholesky.hpp>
#include <mtl/sparse/factorization/sparse_lu.hpp>
#include <mtl/sparse/factorization/sparse_qr.hpp>
#include <mtl/sparse/ordering/amd.hpp>
#include <mtl/sparse/ordering/colamd.hpp>

// External sparse solvers (conditional)
#ifdef MTL5_HAS_UMFPACK
#include <mtl/interface/umfpack.hpp>
#endif
#ifdef MTL5_HAS_CHOLMOD
#include <mtl/interface/cholmod.hpp>
#endif
#ifdef MTL5_HAS_SPQR
#include <mtl/interface/spqr.hpp>
#endif

namespace mtl {

/// Cholesky-based solve with automatic dispatch.
/// For sparse float/double with CHOLMOD: uses CHOLMOD.
/// For sparse (any type): uses native sparse Cholesky with AMD ordering.
/// For dense: uses in-place dense Cholesky.
template <Matrix M, Vector VecX, Vector VecB>
void cholesky_solve_dispatch(const M& A, VecX& x, const VecB& b) {
    using category = traits::category_t<M>;

    if constexpr (std::is_same_v<category, tag::sparse>) {
#ifdef MTL5_HAS_CHOLMOD
        if constexpr (interface::is_suitesparse_eligible_v<M>) {
            interface::cholmod_solve(A, x, b);
            return;
        }
#endif
        // Native sparse Cholesky with AMD ordering
        sparse::factorization::sparse_cholesky_solve(A, x, b,
            sparse::ordering::amd{});
    } else {
        // Dense path: factor a copy, then solve
        M L = A;
        int info = cholesky_factor(L);
        if (info != 0) {
            throw std::runtime_error(
                "cholesky_solve_dispatch: matrix is not SPD (failed at column "
                + std::to_string(info - 1) + ")");
        }
        cholesky_solve(L, x, b);
    }
}

/// LU-based solve with automatic dispatch.
/// For sparse float/double with UMFPACK: uses UMFPACK.
/// For sparse (any type): uses native sparse LU.
/// For dense: uses in-place dense LU with partial pivoting.
template <Matrix M, Vector VecX, Vector VecB>
void lu_solve_dispatch(const M& A, VecX& x, const VecB& b) {
    using category = traits::category_t<M>;

    if constexpr (std::is_same_v<category, tag::sparse>) {
#ifdef MTL5_HAS_UMFPACK
        if constexpr (interface::is_suitesparse_eligible_v<M>) {
            interface::umfpack_solve(A, x, b);
            return;
        }
#endif
        // Native sparse LU with COLAMD ordering
        sparse::factorization::sparse_lu_solve(A, x, b,
            sparse::ordering::colamd{});
    } else {
        // Dense path: factor a copy, then solve
        M LU = A;
        std::vector<typename M::size_type> pivot;
        int info = lu_factor(LU, pivot);
        if (info != 0) {
            throw std::runtime_error(
                "lu_solve_dispatch: singular matrix (zero pivot at column "
                + std::to_string(info - 1) + ")");
        }
        lu_solve(LU, pivot, x, b);
    }
}

/// QR-based solve with automatic dispatch.
/// For sparse float/double with SPQR: uses SPQR.
/// For sparse (any type): uses native sparse QR.
/// For dense: uses in-place dense QR.
template <Matrix M, Vector VecX, Vector VecB>
void qr_solve_dispatch(const M& A, VecX& x, const VecB& b) {
    using category = traits::category_t<M>;

    if constexpr (std::is_same_v<category, tag::sparse>) {
#ifdef MTL5_HAS_SPQR
        if constexpr (interface::is_suitesparse_eligible_v<M>) {
            interface::spqr_solve(A, x, b);
            return;
        }
#endif
        // Native sparse QR
        sparse::factorization::sparse_qr_solve(A, x, b);
    } else {
        // Dense path: factor a copy, then solve
        using value_type = typename M::value_type;
        M QR = A;
        vec::dense_vector<value_type> tau;
        qr_factor(QR, tau);
        qr_solve(QR, tau, x, b);
    }
}

/// General-purpose solve: A*x = b.
/// Dispatches to LU (the most general factorization).
/// For SPD systems, prefer cholesky_solve_dispatch for 2x performance.
template <Matrix M, Vector VecX, Vector VecB>
void solve(const M& A, VecX& x, const VecB& b) {
    lu_solve_dispatch(A, x, b);
}

} // namespace mtl
