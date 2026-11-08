#pragma once
// MTL5 -- Optional SuperLU interface for sparse direct solves
// SuperLU uses a supernodal left-looking LU algorithm for general
// unsymmetric sparse systems.
//
// Guarded by MTL5_HAS_SUPERLU (set by CMake when MTL5_ENABLE_SUPERLU=ON)
//
// Reference: Demmel et al., "A Supernodal Approach to Sparse Partial
//            Pivoting", SIAM J. Matrix Anal. Appl., 20(3), 1999.

#ifdef MTL5_HAS_SUPERLU

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/vec/dense_vector.hpp>

// SuperLU's slu_ddefs.h declares its own extern "C" prototypes for the BLAS
// routines SuperLU calls (dgemm_, dgemv_, dtrsv_, dtrsm_ and, from 6.x, dcopy_
// and daxpy_). Their signatures differ between SuperLU versions (int or void
// returns, with or without const) and from MTL5's const-correct ones in
// interface/blas.hpp, and two different declarations of one C function are
// ill-formed -- so a translation unit with both headers, <mtl/mtl.hpp> among
// them, did not compile (#725). While SuperLU's header is parsed, each name
// is renamed, so its prototypes declare names nothing references. That is
// safe: the header only declares them (it has no inline bodies calling them),
// and the calls are made inside the already compiled libsuperlu, which links
// against the real BLAS as before.
//
// The list is every double-precision BLAS routine MTL5 declares, a superset
// of what any SuperLU version declares; renaming a name the header does not
// mention is harmless. slu_Cnames.h is included first because, for Fortran
// name-mangling conventions other than the usual trailing underscore (Add_),
// it maps these names itself; a name it has already defined is left alone.
#include <slu_Cnames.h>
#  ifndef daxpy_
#    define daxpy_ mtl5_superlu_hidden_daxpy_
#    define MTL5_SLU_HID_daxpy_
#  endif
#  ifndef dcopy_
#    define dcopy_ mtl5_superlu_hidden_dcopy_
#    define MTL5_SLU_HID_dcopy_
#  endif
#  ifndef ddot_
#    define ddot_ mtl5_superlu_hidden_ddot_
#    define MTL5_SLU_HID_ddot_
#  endif
#  ifndef dscal_
#    define dscal_ mtl5_superlu_hidden_dscal_
#    define MTL5_SLU_HID_dscal_
#  endif
#  ifndef dnrm2_
#    define dnrm2_ mtl5_superlu_hidden_dnrm2_
#    define MTL5_SLU_HID_dnrm2_
#  endif
#  ifndef dasum_
#    define dasum_ mtl5_superlu_hidden_dasum_
#    define MTL5_SLU_HID_dasum_
#  endif
#  ifndef idamax_
#    define idamax_ mtl5_superlu_hidden_idamax_
#    define MTL5_SLU_HID_idamax_
#  endif
#  ifndef dgemv_
#    define dgemv_ mtl5_superlu_hidden_dgemv_
#    define MTL5_SLU_HID_dgemv_
#  endif
#  ifndef dger_
#    define dger_ mtl5_superlu_hidden_dger_
#    define MTL5_SLU_HID_dger_
#  endif
#  ifndef dsymv_
#    define dsymv_ mtl5_superlu_hidden_dsymv_
#    define MTL5_SLU_HID_dsymv_
#  endif
#  ifndef dtrmv_
#    define dtrmv_ mtl5_superlu_hidden_dtrmv_
#    define MTL5_SLU_HID_dtrmv_
#  endif
#  ifndef dtrsv_
#    define dtrsv_ mtl5_superlu_hidden_dtrsv_
#    define MTL5_SLU_HID_dtrsv_
#  endif
#  ifndef dgemm_
#    define dgemm_ mtl5_superlu_hidden_dgemm_
#    define MTL5_SLU_HID_dgemm_
#  endif
#  ifndef dgemmt_
#    define dgemmt_ mtl5_superlu_hidden_dgemmt_
#    define MTL5_SLU_HID_dgemmt_
#  endif
#  ifndef dsymm_
#    define dsymm_ mtl5_superlu_hidden_dsymm_
#    define MTL5_SLU_HID_dsymm_
#  endif
#  ifndef dsyrk_
#    define dsyrk_ mtl5_superlu_hidden_dsyrk_
#    define MTL5_SLU_HID_dsyrk_
#  endif
#  ifndef dsyr2k_
#    define dsyr2k_ mtl5_superlu_hidden_dsyr2k_
#    define MTL5_SLU_HID_dsyr2k_
#  endif
#  ifndef dtrmm_
#    define dtrmm_ mtl5_superlu_hidden_dtrmm_
#    define MTL5_SLU_HID_dtrmm_
#  endif
#  ifndef dtrsm_
#    define dtrsm_ mtl5_superlu_hidden_dtrsm_
#    define MTL5_SLU_HID_dtrsm_
#  endif
#include <slu_ddefs.h>
#  ifdef MTL5_SLU_HID_daxpy_
#    undef daxpy_
#    undef MTL5_SLU_HID_daxpy_
#  endif
#  ifdef MTL5_SLU_HID_dcopy_
#    undef dcopy_
#    undef MTL5_SLU_HID_dcopy_
#  endif
#  ifdef MTL5_SLU_HID_ddot_
#    undef ddot_
#    undef MTL5_SLU_HID_ddot_
#  endif
#  ifdef MTL5_SLU_HID_dscal_
#    undef dscal_
#    undef MTL5_SLU_HID_dscal_
#  endif
#  ifdef MTL5_SLU_HID_dnrm2_
#    undef dnrm2_
#    undef MTL5_SLU_HID_dnrm2_
#  endif
#  ifdef MTL5_SLU_HID_dasum_
#    undef dasum_
#    undef MTL5_SLU_HID_dasum_
#  endif
#  ifdef MTL5_SLU_HID_idamax_
#    undef idamax_
#    undef MTL5_SLU_HID_idamax_
#  endif
#  ifdef MTL5_SLU_HID_dgemv_
#    undef dgemv_
#    undef MTL5_SLU_HID_dgemv_
#  endif
#  ifdef MTL5_SLU_HID_dger_
#    undef dger_
#    undef MTL5_SLU_HID_dger_
#  endif
#  ifdef MTL5_SLU_HID_dsymv_
#    undef dsymv_
#    undef MTL5_SLU_HID_dsymv_
#  endif
#  ifdef MTL5_SLU_HID_dtrmv_
#    undef dtrmv_
#    undef MTL5_SLU_HID_dtrmv_
#  endif
#  ifdef MTL5_SLU_HID_dtrsv_
#    undef dtrsv_
#    undef MTL5_SLU_HID_dtrsv_
#  endif
#  ifdef MTL5_SLU_HID_dgemm_
#    undef dgemm_
#    undef MTL5_SLU_HID_dgemm_
#  endif
#  ifdef MTL5_SLU_HID_dgemmt_
#    undef dgemmt_
#    undef MTL5_SLU_HID_dgemmt_
#  endif
#  ifdef MTL5_SLU_HID_dsymm_
#    undef dsymm_
#    undef MTL5_SLU_HID_dsymm_
#  endif
#  ifdef MTL5_SLU_HID_dsyrk_
#    undef dsyrk_
#    undef MTL5_SLU_HID_dsyrk_
#  endif
#  ifdef MTL5_SLU_HID_dsyr2k_
#    undef dsyr2k_
#    undef MTL5_SLU_HID_dsyr2k_
#  endif
#  ifdef MTL5_SLU_HID_dtrmm_
#    undef dtrmm_
#    undef MTL5_SLU_HID_dtrmm_
#  endif
#  ifdef MTL5_SLU_HID_dtrsm_
#    undef dtrsm_
#    undef MTL5_SLU_HID_dtrsm_
#  endif

namespace mtl::interface {

/// RAII wrapper for SuperLU sparse LU solver.
///
/// Usage:
///   superlu_solver solver(A);
///   solver.solve(x, b);
class superlu_solver {
public:
    template <typename Parameters>
    explicit superlu_solver(const mat::compressed2D<double, Parameters>& A)
        : n_(static_cast<int>(A.num_rows()))
    {
        if (A.num_rows() != A.num_cols()) {
            throw std::invalid_argument("superlu_solver: matrix must be square");
        }
        convert_to_ccs(A);
    }

    ~superlu_solver() = default;

    superlu_solver(const superlu_solver&) = delete;
    superlu_solver& operator=(const superlu_solver&) = delete;
    superlu_solver(superlu_solver&&) noexcept = default;
    superlu_solver& operator=(superlu_solver&&) noexcept = default;

    /// Solve A*x = b. Each call performs a fresh factorization via dgssv.
    template <typename ParamsX, typename ParamsB>
    void solve(vec::dense_vector<double, ParamsX>& x,
               const vec::dense_vector<double, ParamsB>& b) const
    {
        if (static_cast<int>(x.size()) != n_ || static_cast<int>(b.size()) != n_) {
            throw std::invalid_argument("superlu_solver::solve: vector size mismatch");
        }

        // Copy b into x (dgssv overwrites the RHS in-place)
        for (int i = 0; i < n_; ++i)
            x(i) = b(i);

        // Local copies for SuperLU (it may modify the matrix)
        auto col_ptr = col_ptr_;
        auto row_ind = row_ind_;
        auto values  = values_;

        SuperMatrix A_slu, B_slu, L_slu, U_slu;
        int nnz = static_cast<int>(values.size());

        dCreate_CompCol_Matrix(&A_slu, n_, n_, nnz,
                               values.data(), row_ind.data(), col_ptr.data(),
                               SLU_NC, SLU_D, SLU_GE);
        dCreate_Dense_Matrix(&B_slu, n_, 1, x.data(), n_,
                             SLU_DN, SLU_D, SLU_GE);

        superlu_options_t options;
        set_default_options(&options);

        std::vector<int> perm_c(n_), perm_r(n_);
        SuperLUStat_t stat;
        StatInit(&stat);

        int info = 0;
        factor_nnz_ = 0;  // clear stale fill so a failed solve never reports it
        dgssv(&options, &A_slu, perm_c.data(), perm_r.data(),
              &L_slu, &U_slu, &B_slu, &stat, &info);

        if (info == 0) {
            // Record the factor fill (nnz of L + U). SuperLU stores L
            // supernodally, so its nnz includes the structural zeros inside the
            // supernodes -- i.e. SuperLU's actual factor storage footprint, the
            // honest quantity to compare against a scalar factor's nnz.
            factor_nnz_ = static_cast<std::size_t>(
                              static_cast<SCformat*>(L_slu.Store)->nnz)
                        + static_cast<std::size_t>(
                              static_cast<NCformat*>(U_slu.Store)->nnz);
        }

        StatFree(&stat);
        Destroy_SuperMatrix_Store(&A_slu);
        Destroy_SuperMatrix_Store(&B_slu);
        Destroy_SuperNode_Matrix(&L_slu);
        Destroy_CompCol_Matrix(&U_slu);

        if (info != 0) {
            throw std::runtime_error("SuperLU dgssv failed (info="
                                      + std::to_string(info) + ")");
        }
    }

    std::size_t num_rows() const { return static_cast<std::size_t>(n_); }
    std::size_t num_cols() const { return static_cast<std::size_t>(n_); }

    /// Fill of the most recent factorization: nnz(L) + nnz(U) (L counted
    /// supernodally, as SuperLU stores it). Zero until solve() has run.
    std::size_t factor_nnz() const { return factor_nnz_; }

private:
    int n_;
    std::vector<int> col_ptr_, row_ind_;
    std::vector<double> values_;
    mutable std::size_t factor_nnz_ = 0;

    template <typename Parameters>
    void convert_to_ccs(const mat::compressed2D<double, Parameters>& A) {
        std::size_t nnz = A.nnz();
        std::size_t ncols = A.num_cols();
        std::size_t nrows = A.num_rows();
        const auto& row_ptr = A.ref_major();
        const auto& col_idx = A.ref_minor();
        const auto& data = A.ref_data();

        col_ptr_.assign(ncols + 1, 0);
        for (std::size_t k = 0; k < nnz; ++k)
            col_ptr_[col_idx[k] + 1]++;
        for (std::size_t j = 1; j <= ncols; ++j)
            col_ptr_[j] += col_ptr_[j - 1];

        row_ind_.resize(nnz);
        values_.resize(nnz);
        std::vector<int> pos(col_ptr_.begin(), col_ptr_.end());
        for (std::size_t r = 0; r < nrows; ++r) {
            for (std::size_t k = row_ptr[r]; k < row_ptr[r + 1]; ++k) {
                int c = static_cast<int>(col_idx[k]);
                int dest = pos[c]++;
                row_ind_[dest] = static_cast<int>(r);
                values_[dest] = data[k];
            }
        }
    }
};

/// Convenience: factor and solve A*x = b in one call.
template <typename Parameters, typename ParamsX, typename ParamsB>
void superlu_solve(const mat::compressed2D<double, Parameters>& A,
                   vec::dense_vector<double, ParamsX>& x,
                   const vec::dense_vector<double, ParamsB>& b) {
    superlu_solver solver(A);
    solver.solve(x, b);
}

} // namespace mtl::interface

#endif // MTL5_HAS_SUPERLU
