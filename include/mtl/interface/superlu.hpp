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

// Include SuperLU headers directly
#include <slu_ddefs.h>

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
        dgssv(&options, &A_slu, perm_c.data(), perm_r.data(),
              &L_slu, &U_slu, &B_slu, &stat, &info);

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

private:
    int n_;
    std::vector<int> col_ptr_, row_ind_;
    std::vector<double> values_;

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
