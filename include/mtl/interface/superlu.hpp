#pragma once
// MTL5 -- Optional SuperLU interface for sparse direct solves
// SuperLU uses a supernodal left-looking LU algorithm for general
// unsymmetric sparse systems. It groups contiguous columns with
// similar structure into supernodes for efficient BLAS operations.
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

// -- SuperLU C API declarations -----------------------------------------------
// SuperLU uses its own matrix types. We forward-declare the minimal API needed.
extern "C" {

// SuperLU matrix storage types
typedef enum { SLU_NC, SLU_NCP, SLU_NR, SLU_SC, SLU_SCP, SLU_SR,
               SLU_DN, SLU_NR_loc } Stype_t;
typedef enum { SLU_S, SLU_D, SLU_C, SLU_Z } Dtype_t;
typedef enum { SLU_GE, SLU_TRILU, SLU_TRUU, SLU_SYL, SLU_HEL, SLU_HEU } Mtype_t;

typedef struct {
    Stype_t Stype;
    Dtype_t Dtype;
    Mtype_t Mtype;
    int nrow, ncol;
    void* Store;
} SuperMatrix;

typedef struct {
    int nnz;
    double* nzval;
    int* rowind;
    int* colptr;
} NCformat;

typedef struct {
    int lda;
    double* nzval;
} DNformat;

// SuperLU options
typedef enum { DOFACT, SamePattern, SamePattern_SameRowPerm, FACTORED } fact_t;
typedef enum { NATURAL, MMD_ATA, MMD_AT_PLUS_A, COLAMD as SLU_COLAMD,
               MY_PERMC } colperm_t;
typedef enum { NOTRANS, TRANS, CONJ } trans_t;
typedef enum { NOEQUIL, ROW, COL, BOTH } DiagScale_t;
typedef enum { NOREFINE, SLU_SINGLE=1, SLU_DOUBLE, SLU_EXTRA } IterRefine_t;
typedef enum { LUSUP, UCOL, LSUB, USUB, LLVL, ULVL } MemType;
typedef enum { NO, YES } yes_no_t;
typedef enum { SYSTEM, USER } LU_space_t;
typedef enum { ONE_NORM, TWO_NORM, INF_NORM } norm_t;

typedef struct {
    fact_t Fact;
    yes_no_t Equil;
    colperm_t ColPerm;
    trans_t Trans;
    IterRefine_t IterRefine;
    double DiagPivotThresh;
    yes_no_t SymmetricMode;
    yes_no_t PivotGrowth;
    yes_no_t ConditionNumber;
    yes_no_t PrintStat;
    int nnzL, nnzU;
    double* work;
    int lwork;
    int nprocs;
    int relax;
    int panel_size;
    double rcond;
    int expansion_ratio;
} superlu_options_t;

typedef struct {
    float for_lu, total_needed;
    int expansions;
} mem_usage_t;

typedef struct {
    int panel_size;
    int relax;
    double diag_pivot_thresh;
    double drop_tol;
} GlobalLU_t;

// SuperLU function prototypes
void set_default_options(superlu_options_t* options);
void dCreate_CompCol_Matrix(SuperMatrix* A, int m, int n, int nnz,
                            double* nzval, int* rowind, int* colptr,
                            Stype_t stype, Dtype_t dtype, Mtype_t mtype);
void dCreate_Dense_Matrix(SuperMatrix* X, int m, int n, double* x, int ldx,
                          Stype_t stype, Dtype_t dtype, Mtype_t mtype);
void dgssv(superlu_options_t* options, SuperMatrix* A, int* perm_c, int* perm_r,
           SuperMatrix* L, SuperMatrix* U, SuperMatrix* B,
           /* SuperLUStat_t* */ void* stat, int* info);
void StatInit(void* stat);
void StatFree(void* stat);
void Destroy_SuperMatrix_Store(SuperMatrix* A);
void Destroy_CompCol_Matrix(SuperMatrix* A);
void Destroy_SuperNode_Matrix(SuperMatrix* L);
void Destroy_CompCol_Permuted(SuperMatrix* U);
void Destroy_Dense_Matrix(SuperMatrix* B);

} // extern "C"

namespace mtl::interface {

/// RAII wrapper for SuperLU sparse LU solver.
/// Performs one-shot factorization + solve (SuperLU's dgssv combines both).
///
/// Usage:
///   superlu_solver solver(A);
///   solver.solve(x, b);
class superlu_solver {
public:
    /// Factor and prepare the sparse matrix A for solving.
    template <typename Parameters>
    explicit superlu_solver(const mat::compressed2D<double, Parameters>& A)
        : n_(static_cast<int>(A.num_rows()))
    {
        if (A.num_rows() != A.num_cols()) {
            throw std::invalid_argument("superlu_solver: matrix must be square");
        }

        // Convert CRS to CCS (SuperLU uses CSC = CompCol format)
        convert_to_ccs(A);
    }

    ~superlu_solver() = default;

    superlu_solver(const superlu_solver&) = delete;
    superlu_solver& operator=(const superlu_solver&) = delete;
    superlu_solver(superlu_solver&&) noexcept = default;
    superlu_solver& operator=(superlu_solver&&) noexcept = default;

    /// Solve A*x = b. Each call re-factors (SuperLU dgssv is combined).
    template <typename ParamsX, typename ParamsB>
    void solve(vec::dense_vector<double, ParamsX>& x,
               const vec::dense_vector<double, ParamsB>& b) const
    {
        if (static_cast<int>(x.size()) != n_ || static_cast<int>(b.size()) != n_) {
            throw std::invalid_argument("superlu_solver::solve: vector size mismatch");
        }

        // Copy b into x (SuperLU overwrites the RHS matrix in-place)
        for (int i = 0; i < n_; ++i)
            x(i) = b(i);

        // Create SuperLU matrices (local copies for thread safety)
        auto col_ptr = col_ptr_;
        auto row_ind = row_ind_;
        auto values  = values_;

        SuperMatrix A_slu, B_slu, L_slu, U_slu;
        dCreate_CompCol_Matrix(&A_slu, n_, n_, static_cast<int>(values.size()),
                               values.data(), row_ind.data(), col_ptr.data(),
                               SLU_NC, SLU_D, SLU_GE);
        dCreate_Dense_Matrix(&B_slu, n_, 1, x.data(), n_,
                             SLU_DN, SLU_D, SLU_GE);

        superlu_options_t options;
        set_default_options(&options);

        std::vector<int> perm_c(n_), perm_r(n_);

        // SuperLU stat (opaque, 128 bytes should be enough)
        alignas(8) char stat_buf[256]{};
        StatInit(stat_buf);

        int info = 0;
        dgssv(&options, &A_slu, perm_c.data(), perm_r.data(),
              &L_slu, &U_slu, &B_slu, stat_buf, &info);

        StatFree(stat_buf);
        Destroy_SuperMatrix_Store(&A_slu);
        Destroy_SuperMatrix_Store(&B_slu);
        Destroy_SuperNode_Matrix(&L_slu);
        Destroy_CompCol_Permuted(&U_slu);

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
