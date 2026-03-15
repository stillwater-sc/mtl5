#pragma once
// MTL5 -- Optional SuiteSparseQR (SPQR) interface for sparse QR solves
// SPQR provides multifrontal multithreaded rank-revealing sparse QR
// factorization for least-squares problems and rank determination.
//
// Guarded by MTL5_HAS_SPQR (set by CMake when MTL5_ENABLE_SPQR=ON)
//
// Reference: Davis, "Algorithm 915: SuiteSparseQR, a Multifrontal
//            Multithreaded Rank-Revealing Sparse QR Factorization Method",
//            ACM TOMS, 38(1), 2011.

#ifdef MTL5_HAS_SPQR

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/vec/dense_vector.hpp>

// -- SPQR C API declarations --------------------------------------------------
// SPQR depends on CHOLMOD for its matrix types.
extern "C" {

// Forward-declare CHOLMOD types used by SPQR
typedef struct cholmod_common_struct cholmod_common;
typedef struct cholmod_sparse_struct cholmod_sparse;
typedef struct cholmod_dense_struct cholmod_dense;

int cholmod_start(cholmod_common* Common);
int cholmod_finish(cholmod_common* Common);

cholmod_sparse* cholmod_allocate_sparse(
    std::size_t nrow, std::size_t ncol, std::size_t nzmax,
    int sorted, int packed, int stype, int xtype,
    cholmod_common* Common);
int cholmod_free_sparse(cholmod_sparse** A, cholmod_common* Common);

cholmod_dense* cholmod_allocate_dense(
    std::size_t nrow, std::size_t ncol, std::size_t d,
    int xtype, cholmod_common* Common);
int cholmod_free_dense(cholmod_dense** X, cholmod_common* Common);

#ifndef CHOLMOD_REAL
#define CHOLMOD_REAL 1
#endif

// SPQR C interface
// SuiteSparseQR_C_backslash: solves min ||Ax - b|| or Ax = b
cholmod_dense* SuiteSparseQR_C_backslash(
    int ordering,       // fill-reducing ordering method
    double tol,         // rank detection tolerance
    cholmod_sparse* A,
    cholmod_dense* B,
    cholmod_common* cc);

// SPQR ordering constants
#ifndef SPQR_ORDERING_DEFAULT
#define SPQR_ORDERING_DEFAULT 0
#endif

} // extern "C"

namespace mtl::interface {

/// RAII wrapper for SuiteSparseQR (SPQR) sparse QR solver.
/// Solves min ||A*x - b||_2 for rectangular or square systems.
///
/// Usage:
///   spqr_solver solver(A);
///   solver.solve(x, b);
class spqr_solver {
public:
    template <typename Parameters>
    explicit spqr_solver(const mat::compressed2D<double, Parameters>& A)
        : nrows_(A.num_rows()), ncols_(A.num_cols()), common_(nullptr),
          sparse_A_(nullptr)
    {
        if (nrows_ < ncols_) {
            throw std::invalid_argument(
                "spqr_solver: requires m >= n (overdetermined or square)");
        }

        common_ = new cholmod_common;
        cholmod_start(common_);
        convert_to_cholmod_sparse(A);
    }

    ~spqr_solver() { cleanup(); }

    spqr_solver(const spqr_solver&) = delete;
    spqr_solver& operator=(const spqr_solver&) = delete;

    spqr_solver(spqr_solver&& o) noexcept
        : nrows_(o.nrows_), ncols_(o.ncols_),
          common_(o.common_), sparse_A_(o.sparse_A_)
    {
        o.common_ = nullptr;
        o.sparse_A_ = nullptr;
    }

    spqr_solver& operator=(spqr_solver&& o) noexcept {
        if (this != &o) {
            cleanup();
            nrows_ = o.nrows_;
            ncols_ = o.ncols_;
            common_ = o.common_;
            sparse_A_ = o.sparse_A_;
            o.common_ = nullptr;
            o.sparse_A_ = nullptr;
        }
        return *this;
    }

    /// Solve min ||A*x - b||_2.
    template <typename ParamsX, typename ParamsB>
    void solve(vec::dense_vector<double, ParamsX>& x,
               const vec::dense_vector<double, ParamsB>& b) const
    {
        if (x.size() != ncols_ || b.size() != nrows_) {
            throw std::invalid_argument("spqr_solver::solve: vector size mismatch");
        }

        // Create CHOLMOD dense for b
        cholmod_dense* b_chol = cholmod_allocate_dense(nrows_, 1, nrows_,
                                                        CHOLMOD_REAL, common_);
        if (!b_chol) {
            throw std::runtime_error("SPQR: failed to allocate dense vector");
        }
        auto* b_data = static_cast<double*>(b_chol->x);
        for (std::size_t i = 0; i < nrows_; ++i)
            b_data[i] = b(i);

        // Solve via SPQR backslash
        cholmod_dense* x_chol = SuiteSparseQR_C_backslash(
            SPQR_ORDERING_DEFAULT, -1.0,  // -1 = default tolerance
            sparse_A_, b_chol, common_);

        cholmod_free_dense(&b_chol, common_);

        if (!x_chol) {
            throw std::runtime_error("SPQR solve failed");
        }

        auto* x_data = static_cast<double*>(x_chol->x);
        for (std::size_t i = 0; i < ncols_; ++i)
            x(i) = x_data[i];

        cholmod_free_dense(&x_chol, common_);
    }

    std::size_t num_rows() const { return nrows_; }
    std::size_t num_cols() const { return ncols_; }

private:
    std::size_t nrows_, ncols_;
    cholmod_common* common_;
    cholmod_sparse* sparse_A_;

    void cleanup() {
        if (sparse_A_ && common_) cholmod_free_sparse(&sparse_A_, common_);
        if (common_) {
            cholmod_finish(common_);
            delete common_;
            common_ = nullptr;
        }
    }

    template <typename Parameters>
    void convert_to_cholmod_sparse(const mat::compressed2D<double, Parameters>& A) {
        std::size_t nnz = A.nnz();

        // stype = 0 means unsymmetric (general)
        sparse_A_ = cholmod_allocate_sparse(nrows_, ncols_, nnz,
                                             1, 1, 0, CHOLMOD_REAL, common_);
        if (!sparse_A_) {
            cleanup();
            throw std::runtime_error("SPQR: failed to allocate sparse matrix");
        }

        auto* col_ptr = static_cast<int*>(sparse_A_->p);
        auto* row_ind = static_cast<int*>(sparse_A_->i);
        auto* values  = static_cast<double*>(sparse_A_->x);

        const auto& row_ptrs = A.ref_major();
        const auto& col_idxs = A.ref_minor();
        const auto& data = A.ref_data();

        std::vector<int> counts(ncols_ + 1, 0);
        for (std::size_t k = 0; k < nnz; ++k)
            counts[col_idxs[k] + 1]++;
        for (std::size_t j = 1; j <= ncols_; ++j)
            counts[j] += counts[j - 1];

        for (std::size_t j = 0; j <= ncols_; ++j)
            col_ptr[j] = counts[j];

        std::vector<int> pos(counts.begin(), counts.end());
        for (std::size_t r = 0; r < nrows_; ++r) {
            for (std::size_t k = row_ptrs[r]; k < row_ptrs[r + 1]; ++k) {
                int c = static_cast<int>(col_idxs[k]);
                int dest = pos[c]++;
                row_ind[dest] = static_cast<int>(r);
                values[dest] = data[k];
            }
        }
    }
};

/// Convenience: solve min ||A*x - b||_2 in one call.
template <typename Parameters, typename ParamsX, typename ParamsB>
void spqr_solve(const mat::compressed2D<double, Parameters>& A,
                vec::dense_vector<double, ParamsX>& x,
                const vec::dense_vector<double, ParamsB>& b) {
    spqr_solver solver(A);
    solver.solve(x, b);
}

} // namespace mtl::interface

#endif // MTL5_HAS_SPQR
