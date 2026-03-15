#pragma once
// MTL5 -- Optional CHOLMOD interface for sparse Cholesky solves
// CHOLMOD provides high-performance supernodal and simplicial sparse
// Cholesky factorization for symmetric positive definite systems.
//
// Guarded by MTL5_HAS_CHOLMOD (set by CMake when MTL5_ENABLE_CHOLMOD=ON)
//
// Reference: Chen, Davis, Hager, Rajamanickam, "Algorithm 887: CHOLMOD,
//            Supernodal Sparse Cholesky Factorization and Update/Downdate",
//            ACM TOMS, 35(3), 2008.

#ifdef MTL5_HAS_CHOLMOD

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/vec/dense_vector.hpp>

// -- CHOLMOD C API declarations -----------------------------------------------
extern "C" {

// Opaque CHOLMOD types
typedef struct cholmod_common_struct cholmod_common;
typedef struct cholmod_sparse_struct cholmod_sparse;
typedef struct cholmod_dense_struct cholmod_dense;
typedef struct cholmod_factor_struct cholmod_factor;

// CHOLMOD core routines
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

cholmod_factor* cholmod_analyze(cholmod_sparse* A, cholmod_common* Common);
int cholmod_factorize(cholmod_sparse* A, cholmod_factor* L, cholmod_common* Common);
cholmod_dense* cholmod_solve(int sys, cholmod_factor* L, cholmod_dense* B,
                             cholmod_common* Common);
int cholmod_free_factor(cholmod_factor** L, cholmod_common* Common);

// CHOLMOD xtype constants
#ifndef CHOLMOD_REAL
#define CHOLMOD_REAL 1
#endif

// CHOLMOD solve system types
#ifndef CHOLMOD_A
#define CHOLMOD_A 0
#endif

} // extern "C"

namespace mtl::interface {

/// RAII wrapper for CHOLMOD sparse Cholesky solver.
/// For symmetric positive definite systems.
///
/// Usage:
///   cholmod_solver solver(A);
///   solver.solve(x, b);
class cholmod_solver {
public:
    template <typename Parameters>
    explicit cholmod_solver(const mat::compressed2D<double, Parameters>& A)
        : n_(A.num_rows()), common_(nullptr), factor_(nullptr),
          sparse_A_(nullptr)
    {
        if (A.num_rows() != A.num_cols()) {
            throw std::invalid_argument("cholmod_solver: matrix must be square");
        }

        // Allocate and start CHOLMOD common
        common_ = new cholmod_common;
        cholmod_start(common_);

        // Convert to CHOLMOD sparse (CSC, symmetric upper)
        convert_to_cholmod_sparse(A);

        // Symbolic analysis
        factor_ = cholmod_analyze(sparse_A_, common_);
        if (!factor_) {
            cleanup();
            throw std::runtime_error("CHOLMOD symbolic analysis failed");
        }

        // Numeric factorization
        int ok = cholmod_factorize(sparse_A_, factor_, common_);
        if (!ok) {
            cleanup();
            throw std::runtime_error("CHOLMOD numeric factorization failed");
        }
    }

    ~cholmod_solver() { cleanup(); }

    cholmod_solver(const cholmod_solver&) = delete;
    cholmod_solver& operator=(const cholmod_solver&) = delete;

    cholmod_solver(cholmod_solver&& o) noexcept
        : n_(o.n_), common_(o.common_), factor_(o.factor_),
          sparse_A_(o.sparse_A_)
    {
        o.common_ = nullptr;
        o.factor_ = nullptr;
        o.sparse_A_ = nullptr;
    }

    cholmod_solver& operator=(cholmod_solver&& o) noexcept {
        if (this != &o) {
            cleanup();
            n_ = o.n_;
            common_ = o.common_;
            factor_ = o.factor_;
            sparse_A_ = o.sparse_A_;
            o.common_ = nullptr;
            o.factor_ = nullptr;
            o.sparse_A_ = nullptr;
        }
        return *this;
    }

    template <typename ParamsX, typename ParamsB>
    void solve(vec::dense_vector<double, ParamsX>& x,
               const vec::dense_vector<double, ParamsB>& b) const
    {
        if (x.size() != n_ || b.size() != n_) {
            throw std::invalid_argument("cholmod_solver::solve: vector size mismatch");
        }

        // Create CHOLMOD dense vector for b
        cholmod_dense* b_chol = cholmod_allocate_dense(n_, 1, n_,
                                                        CHOLMOD_REAL, common_);
        if (!b_chol) {
            throw std::runtime_error("CHOLMOD: failed to allocate dense vector");
        }
        auto* b_data = static_cast<double*>(b_chol->x);
        for (std::size_t i = 0; i < n_; ++i)
            b_data[i] = b(i);

        // Solve
        cholmod_dense* x_chol = cholmod_solve(CHOLMOD_A, factor_, b_chol, common_);
        cholmod_free_dense(&b_chol, common_);

        if (!x_chol) {
            throw std::runtime_error("CHOLMOD solve failed");
        }

        auto* x_data = static_cast<double*>(x_chol->x);
        for (std::size_t i = 0; i < n_; ++i)
            x(i) = x_data[i];

        cholmod_free_dense(&x_chol, common_);
    }

    std::size_t num_rows() const { return n_; }
    std::size_t num_cols() const { return n_; }

private:
    std::size_t n_;
    cholmod_common* common_;
    cholmod_factor* factor_;
    cholmod_sparse* sparse_A_;

    void cleanup() {
        if (factor_ && common_) cholmod_free_factor(&factor_, common_);
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
        std::size_t ncols = A.num_cols();

        // stype = 1 means upper triangular stored (symmetric)
        sparse_A_ = cholmod_allocate_sparse(n_, n_, nnz,
                                             1, 1, 1, CHOLMOD_REAL, common_);
        if (!sparse_A_) {
            cleanup();
            throw std::runtime_error("CHOLMOD: failed to allocate sparse matrix");
        }

        // Convert CRS to CSC and fill CHOLMOD sparse
        auto* col_ptr = static_cast<int*>(sparse_A_->p);
        auto* row_ind = static_cast<int*>(sparse_A_->i);
        auto* values  = static_cast<double*>(sparse_A_->x);

        const auto& row_ptrs = A.ref_major();
        const auto& col_idxs = A.ref_minor();
        const auto& data = A.ref_data();

        // Count entries per column
        std::vector<int> counts(ncols + 1, 0);
        for (std::size_t k = 0; k < nnz; ++k)
            counts[col_idxs[k] + 1]++;
        for (std::size_t j = 1; j <= ncols; ++j)
            counts[j] += counts[j - 1];

        for (std::size_t j = 0; j <= ncols; ++j)
            col_ptr[j] = counts[j];

        std::vector<int> pos(counts.begin(), counts.end());
        for (std::size_t r = 0; r < n_; ++r) {
            for (std::size_t k = row_ptrs[r]; k < row_ptrs[r + 1]; ++k) {
                int c = static_cast<int>(col_idxs[k]);
                int dest = pos[c]++;
                row_ind[dest] = static_cast<int>(r);
                values[dest] = data[k];
            }
        }
    }
};

/// Convenience: factor and solve A*x = b in one call.
template <typename Parameters, typename ParamsX, typename ParamsB>
void cholmod_solve(const mat::compressed2D<double, Parameters>& A,
                   vec::dense_vector<double, ParamsX>& x,
                   const vec::dense_vector<double, ParamsB>& b) {
    cholmod_solver solver(A);
    solver.solve(x, b);
}

} // namespace mtl::interface

#endif // MTL5_HAS_CHOLMOD
