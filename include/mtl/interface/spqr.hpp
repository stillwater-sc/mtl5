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

#include <climits>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/interface/cholmod_workspace.hpp>
#include <mtl/interface/vendor_threads.hpp>

#include <cholmod.h>
#include <SuiteSparseQR_C.h>

namespace mtl::interface {

/// RAII wrapper for SuiteSparseQR (SPQR) sparse QR solver.
/// Solves min ||A*x - b||_2 for rectangular or square systems.
///
/// Usage:
///   spqr_solver solver(A);
///   solver.solve(x, b);
///
/// The constructor copies A into CHOLMOD's column-compressed form; each
/// solve() factors that copy and solves (SuiteSparseQR_C_backslash).
///
/// Threads: solve() is const and may be called from several threads at once on
/// one solver. Each call runs on a workspace of its own and only reads the
/// stored matrix (#839).
class spqr_solver {
public:
    /// Copies A. Throws std::length_error if m, n or nnz exceed the int32
    /// CHOLMOD interface, std::invalid_argument if m < n, and
    /// std::runtime_error if CHOLMOD cannot start or allocate; nothing leaks
    /// on any of these paths (#774).
    template <typename Parameters>
    explicit spqr_solver(const mat::compressed2D<double, Parameters>& A)
        : nrows_(A.num_rows()), ncols_(A.num_cols())
    {
        if (nrows_ > std::size_t(INT_MAX) || ncols_ > std::size_t(INT_MAX) ||
            A.nnz() > std::size_t(INT_MAX))
            throw std::length_error("spqr_solver: m, n or nnz exceeds the int32 CHOLMOD interface");
        if (nrows_ < ncols_) {
            throw std::invalid_argument(
                "spqr_solver: requires m >= n (overdetermined or square)");
        }

        // The destructor does not run if the constructor throws, so the
        // matrix is freed here; common_ is a member and finishes itself.
        common_ = detail::make_cholmod_workspace();
        try {
            convert_to_cholmod_sparse(A);
        } catch (...) {
            release();
            throw;
        }
    }

    ~spqr_solver() { release(); }

    spqr_solver(const spqr_solver&) = delete;
    spqr_solver& operator=(const spqr_solver&) = delete;

    /// The workspace moves as a pointer, so it is finished once (#774). A
    /// moved-from solver is empty: sizes 0, and solve() throws.
    spqr_solver(spqr_solver&& o) noexcept
        : nrows_(std::exchange(o.nrows_, 0)), ncols_(std::exchange(o.ncols_, 0)),
          common_(std::move(o.common_)),
          sparse_A_(std::exchange(o.sparse_A_, nullptr)) {}

    spqr_solver& operator=(spqr_solver&& o) noexcept {
        if (this != &o) {
            release();
            nrows_ = std::exchange(o.nrows_, 0);
            ncols_ = std::exchange(o.ncols_, 0);
            common_ = std::move(o.common_);
            sparse_A_ = std::exchange(o.sparse_A_, nullptr);
        }
        return *this;
    }

    /// Solve min ||A*x - b||_2. Safe to call concurrently on one solver.
    template <typename ParamsX, typename ParamsB>
    void solve(vec::dense_vector<double, ParamsX>& x,
               const vec::dense_vector<double, ParamsB>& b) const
    {
        if (!sparse_A_)
            throw std::logic_error("spqr_solver::solve: moved-from solver");
        if (x.size() != ncols_ || b.size() != nrows_) {
            throw std::invalid_argument("spqr_solver::solve: vector size mismatch");
        }

        auto ws = detail::make_solve_workspace(*common_);
        cholmod_dense* b_chol = cholmod_allocate_dense(
            nrows_, 1, nrows_, CHOLMOD_REAL, ws.get());
        if (!b_chol) {
            throw std::runtime_error("SPQR: failed to allocate dense vector");
        }
        auto* b_data = static_cast<double*>(b_chol->x);
        for (std::size_t i = 0; i < nrows_; ++i)
            b_data[i] = b(i);

        // SPQR_ORDERING_DEFAULT = 0, tol = -1 means default. One thread
        // inside an MTL5 region (#640).
        cholmod_dense* x_chol = nullptr;
        {
            detail::nested_suitesparse_cap<cholmod_common> cap(*ws);
            x_chol = SuiteSparseQR_C_backslash(
                SPQR_ORDERING_DEFAULT, SPQR_DEFAULT_TOL,
                sparse_A_, b_chol, ws.get());
        }

        cholmod_free_dense(&b_chol, ws.get());

        if (!x_chol) {
            throw std::runtime_error("SPQR solve failed (status " +
                                     std::to_string(ws->status) + ")");
        }

        auto* x_data = static_cast<double*>(x_chol->x);
        for (std::size_t i = 0; i < ncols_; ++i)
            x(i) = x_data[i];

        cholmod_free_dense(&x_chol, ws.get());
    }

    std::size_t num_rows() const { return nrows_; }
    std::size_t num_cols() const { return ncols_; }

private:
    std::size_t nrows_ = 0;
    std::size_t ncols_ = 0;
    detail::cholmod_workspace common_;
    cholmod_sparse* sparse_A_ = nullptr;

    void release() noexcept {
        if (sparse_A_) cholmod_free_sparse(&sparse_A_, common_.get());
        common_.reset();   // cholmod_finish, then the struct
    }

    /// CRS to CCS. compressed2D is row-major only (a static_assert in
    /// compressed2D.hpp), so ref_major() is the row pointers and the copy is
    /// A itself, not its transpose.
    template <typename Parameters>
    void convert_to_cholmod_sparse(const mat::compressed2D<double, Parameters>& A) {
        std::size_t nnz = A.nnz();

        // stype = 0 means unsymmetric (general)
        sparse_A_ = cholmod_allocate_sparse(
            nrows_, ncols_, nnz, 1, 1, 0, CHOLMOD_REAL, common_.get());
        if (!sparse_A_) {
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
