#pragma once
// MTL5 -- Optional CHOLMOD interface for sparse Cholesky solves
// CHOLMOD provides high-performance supernodal and simplicial sparse
// Cholesky factorization for symmetric positive definite systems.
//
// Guarded by MTL5_HAS_CHOLMOD (set by CMake when MTL5_WITH_SUITESPARSE_CHOLMOD=ON)
//
// Reference: Chen, Davis, Hager, Rajamanickam, "Algorithm 887: CHOLMOD,
//            Supernodal Sparse Cholesky Factorization and Update/Downdate",
//            ACM TOMS, 35(3), 2008.

#ifdef MTL5_HAS_CHOLMOD

#include <climits>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/interface/vendor_threads.hpp>

#include <cholmod.h>

namespace mtl::interface {

/// RAII wrapper for CHOLMOD's sparse Cholesky, A = L L^T, for symmetric
/// positive definite A. Only the upper triangle of A is read.
///
/// One call does everything:
///   cholmod_solver solver(A);       // analyze + factorize
///   solver.solve(x, b);             // as often as needed
///
/// Or one phase at a time, to reuse the analysis for new values:
///   cholmod_solver solver;
///   solver.analyze(A);              // ordering + symbolic factorization
///   solver.factorize();             // numeric factorization of A's values
///   solver.factorize(A2);           // same pattern, new values
///   solver.solve(x, b);
///
/// A matrix that is not positive definite throws std::runtime_error from
/// factorize(), on the simplicial path as well as the supernodal one: the
/// factorization is always L L^T (Common->final_ll), because CHOLMOD's
/// simplicial default, L D L^T, factors an indefinite matrix without a word.
class cholmod_solver {
public:
    /// A started CHOLMOD workspace, with nothing analyzed yet. Settings in
    /// common() (ordering, supernodal choice, ...) apply to the next analyze().
    cholmod_solver() : common_(new cholmod_common) {
        cholmod_start(common_.get());
        common_->final_ll = 1;
    }

    /// Analyze and factorize A. A delegating constructor, so the destructor
    /// frees the workspace if either phase throws (#774).
    template <typename Parameters>
    explicit cholmod_solver(const mat::compressed2D<double, Parameters>& A)
        : cholmod_solver()
    {
        analyze(A);
        factorize();
    }

    ~cholmod_solver() { release(); }

    cholmod_solver(const cholmod_solver&) = delete;
    cholmod_solver& operator=(const cholmod_solver&) = delete;

    /// The workspace lives on the heap and moves as a pointer. Copying the
    /// cholmod_common struct itself would leave two owners of its buffers,
    /// and both destructors would free them (#774).
    cholmod_solver(cholmod_solver&& o) noexcept
        : n_(o.n_), nnz_(o.nnz_), common_(std::move(o.common_)),
          A_(std::exchange(o.A_, nullptr)), L_(std::exchange(o.L_, nullptr)),
          factored_(std::exchange(o.factored_, false)) {}

    cholmod_solver& operator=(cholmod_solver&& o) noexcept {
        if (this != &o) {
            release();
            n_ = o.n_;
            nnz_ = o.nnz_;
            common_ = std::move(o.common_);
            A_ = std::exchange(o.A_, nullptr);
            L_ = std::exchange(o.L_, nullptr);
            factored_ = std::exchange(o.factored_, false);
        }
        return *this;
    }

    /// Ordering and symbolic factorization of A's pattern. Copies A (upper
    /// triangle) into CHOLMOD's column-compressed form; discards any earlier
    /// analysis or factor.
    template <typename Parameters>
    void analyze(const mat::compressed2D<double, Parameters>& A) {
        require_workspace("analyze");
        if (A.num_rows() != A.num_cols())
            throw std::invalid_argument("cholmod_solver: matrix must be square");
        if (A.num_rows() > std::size_t(INT_MAX) || A.nnz() > std::size_t(INT_MAX))
            throw std::length_error("cholmod_solver: n or nnz exceeds the int32 CHOLMOD interface");
        free_factor_and_matrix();
        n_ = A.num_rows();
        nnz_ = A.nnz();   // not A_->nzmax: CHOLMOD allocates at least one entry

        // stype = 1: symmetric, upper triangle used; sorted, packed.
        A_ = cholmod_allocate_sparse(n_, n_, A.nnz(), 1, 1, 1, CHOLMOD_REAL, common_.get());
        if (!A_) throw_status("failed to allocate the sparse matrix");
        copy_ccs(A, /*refill=*/false);

        detail::nested_suitesparse_cap<cholmod_common> cap(*common_);
        L_ = cholmod_analyze(A_, common_.get());
        if (!L_) throw_status("symbolic analysis failed");
    }

    /// Numeric factorization of the matrix given to analyze().
    void factorize() {
        require_analyzed("factorize");
        factored_ = false;
        detail::nested_suitesparse_cap<cholmod_common> cap(*common_);
        if (!cholmod_factorize(A_, L_, common_.get()))
            throw_status("numeric factorization failed");
        if (common_->status == CHOLMOD_NOT_POSDEF || L_->minor < n_)
            throw std::runtime_error("CHOLMOD: matrix is not positive definite (column " +
                                     std::to_string(L_->minor) + ")");
        factored_ = true;
    }

    /// Numeric factorization of A, which must have the pattern analyze() saw.
    template <typename Parameters>
    void factorize(const mat::compressed2D<double, Parameters>& A) {
        require_analyzed("factorize");
        if (A.num_rows() != n_ || A.num_cols() != n_ || A.nnz() != nnz_)
            throw std::invalid_argument("cholmod_solver::factorize: pattern differs from analyze()");
        copy_ccs(A, /*refill=*/true);   // a mismatch throws with the old factor intact
        factorize();
    }

    template <typename ParamsX, typename ParamsB>
    void solve(vec::dense_vector<double, ParamsX>& x,
               const vec::dense_vector<double, ParamsB>& b) const
    {
        if (!factored_)
            throw std::logic_error("cholmod_solver::solve: no factorization");
        if (x.size() != n_ || b.size() != n_)
            throw std::invalid_argument("cholmod_solver::solve: vector size mismatch");

        cholmod_dense* b_chol = cholmod_allocate_dense(n_, 1, n_, CHOLMOD_REAL, common_.get());
        if (!b_chol) throw_status("failed to allocate a dense vector");
        auto* b_data = static_cast<double*>(b_chol->x);
        for (std::size_t i = 0; i < n_; ++i) b_data[i] = b(i);

        // One thread inside an MTL5 region (#640)
        cholmod_dense* x_chol = nullptr;
        {
            detail::nested_suitesparse_cap<cholmod_common> cap(*common_);
            x_chol = cholmod_solve(CHOLMOD_A, L_, b_chol, common_.get());
        }
        cholmod_free_dense(&b_chol, common_.get());
        if (!x_chol) throw_status("solve failed");

        auto* x_data = static_cast<double*>(x_chol->x);
        for (std::size_t i = 0; i < n_; ++i) x(i) = x_data[i];
        cholmod_free_dense(&x_chol, common_.get());
    }

    std::size_t num_rows() const { return n_; }
    std::size_t num_cols() const { return n_; }
    bool analyzed() const { return L_ != nullptr; }
    bool factored() const { return factored_; }
    /// Whether analyze() chose the supernodal factorization.
    bool is_supernodal() const { return L_ && L_->is_super; }
    /// Nonzeros in L, from the last analyze() (Common->lnz).
    double factor_nnz() const { return common_ ? common_->lnz : 0.0; }

    /// CHOLMOD's settings and statistics. Change settings before analyze().
    cholmod_common& common() {
        require_workspace("common");
        return *common_;
    }

private:
    std::size_t n_ = 0;
    std::size_t nnz_ = 0;
    std::unique_ptr<cholmod_common> common_;
    cholmod_sparse* A_ = nullptr;
    cholmod_factor* L_ = nullptr;
    bool factored_ = false;

    void free_factor_and_matrix() noexcept {
        if (L_) cholmod_free_factor(&L_, common_.get());
        if (A_) cholmod_free_sparse(&A_, common_.get());
        factored_ = false;
    }

    void release() noexcept {
        if (!common_) return;   // moved from
        free_factor_and_matrix();
        cholmod_finish(common_.get());
        common_.reset();
    }

    void require_workspace(const char* what) const {
        if (!common_)
            throw std::logic_error(std::string("cholmod_solver::") + what + ": moved-from solver");
    }

    void require_analyzed(const char* what) const {
        require_workspace(what);
        if (!L_)
            throw std::logic_error(std::string("cholmod_solver::") + what + ": analyze() first");
    }

    [[noreturn]] void throw_status(const char* what) const {
        throw std::runtime_error(std::string("CHOLMOD: ") + what + " (status " +
                                 std::to_string(common_->status) + ")");
    }

    /// CRS to CCS. Row r's entries scatter into column c in increasing r, so
    /// every column comes out sorted, as the allocation promised CHOLMOD.
    /// With refill, A_ already holds the analyzed pattern: A's pattern is
    /// checked against it in full before any value is replaced, so a
    /// mismatch leaves the stored matrix as it was.
    template <typename Parameters>
    void copy_ccs(const mat::compressed2D<double, Parameters>& A, bool refill) {
        const std::size_t nnz = A.nnz();
        auto* col_ptr = static_cast<int*>(A_->p);
        auto* row_ind = static_cast<int*>(A_->i);
        auto* values  = static_cast<double*>(A_->x);

        const auto& row_ptrs = A.ref_major();
        const auto& col_idxs = A.ref_minor();
        const auto& data = A.ref_data();

        std::vector<int> counts(n_ + 1, 0);
        for (std::size_t k = 0; k < nnz; ++k) counts[col_idxs[k] + 1]++;
        for (std::size_t j = 1; j <= n_; ++j) counts[j] += counts[j - 1];

        auto scatter = [&](bool write) {
            std::vector<int> pos(counts.begin(), counts.end() - 1);
            for (std::size_t r = 0; r < n_; ++r) {
                for (std::size_t k = row_ptrs[r]; k < row_ptrs[r + 1]; ++k) {
                    const int dest = pos[col_idxs[k]]++;
                    if (write) {
                        row_ind[dest] = static_cast<int>(r);
                        values[dest] = data[k];
                    } else if (row_ind[dest] != static_cast<int>(r)) {
                        return false;
                    }
                }
            }
            return true;
        };

        if (refill) {
            bool same = true;
            for (std::size_t j = 0; j <= n_ && same; ++j) same = col_ptr[j] == counts[j];
            if (!same || !scatter(false))
                throw std::invalid_argument("cholmod_solver::factorize: pattern differs from analyze()");
        }
        for (std::size_t j = 0; j <= n_; ++j) col_ptr[j] = counts[j];
        scatter(true);
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
