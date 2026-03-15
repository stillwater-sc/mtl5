#pragma once
// MTL5 -- Optional KLU interface for sparse direct solves
// KLU (Clark Kent LU) is a sparse LU solver optimized for circuit simulation
// matrices. It uses Dulmage-Mendelsohn decomposition to permute the matrix
// into block triangular form, then applies Gilbert-Peierls left-looking LU
// to each diagonal block independently.
//
// Guarded by MTL5_HAS_KLU (set by CMake when MTL5_ENABLE_KLU=ON)
//
// Reference: Davis & Palamadai Natarajan, "Algorithm 907: KLU, A Direct
//            Sparse Solver for Circuit Simulation Problems", ACM TOMS, 2010.

#ifdef MTL5_HAS_KLU

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/vec/dense_vector.hpp>

#include <klu.h>

namespace mtl::interface {

/// RAII wrapper for KLU sparse LU solver.
/// Optimized for circuit simulation matrices with block triangular structure.
///
/// Usage:
///   klu_solver solver(A);
///   solver.solve(x, b);
class klu_solver {
public:
    template <typename Parameters>
    explicit klu_solver(const mat::compressed2D<double, Parameters>& A)
        : n_(static_cast<int>(A.num_rows())),
          symbolic_(nullptr), numeric_(nullptr)
    {
        if (A.num_rows() != A.num_cols()) {
            throw std::invalid_argument("klu_solver: matrix must be square");
        }

        klu_defaults(&common_);
        convert_to_ccs(A);

        symbolic_ = klu_analyze(n_, Cp_.data(), Ci_.data(), &common_);
        if (!symbolic_) {
            throw std::runtime_error("KLU symbolic analysis failed");
        }

        numeric_ = klu_factor(Cp_.data(), Ci_.data(), Cx_.data(),
                              symbolic_, &common_);
        if (!numeric_) {
            klu_free_symbolic(&symbolic_, &common_);
            throw std::runtime_error("KLU numeric factorization failed");
        }
    }

    ~klu_solver() {
        if (numeric_)  klu_free_numeric(&numeric_, &common_);
        if (symbolic_) klu_free_symbolic(&symbolic_, &common_);
    }

    klu_solver(const klu_solver&) = delete;
    klu_solver& operator=(const klu_solver&) = delete;

    klu_solver(klu_solver&& o) noexcept
        : n_(o.n_), common_(o.common_),
          Cp_(std::move(o.Cp_)), Ci_(std::move(o.Ci_)), Cx_(std::move(o.Cx_)),
          symbolic_(o.symbolic_), numeric_(o.numeric_)
    {
        o.symbolic_ = nullptr;
        o.numeric_ = nullptr;
    }

    klu_solver& operator=(klu_solver&& o) noexcept {
        if (this != &o) {
            if (numeric_)  klu_free_numeric(&numeric_, &common_);
            if (symbolic_) klu_free_symbolic(&symbolic_, &common_);
            n_ = o.n_;
            common_ = o.common_;
            Cp_ = std::move(o.Cp_);
            Ci_ = std::move(o.Ci_);
            Cx_ = std::move(o.Cx_);
            symbolic_ = o.symbolic_;
            numeric_ = o.numeric_;
            o.symbolic_ = nullptr;
            o.numeric_ = nullptr;
        }
        return *this;
    }

    template <typename ParamsX, typename ParamsB>
    void solve(vec::dense_vector<double, ParamsX>& x,
               const vec::dense_vector<double, ParamsB>& b) const
    {
        if (static_cast<int>(x.size()) != n_ || static_cast<int>(b.size()) != n_) {
            throw std::invalid_argument("klu_solver::solve: vector size mismatch");
        }

        // KLU solves in-place: copy b into x, then solve
        for (int i = 0; i < n_; ++i)
            x(i) = b(i);

        klu_common common_copy = common_;
        int ok = klu_solve(symbolic_, numeric_, n_, 1, x.data(), &common_copy);
        if (!ok) {
            throw std::runtime_error("KLU solve failed");
        }
    }

    std::size_t num_rows() const { return static_cast<std::size_t>(n_); }
    std::size_t num_cols() const { return static_cast<std::size_t>(n_); }

private:
    int n_;
    klu_common common_;
    std::vector<int> Cp_, Ci_;
    std::vector<double> Cx_;
    klu_symbolic* symbolic_;
    klu_numeric* numeric_;

    template <typename Parameters>
    void convert_to_ccs(const mat::compressed2D<double, Parameters>& A) {
        std::size_t nnz = A.nnz();
        std::size_t ncols = A.num_cols();
        std::size_t nrows = A.num_rows();
        const auto& row_ptr = A.ref_major();
        const auto& col_idx = A.ref_minor();
        const auto& data = A.ref_data();

        Cp_.assign(ncols + 1, 0);
        for (std::size_t k = 0; k < nnz; ++k)
            Cp_[col_idx[k] + 1]++;
        for (std::size_t j = 1; j <= ncols; ++j)
            Cp_[j] += Cp_[j - 1];

        Ci_.resize(nnz);
        Cx_.resize(nnz);
        std::vector<int> pos(Cp_.begin(), Cp_.end());
        for (std::size_t r = 0; r < nrows; ++r) {
            for (std::size_t k = row_ptr[r]; k < row_ptr[r + 1]; ++k) {
                int c = static_cast<int>(col_idx[k]);
                int dest = pos[c]++;
                Ci_[dest] = static_cast<int>(r);
                Cx_[dest] = data[k];
            }
        }
    }
};

/// Convenience: factor and solve A*x = b in one call.
template <typename Parameters, typename ParamsX, typename ParamsB>
void klu_solve(const mat::compressed2D<double, Parameters>& A,
               vec::dense_vector<double, ParamsX>& x,
               const vec::dense_vector<double, ParamsB>& b) {
    klu_solver solver(A);
    solver.solve(x, b);
}

} // namespace mtl::interface

#endif // MTL5_HAS_KLU
