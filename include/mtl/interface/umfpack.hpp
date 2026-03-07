#pragma once
// MTL5 -- Optional UMFPACK interface for sparse direct solves
// Port from MTL4: boost/numeric/mtl/interface/umfpack.hpp
// Guarded by MTL5_HAS_UMFPACK (set by CMake when MTL5_ENABLE_UMFPACK=ON)

#ifdef MTL5_HAS_UMFPACK

#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <vector>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/vec/dense_vector.hpp>

// -- UMFPACK C API declarations ---------------------------------------------
extern "C" {

int umfpack_di_symbolic(int n_row, int n_col,
                        const int* Ap, const int* Ai, const double* Ax,
                        void** Symbolic, const double* Control, double* Info);

int umfpack_di_numeric(const int* Ap, const int* Ai, const double* Ax,
                       void* Symbolic, void** Numeric,
                       const double* Control, double* Info);

int umfpack_di_solve(int sys,
                     const int* Ap, const int* Ai, const double* Ax,
                     double* X, const double* B,
                     void* Numeric, const double* Control, double* Info);

void umfpack_di_free_symbolic(void** Symbolic);
void umfpack_di_free_numeric(void** Numeric);

} // extern "C"

namespace mtl::interface {

/// Convert a CRS (compressed row storage) matrix to CCS (compressed column storage)
/// needed by UMFPACK. Produces column pointers (Cp), row indices (Ci), values (Cx).
inline void crs_to_ccs(std::size_t nrows, std::size_t ncols,
                       const std::vector<std::size_t>& row_ptr,
                       const std::vector<std::size_t>& col_idx,
                       const std::vector<double>& vals,
                       std::vector<int>& Cp,
                       std::vector<int>& Ci,
                       std::vector<double>& Cx) {
    std::size_t nnz = vals.size();

    // Count entries per column
    Cp.assign(ncols + 1, 0);
    for (std::size_t i = 0; i < nnz; ++i)
        Cp[col_idx[i] + 1]++;
    for (std::size_t j = 1; j <= ncols; ++j)
        Cp[j] += Cp[j - 1];

    Ci.resize(nnz);
    Cx.resize(nnz);

    // Fill column-compressed arrays
    std::vector<int> pos(Cp.begin(), Cp.end());
    for (std::size_t r = 0; r < nrows; ++r) {
        for (std::size_t k = row_ptr[r]; k < row_ptr[r + 1]; ++k) {
            int c = static_cast<int>(col_idx[k]);
            int dest = pos[c]++;
            Ci[dest] = static_cast<int>(r);
            Cx[dest] = vals[k];
        }
    }
}

/// RAII wrapper for UMFPACK symbolic + numeric factorization.
/// Usage:
///   umfpack_solver solver(A);
///   solver.solve(x, b);
class umfpack_solver {
public:
    /// Factor the sparse matrix A (compressed2D<double>).
    template <typename Parameters>
    explicit umfpack_solver(const mat::compressed2D<double, Parameters>& A)
        : n_(static_cast<int>(A.num_rows())), symbolic_(nullptr), numeric_(nullptr)
    {
        assert(A.num_rows() == A.num_cols());

        // Convert CRS to CCS for UMFPACK
        crs_to_ccs(A.num_rows(), A.num_cols(),
                    A.ref_major(), A.ref_minor(), A.ref_data(),
                    Cp_, Ci_, Cx_);

        int status = umfpack_di_symbolic(n_, n_, Cp_.data(), Ci_.data(), Cx_.data(),
                                          &symbolic_, nullptr, nullptr);
        if (status != 0)
            throw std::runtime_error("UMFPACK symbolic factorization failed (status "
                                      + std::to_string(status) + ")");

        status = umfpack_di_numeric(Cp_.data(), Ci_.data(), Cx_.data(),
                                     symbolic_, &numeric_, nullptr, nullptr);
        if (status != 0) {
            umfpack_di_free_symbolic(&symbolic_);
            throw std::runtime_error("UMFPACK numeric factorization failed (status "
                                      + std::to_string(status) + ")");
        }
    }

    ~umfpack_solver() {
        if (numeric_)  umfpack_di_free_numeric(&numeric_);
        if (symbolic_) umfpack_di_free_symbolic(&symbolic_);
    }

    // Non-copyable
    umfpack_solver(const umfpack_solver&) = delete;
    umfpack_solver& operator=(const umfpack_solver&) = delete;

    // Movable
    umfpack_solver(umfpack_solver&& o) noexcept
        : n_(o.n_), Cp_(std::move(o.Cp_)), Ci_(std::move(o.Ci_)), Cx_(std::move(o.Cx_)),
          symbolic_(o.symbolic_), numeric_(o.numeric_) {
        o.symbolic_ = nullptr;
        o.numeric_ = nullptr;
    }

    /// Solve A*x = b using the precomputed factorization.
    template <typename ParamsX, typename ParamsB>
    void solve(vec::dense_vector<double, ParamsX>& x,
               const vec::dense_vector<double, ParamsB>& b) const {
        assert(static_cast<int>(x.size()) == n_);
        assert(static_cast<int>(b.size()) == n_);
        // UMFPACK_A = 0: solve Ax = b
        int status = umfpack_di_solve(0, Cp_.data(), Ci_.data(), Cx_.data(),
                                       x.data(), b.data(), numeric_, nullptr, nullptr);
        if (status != 0)
            throw std::runtime_error("UMFPACK solve failed (status "
                                      + std::to_string(status) + ")");
    }

private:
    int n_;
    std::vector<int> Cp_, Ci_;
    std::vector<double> Cx_;
    void* symbolic_;
    void* numeric_;
};

/// Convenience: factor and solve A*x = b in one call.
template <typename Parameters, typename ParamsX, typename ParamsB>
void umfpack_solve(const mat::compressed2D<double, Parameters>& A,
                   vec::dense_vector<double, ParamsX>& x,
                   const vec::dense_vector<double, ParamsB>& b) {
    umfpack_solver solver(A);
    solver.solve(x, b);
}

} // namespace mtl::interface

#endif // MTL5_HAS_UMFPACK
