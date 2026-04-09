#pragma once
// MTL5 -- Sparse LDL^T factorization (square-root-free Cholesky)
// for symmetric matrices (positive definite or indefinite).
//
// A = L * D * L^T where L is unit lower triangular, D is diagonal.
//
// The symbolic phase is identical to Cholesky (same elimination tree,
// same fill-in pattern), so we reuse cholesky_symbolic directly.
//
// The numeric phase is the up-looking LDL^T algorithm: same structure
// as up-looking Cholesky but stores D separately and avoids sqrt.
//
// Key advantages over LL^T:
//   - No square roots — avoids precision loss for ill-conditioned matrices
//   - Works for symmetric indefinite matrices (D can have negative entries)
//   - Only fails on zero pivots (D(j) == 0)
//
// Reference: Davis, "Direct Methods for Sparse Linear Systems", SIAM, 2006.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/sparse/util/csc.hpp>
#include <mtl/sparse/util/permutation.hpp>
#include <mtl/sparse/analysis/elimination_tree.hpp>
#include <mtl/sparse/analysis/postorder.hpp>
#include <mtl/sparse/factorization/triangular_solve.hpp>
#include <mtl/sparse/factorization/sparse_cholesky.hpp>  // for cholesky_symbolic

namespace mtl::sparse::factorization {

/// Symbolic analysis result — identical to Cholesky since sparsity of L
/// is the same for LDL^T and LL^T.
using ldlt_symbolic = cholesky_symbolic;

/// Result of numeric LDL^T factorization.
/// Contains the unit lower triangular factor L in CSC format,
/// the diagonal D, and the symbolic analysis (for permutation during solve).
template <typename Value>
struct ldlt_numeric {
    util::csc_matrix<Value> L;            // unit lower triangular (CSC)
    std::vector<Value>      D;            // diagonal entries
    ldlt_symbolic           symbolic;     // symbolic analysis used

    std::size_t num_rows() const { return symbolic.n; }
    std::size_t num_cols() const { return symbolic.n; }

    /// Solve A*x = b using the LDL^T factorization.
    /// A = P^T L D L^T P, so x = P^T L^{-T} D^{-1} L^{-1} P b
    template <typename VecX, typename VecB>
    void solve(VecX& x, const VecB& b) const {
        std::size_t n = symbolic.n;
        if (static_cast<std::size_t>(x.size()) != n ||
            static_cast<std::size_t>(b.size()) != n) {
            throw std::invalid_argument(
                "ldlt_numeric::solve: vector size mismatch (expected "
                + std::to_string(n) + ")");
        }

        // Step 1: Apply permutation: w = P * b
        std::vector<Value> w(n);
        for (std::size_t i = 0; i < n; ++i)
            w[i] = static_cast<Value>(b(symbolic.perm[i]));

        // Step 2: Forward solve: L * y = w  (L is unit lower triangular)
        dense_unit_lower_solve(L, w);

        // Step 3: Diagonal solve: D * z = y
        for (std::size_t i = 0; i < n; ++i)
            w[i] /= D[i];

        // Step 4: Back solve: L^T * u = z  (L^T is unit upper triangular)
        dense_unit_lower_transpose_solve(L, w);

        // Step 5: Apply inverse permutation: x = P^T * u
        for (std::size_t i = 0; i < n; ++i)
            x(symbolic.perm[i]) = static_cast<typename VecX::value_type>(w[i]);
    }
};

/// Perform symbolic LDL^T analysis on a symmetric sparse matrix.
/// Delegates to sparse_cholesky_symbolic since the sparsity structure is identical.
template <typename Value, typename Parameters, typename Ordering>
ldlt_symbolic sparse_ldlt_symbolic(
    const mat::compressed2D<Value, Parameters>& A,
    const Ordering& ordering)
{
    return sparse_cholesky_symbolic(A, ordering);
}

/// Overload without ordering: uses identity permutation (natural ordering).
template <typename Value, typename Parameters>
ldlt_symbolic sparse_ldlt_symbolic(
    const mat::compressed2D<Value, Parameters>& A)
{
    return sparse_cholesky_symbolic(A);
}

/// Perform numeric LDL^T factorization using pre-computed symbolic analysis.
///
/// Up-looking (left-looking) LDL^T: for each column j, compute L(:,j)
/// and D(j) by subtracting contributions from previously factored columns.
/// Identical to Cholesky but replaces L(j,j) = sqrt(c(j)) with D(j) = c(j)
/// and L(i,j) = c(i)/D(j).
///
/// \param A    Symmetric sparse matrix in CRS format
/// \param sym  Symbolic analysis from sparse_ldlt_symbolic()
/// \return     Numeric factorization result containing L (unit lower) and D
///
/// \throws std::runtime_error if a zero pivot is encountered (D(j) == 0)
template <typename Value, typename Parameters>
ldlt_numeric<Value> sparse_ldlt_numeric(
    const mat::compressed2D<Value, Parameters>& A,
    const ldlt_symbolic& sym)
{
    using size_type = std::size_t;
    size_type n = sym.n;
    if (A.num_rows() != n || A.num_cols() != n) {
        throw std::invalid_argument(
            "sparse_ldlt_numeric: matrix dimensions ("
            + std::to_string(A.num_rows()) + "x" + std::to_string(A.num_cols())
            + ") do not match symbolic analysis (n=" + std::to_string(n) + ")");
    }

    // Apply symmetric permutation and convert to CSC
    auto PA = util::symmetric_permute(A, sym.perm);
    auto C = util::crs_to_csc(PA);

    // Allocate L in CSC format using predicted column counts.
    // For LDL^T, L is unit lower triangular so we don't store the diagonal in L.
    // However, col_counts from the symbolic phase include the diagonal.
    // We allocate with col_counts (which includes diagonal) and store a 1.0
    // placeholder on the diagonal to keep the CSC structure consistent with
    // the existing triangular solve infrastructure.
    util::csc_matrix<Value> L;
    L.nrows = n;
    L.ncols = n;
    L.col_ptr.resize(n + 1);
    L.col_ptr[0] = 0;
    for (size_type j = 0; j < n; ++j)
        L.col_ptr[j + 1] = L.col_ptr[j] + sym.col_counts[j];

    size_type nnz_L = L.col_ptr[n];
    L.row_ind.resize(nnz_L);
    L.values.resize(nnz_L);

    // Diagonal vector
    std::vector<Value> D(n, Value{0});

    // Working arrays
    std::vector<Value> x(n, Value{0});    // dense workspace for column assembly
    std::vector<size_type> nz(n, 0);      // next free slot in each column of L

    // Up-looking LDL^T: process columns in order 0..n-1
    for (size_type j = 0; j < n; ++j) {
        // Scatter column j of C into dense workspace x
        // Only the lower triangular part (rows >= j)
        for (size_type p = C.col_ptr[j]; p < C.col_ptr[j + 1]; ++p) {
            size_type i = C.row_ind[p];
            if (i >= j) {
                x[i] = C.values[p];
            }
        }

        // Collect the set of columns k < j that affect column j
        // by walking the etree from each row index i < j in column j of C
        std::vector<size_type> affecting_cols;
        for (size_type p = C.col_ptr[j]; p < C.col_ptr[j + 1]; ++p) {
            size_type i = C.row_ind[p];
            if (i >= j) continue;
            size_type node = i;
            while (node != analysis::no_parent && node < j) {
                affecting_cols.push_back(node);
                node = sym.parent[node];
            }
        }

        // Remove duplicates and sort
        std::sort(affecting_cols.begin(), affecting_cols.end());
        affecting_cols.erase(
            std::unique(affecting_cols.begin(), affecting_cols.end()),
            affecting_cols.end());

        // For each affecting column k, subtract L(j,k) * D(k) * L(:,k) from x
        for (size_type col_k : affecting_cols) {
            // Find L(j, col_k): search column col_k of L for row j
            Value ljk = Value{0};
            size_type col_start = L.col_ptr[col_k];
            size_type col_end = L.col_ptr[col_k] + nz[col_k];
            for (size_type p = col_start; p < col_end; ++p) {
                if (L.row_ind[p] == j) {
                    ljk = L.values[p];
                    break;
                }
            }

            if (ljk == Value{0}) continue;

            Value ljk_dk = ljk * D[col_k];

            // Subtract ljk * D(k) * L(i, col_k) from x(i) for i >= j
            // For row j: x[j] -= ljk * D(k) * ljk = ljk^2 * D(k)
            x[j] -= ljk_dk * ljk;

            // For rows i > j: x[i] -= ljk * D(k) * L(i, col_k)
            for (size_type p = col_start; p < col_end; ++p) {
                size_type i = L.row_ind[p];
                if (i > j) {
                    x[i] -= ljk_dk * L.values[p];
                }
            }
        }

        // D(j) = x[j] (the accumulated diagonal value)
        Value dj = x[j];
        if (dj == Value{0}) {
            throw std::runtime_error(
                "sparse_ldlt_numeric: zero pivot at column " + std::to_string(j));
        }
        D[j] = dj;

        // Guarded write into column j of L
        size_type col_capacity = sym.col_counts[j];
        auto push_entry = [&](size_type row, Value val) {
            if (nz[j] >= col_capacity) {
                throw std::runtime_error(
                    "sparse_ldlt_numeric: column count underestimated at column "
                    + std::to_string(j));
            }
            size_type pos = L.col_ptr[j] + nz[j];
            L.row_ind[pos] = row;
            L.values[pos] = val;
            ++nz[j];
        };

        // Store diagonal as 1.0 (unit lower triangular)
        push_entry(j, Value{1});

        // Store off-diagonal entries L(i,j) = x[i] / D(j) for i > j
        for (size_type p = C.col_ptr[j]; p < C.col_ptr[j + 1]; ++p) {
            size_type i = C.row_ind[p];
            if (i > j && x[i] != Value{0}) {
                push_entry(i, x[i] / dj);
            }
        }

        // Also store fill-in entries: rows where x[i] != 0 but not in C(:,j)
        for (size_type col_k : affecting_cols) {
            size_type col_start = L.col_ptr[col_k];
            size_type col_end = L.col_ptr[col_k] + nz[col_k];
            for (size_type p = col_start; p < col_end; ++p) {
                size_type i = L.row_ind[p];
                if (i > j && x[i] != Value{0}) {
                    bool already = false;
                    for (size_type q = L.col_ptr[j]; q < L.col_ptr[j] + nz[j]; ++q) {
                        if (L.row_ind[q] == i) { already = true; break; }
                    }
                    if (!already) {
                        push_entry(i, x[i] / dj);
                    }
                }
            }
        }

        // Clear workspace for rows we touched
        for (size_type p = C.col_ptr[j]; p < C.col_ptr[j + 1]; ++p)
            x[C.row_ind[p]] = Value{0};
        for (size_type col_k : affecting_cols) {
            size_type col_start = L.col_ptr[col_k];
            size_type col_end = L.col_ptr[col_k] + nz[col_k];
            for (size_type p = col_start; p < col_end; ++p)
                x[L.row_ind[p]] = Value{0};
        }
        x[j] = Value{0};

        // Sort row indices within this column for consistent ordering
        size_type col_begin = L.col_ptr[j];
        size_type col_actual_end = L.col_ptr[j] + nz[j];

        // Insertion sort (columns are typically small)
        for (size_type a = col_begin + 1; a < col_actual_end; ++a) {
            size_type key_idx = L.row_ind[a];
            Value key_val = L.values[a];
            size_type b = a;
            while (b > col_begin && L.row_ind[b - 1] > key_idx) {
                L.row_ind[b] = L.row_ind[b - 1];
                L.values[b] = L.values[b - 1];
                --b;
            }
            L.row_ind[b] = key_idx;
            L.values[b] = key_val;
        }
    }

    // Trim L to actual nnz
    size_type actual_nnz = 0;
    for (size_type j = 0; j < n; ++j)
        actual_nnz += nz[j];

    if (actual_nnz < nnz_L) {
        util::csc_matrix<Value> L_compact;
        L_compact.nrows = n;
        L_compact.ncols = n;
        L_compact.col_ptr.resize(n + 1);
        L_compact.row_ind.resize(actual_nnz);
        L_compact.values.resize(actual_nnz);

        L_compact.col_ptr[0] = 0;
        size_type pos = 0;
        for (size_type j = 0; j < n; ++j) {
            for (size_type k = 0; k < nz[j]; ++k) {
                L_compact.row_ind[pos] = L.row_ind[L.col_ptr[j] + k];
                L_compact.values[pos] = L.values[L.col_ptr[j] + k];
                ++pos;
            }
            L_compact.col_ptr[j + 1] = pos;
        }
        L = std::move(L_compact);
    }

    ldlt_numeric<Value> result;
    result.L = std::move(L);
    result.D = std::move(D);
    result.symbolic = sym;
    return result;
}

/// One-shot sparse LDL^T solve: factor and solve A*x = b.
template <typename Value, typename Parameters, typename VecX, typename VecB,
          typename Ordering>
void sparse_ldlt_solve(
    const mat::compressed2D<Value, Parameters>& A,
    VecX& x, const VecB& b,
    const Ordering& ordering)
{
    auto sym = sparse_ldlt_symbolic(A, ordering);
    auto num = sparse_ldlt_numeric(A, sym);
    num.solve(x, b);
}

/// One-shot sparse LDL^T solve without ordering (natural ordering).
template <typename Value, typename Parameters, typename VecX, typename VecB>
void sparse_ldlt_solve(
    const mat::compressed2D<Value, Parameters>& A,
    VecX& x, const VecB& b)
{
    auto sym = sparse_ldlt_symbolic(A);
    auto num = sparse_ldlt_numeric(A, sym);
    num.solve(x, b);
}

} // namespace mtl::sparse::factorization
