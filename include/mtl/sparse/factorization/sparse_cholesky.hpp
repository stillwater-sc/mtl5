#pragma once
// MTL5 -- Sparse Cholesky factorization (LL^T) for symmetric positive definite matrices
//
// Implements the up-looking (left-looking) sparse Cholesky algorithm from
// Davis's "Direct Methods for Sparse Linear Systems", Chapter 4.
//
// Key features:
//   - Symbolic/numeric phase separation: symbolic analysis can be reused
//     for multiple matrices with the same sparsity pattern
//   - Fill-reducing ordering support via pluggable ordering algorithms
//   - Uses elimination tree for efficient column dependency traversal
//   - O(nnz(L)) arithmetic operations
//
// Algorithm outline (up-looking Cholesky for column j):
//   1. Gather column j of the permuted matrix: c = P*A*P^T(:,j)
//   2. For each column k < j where L(j,k) != 0 (found via etree):
//      c(k:n) -= L(j,k) * L(k:n, k)
//   3. L(j,j) = sqrt(c(j))
//   4. L(j+1:n, j) = c(j+1:n) / L(j,j)
//
// Reference: Davis, "Direct Methods for Sparse Linear Systems", SIAM, 2006.
//            CSparse: cs_schol (symbolic), cs_chol (numeric).

#include <algorithm>
#include <cassert>
#include <cmath>
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

namespace mtl::sparse::factorization {

/// Result of symbolic Cholesky analysis.
/// Captures the sparsity structure of L so it can be reused for multiple
/// matrices with the same pattern.
struct cholesky_symbolic {
    std::vector<std::size_t> parent;      // elimination tree
    std::vector<std::size_t> post;        // postorder of etree
    std::vector<std::size_t> col_counts;  // nnz per column of L (incl. diagonal)
    std::vector<std::size_t> perm;        // fill-reducing permutation (p[new]=old)
    std::vector<std::size_t> pinv;        // inverse permutation (pinv[old]=new)
    std::size_t nnz_L{0};                 // total predicted nnz in L
    std::size_t n{0};                     // matrix dimension
};

/// Result of numeric Cholesky factorization.
/// Contains the lower triangular factor L in CSC format and the
/// symbolic analysis used (for permutation information during solve).
template <typename Value>
struct cholesky_numeric {
    util::csc_matrix<Value> L;            // lower triangular Cholesky factor (CSC)
    cholesky_symbolic symbolic;           // symbolic analysis used

    std::size_t num_rows() const { return symbolic.n; }
    std::size_t num_cols() const { return symbolic.n; }

    /// Solve A*x = b using the Cholesky factorization.
    /// A = P^T L L^T P, so x = P^T L^{-T} L^{-1} P b
    template <typename VecX, typename VecB>
    void solve(VecX& x, const VecB& b) const {
        std::size_t n = symbolic.n;
        if (static_cast<std::size_t>(x.size()) != n ||
            static_cast<std::size_t>(b.size()) != n) {
            throw std::invalid_argument(
                "cholesky_numeric::solve: vector size mismatch (expected "
                + std::to_string(n) + ")");
        }

        // Step 1: Apply permutation: w = P * b
        std::vector<Value> w(n);
        for (std::size_t i = 0; i < n; ++i)
            w[i] = static_cast<Value>(b(symbolic.perm[i]));

        // Step 2: Forward solve: L * y = w
        dense_lower_solve(L, w);

        // Step 3: Back solve: L^T * z = y
        dense_lower_transpose_solve(L, w);

        // Step 4: Apply inverse permutation: x = P^T * z
        for (std::size_t i = 0; i < n; ++i)
            x(symbolic.perm[i]) = static_cast<typename VecX::value_type>(w[i]);
    }
};

/// Perform symbolic Cholesky analysis on a symmetric sparse matrix.
///
/// Computes the elimination tree, fill-reducing ordering, and predicted
/// sparsity structure of L. This result can be reused for multiple matrices
/// with the same nonzero pattern.
///
/// \param A       Symmetric sparse matrix in CRS format
/// \param ordering  Fill-reducing ordering functor (e.g., ordering::rcm{})
/// \return        Symbolic analysis result
template <typename Value, typename Parameters, typename Ordering>
cholesky_symbolic sparse_cholesky_symbolic(
    const mat::compressed2D<Value, Parameters>& A,
    const Ordering& ordering)
{
    if (A.num_rows() != A.num_cols()) {
        throw std::invalid_argument(
            "sparse_cholesky_symbolic: matrix must be square");
    }
    std::size_t n = A.num_rows();

    cholesky_symbolic sym;
    sym.n = n;

    // Step 1: Compute fill-reducing ordering and validate
    sym.perm = ordering(A);
    if (!util::is_valid_permutation(sym.perm) || sym.perm.size() != n) {
        throw std::invalid_argument(
            "sparse_cholesky_symbolic: ordering returned invalid permutation");
    }
    sym.pinv = util::invert_permutation(sym.perm);

    // Step 2: Apply symmetric permutation and convert to CSC
    auto PA = util::symmetric_permute(A, sym.perm);
    auto C = util::crs_to_csc(PA);

    // Step 3: Compute elimination tree of permuted matrix
    sym.parent = analysis::elimination_tree(C);

    // Step 4: Compute postorder
    sym.post = analysis::tree_postorder(sym.parent);

    // Step 5: Compute column counts
    sym.col_counts = analysis::column_counts(C, sym.parent);

    // Step 6: Total nnz
    sym.nnz_L = analysis::total_nnz(sym.col_counts);

    return sym;
}

/// Overload without ordering: uses identity permutation (natural ordering).
template <typename Value, typename Parameters>
cholesky_symbolic sparse_cholesky_symbolic(
    const mat::compressed2D<Value, Parameters>& A)
{
    if (A.num_rows() != A.num_cols()) {
        throw std::invalid_argument(
            "sparse_cholesky_symbolic: matrix must be square");
    }
    std::size_t n = A.num_rows();

    cholesky_symbolic sym;
    sym.n = n;
    sym.perm = util::identity_permutation(n);
    sym.pinv = sym.perm;

    auto C = util::crs_to_csc(A);
    sym.parent = analysis::elimination_tree(C);
    sym.post = analysis::tree_postorder(sym.parent);
    sym.col_counts = analysis::column_counts(C, sym.parent);
    sym.nnz_L = analysis::total_nnz(sym.col_counts);

    return sym;
}

/// Perform numeric Cholesky factorization using pre-computed symbolic analysis.
///
/// This is the up-looking (left-looking) Cholesky algorithm:
/// For each column j, compute L(:,j) by subtracting contributions from
/// all previously factored columns k where L(j,k) != 0.
///
/// \param A    Symmetric positive definite sparse matrix in CRS format
/// \param sym  Symbolic analysis from sparse_cholesky_symbolic()
/// \return     Numeric factorization result containing L in CSC
///
/// \throws std::runtime_error if the matrix is not positive definite
///         (diagonal entry becomes non-positive during factorization)
template <typename Value, typename Parameters>
cholesky_numeric<Value> sparse_cholesky_numeric(
    const mat::compressed2D<Value, Parameters>& A,
    const cholesky_symbolic& sym)
{
    using size_type = std::size_t;
    size_type n = sym.n;
    if (A.num_rows() != n || A.num_cols() != n) {
        throw std::invalid_argument(
            "sparse_cholesky_numeric: matrix dimensions ("
            + std::to_string(A.num_rows()) + "x" + std::to_string(A.num_cols())
            + ") do not match symbolic analysis (n=" + std::to_string(n) + ")");
    }

    // Apply symmetric permutation and convert to CSC
    auto PA = util::symmetric_permute(A, sym.perm);
    auto C = util::crs_to_csc(PA);

    // Allocate L in CSC format using predicted column counts
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

    // Working arrays
    std::vector<Value> x(n, Value{0});    // dense workspace for column assembly
    std::vector<size_type> nz(n, 0);      // next free slot in each column of L

    // Up-looking Cholesky: process columns in order 0..n-1
    for (size_type j = 0; j < n; ++j) {
        // Scatter column j of C into dense workspace x
        // Only the lower triangular part (rows >= j)
        for (size_type p = C.col_ptr[j]; p < C.col_ptr[j + 1]; ++p) {
            size_type i = C.row_ind[p];
            if (i >= j) {
                x[i] = C.values[p];
            }
        }

        // Also scatter upper triangle entries as lower triangle
        // (for symmetric matrix, A(i,j) with i < j means L column j
        //  gets contribution at position j from column i)
        // The diagonal is handled by the i >= j case above.

        // Walk up the elimination tree from j, subtracting contributions
        // from ancestor columns. For each column k that is an ancestor of j
        // in the etree where L(j,k) != 0:
        //   x(k:n) -= L(j,k) * L(k:n, k)
        size_type k = j;
        // Use the etree: walk from j's children upward.
        // Actually, up-looking processes: for each k where L(j,k) != 0,
        // i.e., for each column k < j that has a nonzero in row j.
        // We find these by walking the etree from the nonzero rows of C(:,j)
        // that are < j, up to j.

        // Collect the set of columns k < j that affect column j
        // by walking the etree from each row index i < j in column j of C
        std::vector<size_type> affecting_cols;
        for (size_type p = C.col_ptr[j]; p < C.col_ptr[j + 1]; ++p) {
            size_type i = C.row_ind[p];
            if (i >= j) continue;
            // Walk from i up the etree to j
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

        // For each affecting column k, subtract L(j,k) * L(:,k) from x
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

            // Subtract ljk * L(j:n, col_k) from x(j:n)
            for (size_type p = col_start; p < col_end; ++p) {
                size_type i = L.row_ind[p];
                if (i >= j) {
                    x[i] -= ljk * L.values[p];
                }
            }
        }

        // Compute L(j,j) = sqrt(x[j])
        Value diag = x[j];
        if (diag <= Value{0}) {
            throw std::runtime_error(
                "sparse_cholesky_numeric: matrix is not positive definite "
                "(non-positive diagonal at column " + std::to_string(j) + ")");
        }
        Value ljj = std::sqrt(diag);

        // Guarded write into column j of L
        size_type col_capacity = sym.col_counts[j];
        auto push_entry = [&](size_type row, Value val) {
            if (nz[j] >= col_capacity) {
                throw std::runtime_error(
                    "sparse_cholesky_numeric: column count underestimated at column "
                    + std::to_string(j));
            }
            size_type pos = L.col_ptr[j] + nz[j];
            L.row_ind[pos] = row;
            L.values[pos] = val;
            ++nz[j];
        };

        // Store diagonal
        push_entry(j, ljj);

        // Store off-diagonal entries L(i,j) = x[i] / L(j,j) for i > j
        for (size_type p = C.col_ptr[j]; p < C.col_ptr[j + 1]; ++p) {
            size_type i = C.row_ind[p];
            if (i > j && x[i] != Value{0}) {
                push_entry(i, x[i] / ljj);
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
                        push_entry(i, x[i] / ljj);
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

        // Pair-sort row_ind and values together
        // Simple insertion sort (columns are typically small)
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

    // Trim L to actual nnz (may be less than predicted if column_counts overestimated)
    size_type actual_nnz = 0;
    for (size_type j = 0; j < n; ++j)
        actual_nnz += nz[j];

    if (actual_nnz < nnz_L) {
        // Compact: shift entries to remove gaps
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

    cholesky_numeric<Value> result;
    result.L = std::move(L);
    result.symbolic = sym;
    return result;
}

/// One-shot sparse Cholesky solve: factor and solve A*x = b.
///
/// \param A   Symmetric positive definite sparse matrix
/// \param x   Solution vector (output)
/// \param b   Right-hand side vector
/// \param ordering  Fill-reducing ordering functor
template <typename Value, typename Parameters, typename VecX, typename VecB,
          typename Ordering>
void sparse_cholesky_solve(
    const mat::compressed2D<Value, Parameters>& A,
    VecX& x, const VecB& b,
    const Ordering& ordering)
{
    auto sym = sparse_cholesky_symbolic(A, ordering);
    auto num = sparse_cholesky_numeric(A, sym);
    num.solve(x, b);
}

/// One-shot sparse Cholesky solve without ordering (natural ordering).
template <typename Value, typename Parameters, typename VecX, typename VecB>
void sparse_cholesky_solve(
    const mat::compressed2D<Value, Parameters>& A,
    VecX& x, const VecB& b)
{
    auto sym = sparse_cholesky_symbolic(A);
    auto num = sparse_cholesky_numeric(A, sym);
    num.solve(x, b);
}

} // namespace mtl::sparse::factorization
