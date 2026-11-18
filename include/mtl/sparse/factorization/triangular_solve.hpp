#pragma once
// MTL5 -- Sparse triangular solve using the reach algorithm
// This is the core routine of Gilbert-Peierls: solve Lx = b where both
// L and b are sparse, in time proportional to nnz(x) rather than nnz(L).
//
// The key insight is the "reach" computation: a topological ordering of
// the nodes in L that are reachable from the nonzero pattern of b.
// Only those columns of L participate in the solve.
//
// Reference: Gilbert & Peierls, "Sparse Partial Pivoting in Time
//            Proportional to Arithmetic Operations", SIAM J. Sci. Stat.
//            Comput., 9(5), 1988.
// Also: Davis, "Direct Methods for Sparse Linear Systems", Ch. 3 & 6.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <mtl/sparse/util/csc.hpp>

namespace mtl::sparse::factorization {

/// Sentinel for unvisited nodes in DFS
inline constexpr std::size_t unvisited = std::numeric_limits<std::size_t>::max();

/// Compute the topological ordering of nodes reachable from the nonzero
/// pattern of b in the graph of L (given in CSC).
///
/// This is the "reach" of b in L: the set of columns of L that will have
/// nonzeros when solving Lx = b, ordered so that dependencies come first.
///
/// \param L_col_ptr  Column pointers of L (CSC)
/// \param L_row_ind  Row indices of L (CSC)
/// \param n          Dimension of L
/// \param b_indices  Nonzero row indices of the right-hand side b
/// \param xi         Output: topological order (filled from the back)
/// \param stack_workspace  length n: the DFS stack of nodes
/// \param pos_workspace    length n: for each stack level, the next entry of
///                         that node's column to scan
/// \param marked     length n, caller manages
/// \return           Starting index in xi (the reach is xi[top..n-1])
///
/// The algorithm uses iterative DFS with an explicit stack. The output
/// xi is filled from the end backward, so xi[return_value..n-1] gives
/// the topological order. Each node's column is scanned once in all: a node
/// resumes its scan where it left off when a child returns (as CSparse's
/// cs_dfs), where a rescan from the start made a node with d children cost
/// O(d^2) (issue 907).
inline std::size_t reach(
    const std::vector<std::size_t>& L_col_ptr,
    const std::vector<std::size_t>& L_row_ind,
    std::size_t n,
    const std::vector<std::size_t>& b_indices,
    std::vector<std::size_t>& xi,              // output, length n
    std::vector<std::size_t>& stack_workspace, // length n
    std::vector<std::size_t>& pos_workspace,   // length n
    std::vector<bool>& marked)                 // length n, caller manages
{
    // xi is filled from position n backward
    std::size_t top = n;

    for (std::size_t idx = 0; idx < b_indices.size(); ++idx) {
        std::size_t start = b_indices[idx];
        if (marked[start]) continue;

        // DFS from start
        std::size_t stack_top = 0;
        stack_workspace[0] = start;
        pos_workspace[0] = L_col_ptr[start];

        while (true) {
            const std::size_t node = stack_workspace[stack_top];

            // The next unmarked child in node's column, from where the scan
            // stopped last time.
            bool found_child = false;
            std::size_t p = pos_workspace[stack_top];
            for (const std::size_t pend = L_col_ptr[node + 1]; p < pend; ++p) {
                const std::size_t child = L_row_ind[p];
                if (child <= node) continue;  // only look at entries below diagonal
                if (!marked[child]) {
                    pos_workspace[stack_top] = p + 1;
                    ++stack_top;
                    stack_workspace[stack_top] = child;
                    pos_workspace[stack_top] = L_col_ptr[child];
                    found_child = true;
                    break;
                }
            }

            if (!found_child) {
                // All children visited: postorder emit
                marked[node] = true;
                xi[--top] = node;
                if (stack_top == 0) break;
                --stack_top;
            }
        }
    }

    return top;
}

/// reach() with its own position workspace: allocates O(n) per call. Callers
/// that compute many reaches should pass a pos_workspace of their own.
inline std::size_t reach(
    const std::vector<std::size_t>& L_col_ptr,
    const std::vector<std::size_t>& L_row_ind,
    std::size_t n,
    const std::vector<std::size_t>& b_indices,
    std::vector<std::size_t>& xi,              // output, length n
    std::vector<std::size_t>& stack_workspace, // length n
    std::vector<bool>& marked)                 // length n, caller manages
{
    std::vector<std::size_t> pos(n);
    return reach(L_col_ptr, L_row_ind, n, b_indices, xi, stack_workspace, pos, marked);
}

namespace detail {

/// The diagonal of column `col` is not where the solve expects it (first in a
/// column of L, last in a column of U): a wrong layout, or no diagonal stored.
template <typename SizeType>
[[noreturn]] inline void misplaced_diagonal(const char* fn, SizeType col) {
    throw std::invalid_argument(std::string(fn) + ": column " + std::to_string(col) +
                                " has no diagonal entry where the solve expects it");
}

/// A zero diagonal: the triangular matrix is singular.
template <typename SizeType>
[[noreturn]] inline void zero_diagonal(const char* fn, SizeType col) {
    throw std::runtime_error(std::string(fn) + ": zero diagonal in column " + std::to_string(col) +
                             " (singular triangular factor)");
}

} // namespace detail

/// Sparse triangular solve: solve Lx = b where L is lower triangular in CSC
/// and b is sparse.
///
/// \param L       Lower triangular matrix in CSC format
/// \param b       Dense right-hand side vector (overwritten with solution x)
/// \param xi      Topological order from reach() - xi[top..n-1]
/// \param top     Starting index in xi (from reach())
///
/// On entry, b contains the right-hand side. On exit, b contains the solution.
/// Only the entries in the reach are touched.
template <typename Value, typename SizeType>
void sparse_lower_solve(
    const util::csc_matrix<Value, SizeType>& L,
    std::vector<Value>& x,
    const std::vector<std::size_t>& xi,
    std::size_t top)
{
    SizeType n = L.ncols;

    // Process columns in topological order
    for (std::size_t idx = top; idx < n; ++idx) {
        std::size_t j = xi[idx];
        assert(j < n);

        // Find diagonal entry (first entry in column j should be row j)
        SizeType diag_pos = L.col_ptr[j];
        // Checked in every build (issue 907): an assert left a missing or zero
        // diagonal to read past the column or divide by zero under NDEBUG.
        if (diag_pos >= L.col_ptr[j + 1] || L.row_ind[diag_pos] != j)
            detail::misplaced_diagonal("sparse_lower_solve", j);

        Value diag = L.values[diag_pos];
        if (diag == Value{0}) detail::zero_diagonal("sparse_lower_solve", j);

        x[j] /= diag;

        // Scatter: for each off-diagonal entry in column j
        for (SizeType p = diag_pos + 1; p < L.col_ptr[j + 1]; ++p) {
            SizeType i = L.row_ind[p];
            x[i] -= L.values[p] * x[j];
        }
    }
}

/// Dense lower triangular solve: solve Lx = b where L is in CSC format.
/// b is overwritten with the solution x.
/// This processes all columns (no reach computation needed for dense b).
template <typename Value, typename SizeType>
void dense_lower_solve(
    const util::csc_matrix<Value, SizeType>& L,
    std::vector<Value>& x)
{
    SizeType n = L.ncols;

    for (SizeType j = 0; j < n; ++j) {
        SizeType diag_pos = L.col_ptr[j];
        // Checked in every build (issue 907), not asserted. An empty column
        // has no diagonal: skipping it solved the system as if it were 1.
        if (diag_pos >= L.col_ptr[j + 1] || L.row_ind[diag_pos] != j)
            detail::misplaced_diagonal("dense_lower_solve", j);
        Value diag = L.values[diag_pos];
        if (diag == Value{0}) detail::zero_diagonal("dense_lower_solve", j);

        x[j] /= diag;

        for (SizeType p = diag_pos + 1; p < L.col_ptr[j + 1]; ++p) {
            SizeType i = L.row_ind[p];
            x[i] -= L.values[p] * x[j];
        }
    }
}

/// Dense upper triangular solve: solve Ux = b where U is in CSC format.
/// b is overwritten with the solution x.
template <typename Value, typename SizeType>
void dense_upper_solve(
    const util::csc_matrix<Value, SizeType>& U,
    std::vector<Value>& x)
{
    SizeType n = U.ncols;

    for (SizeType j = n; j > 0; --j) {
        SizeType col = j - 1;
        // An empty column has no diagonal (issue 907). It is tested before the
        // diagonal's position is formed: with col_ptr[col + 1] == 0,
        // col_ptr[col + 1] - 1 wrapped to SIZE_MAX, passed the old test and
        // read row_ind[SIZE_MAX]; and skipping it solved the system as if the
        // diagonal were 1.
        if (U.col_ptr[col] >= U.col_ptr[col + 1]) detail::misplaced_diagonal("dense_upper_solve", col);
        // Find diagonal (last entry in column col for upper triangular)
        SizeType diag_pos = U.col_ptr[col + 1] - 1;

        // Checked in every build (issue 907), not asserted.
        if (U.row_ind[diag_pos] != col) detail::misplaced_diagonal("dense_upper_solve", col);
        Value diag = U.values[diag_pos];
        if (diag == Value{0}) detail::zero_diagonal("dense_upper_solve", col);

        x[col] /= diag;

        // Scatter: off-diagonal entries are above the diagonal
        for (SizeType p = U.col_ptr[col]; p < diag_pos; ++p) {
            SizeType i = U.row_ind[p];
            x[i] -= U.values[p] * x[col];
        }
    }
}

/// Dense lower triangular transpose solve: solve L^T x = b where L is in CSC.
/// Processes columns in reverse order.
template <typename Value, typename SizeType>
void dense_lower_transpose_solve(
    const util::csc_matrix<Value, SizeType>& L,
    std::vector<Value>& x)
{
    SizeType n = L.ncols;

    for (SizeType j = n; j > 0; --j) {
        SizeType col = j - 1;
        SizeType diag_pos = L.col_ptr[col];
        // Checked in every build (issue 907), not asserted; an empty column
        // has no diagonal.
        if (diag_pos >= L.col_ptr[col + 1] || L.row_ind[diag_pos] != col)
            detail::misplaced_diagonal("dense_lower_transpose_solve", col);
        if (L.values[diag_pos] == Value{0}) detail::zero_diagonal("dense_lower_transpose_solve", col);

        // Gather: subtract contributions from off-diagonal entries
        for (SizeType p = diag_pos + 1; p < L.col_ptr[col + 1]; ++p) {
            SizeType i = L.row_ind[p];
            x[col] -= L.values[p] * x[i];
        }

        x[col] /= L.values[diag_pos];
    }
}

/// Dense unit lower triangular solve: solve Lx = b where L is unit lower
/// triangular in CSC format (diagonal entries in L are ignored; assumed 1).
/// b is overwritten with the solution x.
template <typename Value, typename SizeType>
void dense_unit_lower_solve(
    const util::csc_matrix<Value, SizeType>& L,
    std::vector<Value>& x)
{
    SizeType n = L.ncols;

    for (SizeType j = 0; j < n; ++j) {
        // Skip diagonal entry (implicit 1), scatter off-diagonal entries
        for (SizeType p = L.col_ptr[j]; p < L.col_ptr[j + 1]; ++p) {
            SizeType i = L.row_ind[p];
            if (i > j) {
                x[i] -= L.values[p] * x[j];
            }
        }
    }
}

/// Dense unit lower triangular transpose solve: solve L^T x = b where L is
/// unit lower triangular in CSC (diagonal entries ignored; assumed 1).
/// Processes columns in reverse order. b is overwritten with x.
template <typename Value, typename SizeType>
void dense_unit_lower_transpose_solve(
    const util::csc_matrix<Value, SizeType>& L,
    std::vector<Value>& x)
{
    SizeType n = L.ncols;

    for (SizeType j = n; j > 0; --j) {
        SizeType col = j - 1;
        // Gather: subtract contributions from off-diagonal entries below diagonal
        for (SizeType p = L.col_ptr[col]; p < L.col_ptr[col + 1]; ++p) {
            SizeType i = L.row_ind[p];
            if (i > col) {
                x[col] -= L.values[p] * x[i];
            }
        }
        // No diagonal division (unit diagonal)
    }
}

} // namespace mtl::sparse::factorization
