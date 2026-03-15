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
/// \return           Starting index in xi (the reach is xi[top..n-1])
///
/// The algorithm uses iterative DFS with an explicit stack. The output
/// xi is filled from the end backward, so xi[return_value..n-1] gives
/// the topological order.
inline std::size_t reach(
    const std::vector<std::size_t>& L_col_ptr,
    const std::vector<std::size_t>& L_row_ind,
    std::size_t n,
    const std::vector<std::size_t>& b_indices,
    std::vector<std::size_t>& xi,           // output, length n
    std::vector<std::size_t>& stack_workspace, // length n
    std::vector<bool>& marked)              // length n, caller manages
{
    // xi is filled from position n backward
    std::size_t top = n;

    for (std::size_t idx = 0; idx < b_indices.size(); ++idx) {
        std::size_t start = b_indices[idx];
        if (marked[start]) continue;

        // DFS from start
        std::size_t stack_top = 0;
        stack_workspace[0] = start;

        while (stack_top != unvisited) {
            std::size_t node = stack_workspace[stack_top];

            // Find first unmarked child in L's column
            bool found_child = false;
            // We need to track where we are in node's adjacency list.
            // Use a simple approach: scan from the beginning each time.
            // (CSparse uses a head pointer array for efficiency; we
            //  use a slightly simpler approach here.)
            for (std::size_t p = L_col_ptr[node]; p < L_col_ptr[node + 1]; ++p) {
                std::size_t child = L_row_ind[p];
                if (child <= node) continue;  // only look at entries below diagonal
                if (!marked[child]) {
                    stack_workspace[++stack_top] = child;
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

/// Sparse triangular solve: solve Lx = b where L is lower triangular in CSC
/// and b is sparse.
///
/// \param L       Lower triangular matrix in CSC format
/// \param b       Dense right-hand side vector (overwritten with solution x)
/// \param xi      Topological order from reach() — xi[top..n-1]
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
        assert(diag_pos < L.col_ptr[j + 1]);
        assert(L.row_ind[diag_pos] == j);

        Value diag = L.values[diag_pos];
        assert(diag != Value{0});

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
        if (diag_pos >= L.col_ptr[j + 1]) continue;

        assert(L.row_ind[diag_pos] == j);
        Value diag = L.values[diag_pos];
        assert(diag != Value{0});

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
        // Find diagonal (last entry in column col for upper triangular)
        SizeType diag_pos = U.col_ptr[col + 1] - 1;
        if (U.col_ptr[col] > diag_pos) continue;

        assert(U.row_ind[diag_pos] == col);
        Value diag = U.values[diag_pos];
        assert(diag != Value{0});

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
        if (diag_pos >= L.col_ptr[col + 1]) continue;

        assert(L.row_ind[diag_pos] == col);

        // Gather: subtract contributions from off-diagonal entries
        for (SizeType p = diag_pos + 1; p < L.col_ptr[col + 1]; ++p) {
            SizeType i = L.row_ind[p];
            x[col] -= L.values[p] * x[i];
        }

        x[col] /= L.values[diag_pos];
    }
}

} // namespace mtl::sparse::factorization
