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
//   - No square roots - avoids precision loss for ill-conditioned matrices
//   - Works for symmetric indefinite matrices (D can have negative entries)
//   - Only fails on zero pivots (D(j) == 0)
//
// Reference: Davis, "Direct Methods for Sparse Linear Systems", SIAM, 2006.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <limits>
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
#include <mtl/sparse/factorization/level_schedule.hpp>
#include <mtl/sparse/factorization/sparse_cholesky.hpp>  // for cholesky_symbolic

namespace mtl::sparse::factorization {

/// Symbolic analysis result - identical to Cholesky since sparsity of L
/// is the same for LDL^T and LL^T.
using ldlt_symbolic = cholesky_symbolic;

/// Result of numeric LDL^T factorization.
/// Contains the unit lower triangular factor L in CSC format,
/// the diagonal D, and the symbolic analysis (for permutation during solve).
template <typename Value>
struct ldlt_numeric {
    ldlt_symbolic symbolic;               // symbolic analysis used

    std::size_t num_rows() const { return symbolic.n; }
    std::size_t num_cols() const { return symbolic.n; }

    /// Read-only access to the unit lower factor L (CSC) and the diagonal D.
    const util::csc_matrix<Value>& factorL() const { return L_; }
    const std::vector<Value>&      diagonal() const { return D_; }

    /// Install the factor (unit lower L, diagonal D) and (re)build the coupled
    /// unit forward and transpose solve schedules from L. L, D and the schedules
    /// are always set together, so the schedules' cached structure always matches
    /// L's pattern. Strongly exception safe: the schedules are built into locals
    /// before any member is replaced.
    ///
    /// Precondition: the symbolic analysis (`symbolic`) MUST be installed first --
    /// set_factor validates L and D against `symbolic.n`. Calling it while
    /// `symbolic` is still default-constructed (n == 0) with a non-empty factor
    /// throws (see the guard below), rather than silently accepting a factor that
    /// no later solve could use.
    void set_factor(util::csc_matrix<Value> L, std::vector<Value> D) {
        const std::size_t n = symbolic.n;
        // Guard the ordering hazard explicitly: n == 0 with a non-empty factor
        // almost always means set_factor was called before `symbolic` was
        // assigned, so point at that cause instead of the generic mismatch below.
        if (n == 0 && (L.ncols != 0 || L.nrows != 0 || !D.empty()))
            throw std::invalid_argument(
                "ldlt_numeric::set_factor: symbolic analysis not installed (symbolic.n == 0); "
                "assign `symbolic` before calling set_factor");
        if (static_cast<std::size_t>(L.ncols) != n || static_cast<std::size_t>(L.nrows) != n)
            throw std::invalid_argument(
                "ldlt_numeric::set_factor: factor dimension does not match the symbolic analysis");
        if (D.size() != n)
            throw std::invalid_argument(
                "ldlt_numeric::set_factor: diagonal size does not match the symbolic analysis");
        unit_lower_solve_schedule           fsched = build_unit_lower_solve_schedule(L);
        unit_lower_transpose_solve_schedule bsched = build_unit_lower_transpose_solve_schedule(L);
        L_ = std::move(L);            // noexcept
        D_ = std::move(D);            // noexcept
        fwd_sched_ = std::move(fsched);  // noexcept
        bwd_sched_ = std::move(bsched);  // noexcept
        has_factor_ = true;              // commit last: solve() is now valid
    }

    /// Solve A*x = b using the LDL^T factorization.
    /// A = P^T L D L^T P, so x = P^T L^{-T} D^{-1} L^{-1} P b
    template <typename VecX, typename VecB>
    void solve(VecX& x, const VecB& b) const {
        // A caller can set `symbolic` (n > 0) yet never install a factor; the
        // divide by D_ below would then index an empty vector. Reject that
        // explicitly instead of walking off the end.
        if (!has_factor_)
            throw std::logic_error(
                "ldlt_numeric::solve: no factor installed (call set_factor "
                "or sparse_ldlt_numeric first)");
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

        // Step 2: Forward solve: L * y = w (unit lower). Level-scheduled
        // (parallel, bit-identical to dense_unit_lower_solve); serial at
        // MTL5_NUM_THREADS=1.
        level_scheduled_unit_lower_solve(L_, fwd_sched_, w);

        // Step 3: Diagonal solve: D * z = y
        for (std::size_t i = 0; i < n; ++i)
            w[i] /= D_[i];

        // Step 4: Back solve: L^T * u = z (unit upper). Level-scheduled
        // (parallel, bit-identical to dense_unit_lower_transpose_solve).
        level_scheduled_unit_lower_transpose_solve(L_, bwd_sched_, w);

        // Step 5: Apply inverse permutation: x = P^T * u
        for (std::size_t i = 0; i < n; ++i)
            x(symbolic.perm[i]) = static_cast<typename VecX::value_type>(w[i]);
    }

private:
    util::csc_matrix<Value>             L_;          // unit lower triangular (CSC), no stored diagonal
    std::vector<Value>                  D_;          // diagonal entries
    unit_lower_solve_schedule           fwd_sched_;  // forward (L y = w) schedule, bound to L_
    unit_lower_transpose_solve_schedule bwd_sched_;  // transpose (L^T u = z) schedule, bound to L_
    bool                                has_factor_ = false;  // set by set_factor; solve() requires it
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
/// Up-looking (left-looking) LDL^T using marker-based reach and single-pass
/// scatter for O(nnz(L)) work. For each column j:
///   1. Scatter lower-triangular entries of permuted A(:,j) into workspace
///   2. Compute reach of j in the etree using markers (no sort/unique needed)
///   3. For each reached column k, scatter L(j,k)*D(k)*L(:,k) in one pass
///   4. Store D(j) and L(j+1:n, j) = x/D(j)
///
/// \param A    Symmetric sparse matrix in CRS format
/// \param sym  Symbolic analysis from sparse_ldlt_symbolic()
/// \return     Numeric factorization result containing L (unit lower, no
///             diagonal stored) and D
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

    // Allocate L in CSC format. col_counts includes the diagonal, but we
    // don't store the unit diagonal - allocate col_counts[j]-1 per column.
    util::csc_matrix<Value> L;
    L.nrows = n;
    L.ncols = n;
    L.col_ptr.resize(n + 1);
    L.col_ptr[0] = 0;
    for (size_type j = 0; j < n; ++j) {
        size_type off_diag = (sym.col_counts[j] > 0) ? sym.col_counts[j] - 1 : 0;
        L.col_ptr[j + 1] = L.col_ptr[j] + off_diag;
    }

    size_type nnz_L = L.col_ptr[n];
    L.row_ind.resize(nnz_L);
    L.values.resize(nnz_L);

    // Diagonal vector
    std::vector<Value> D(n, Value{0});

    // Working arrays
    std::vector<Value> x(n, Value{0});      // dense workspace for column assembly
    std::vector<size_type> nz(n, 0);        // next free slot in each column of L

    // Reach workspace: marker-based etree walk (avoids sort/unique per column)
    constexpr size_type unmarked = std::numeric_limits<size_type>::max();
    std::vector<size_type> mark(n, unmarked);    // mark[k] = j if col k is in reach of j
    std::vector<size_type> emitted(n, unmarked); // emitted[i] = j if row i already stored in L(:,j)
    std::vector<size_type> reach_stack(n);       // stack for reach computation
    std::vector<size_type> reach_list;           // accumulated reach in topological order
    reach_list.reserve(n);

    // Up-looking LDL^T: process columns in order 0..n-1
    for (size_type j = 0; j < n; ++j) {
        // Scatter column j of C into dense workspace x (lower triangle only)
        for (size_type p = C.col_ptr[j]; p < C.col_ptr[j + 1]; ++p) {
            size_type i = C.row_ind[p];
            if (i >= j)
                x[i] = C.values[p];
        }

        // Compute reach: walk etree from each row i < j in C(:,j),
        // marking nodes and collecting in topological (ascending) order.
        reach_list.clear();
        for (size_type p = C.col_ptr[j]; p < C.col_ptr[j + 1]; ++p) {
            size_type i = C.row_ind[p];
            if (i >= j) continue;

            // Walk from i up the etree, pushing unmarked nodes onto stack
            size_type stack_top = 0;
            size_type node = i;
            while (node != analysis::no_parent && node < j && mark[node] != j) {
                reach_stack[stack_top++] = node;
                mark[node] = j;
                node = sym.parent[node];
            }
            // Pop stack in reverse to get topological (ascending) order
            while (stack_top > 0)
                reach_list.push_back(reach_stack[--stack_top]);
        }

        // Sort reach_list ascending so we process columns in order
        std::sort(reach_list.begin(), reach_list.end());

        // For each reached column k, single-pass scatter:
        // find L(j,k) on the fly, then subtract L(j,k)*D(k)*L(:,k) from x
        for (size_type col_k : reach_list) {
            size_type col_start = L.col_ptr[col_k];
            size_type col_end = L.col_ptr[col_k] + nz[col_k];

            // Find L(j, col_k) in column col_k of L
            // Since rows are sorted ascending and all > col_k, we can scan
            Value ljk = Value{0};
            for (size_type p = col_start; p < col_end; ++p) {
                if (L.row_ind[p] == j) {
                    ljk = L.values[p];
                    break;
                }
                if (L.row_ind[p] > j) break;  // sorted - won't find it
            }

            if (ljk == Value{0}) continue;

            Value ljk_dk = ljk * D[col_k];

            // Subtract from diagonal: x[j] -= ljk^2 * D(k)
            x[j] -= ljk_dk * ljk;

            // Subtract from off-diagonals: x[i] -= ljk*D(k)*L(i,k) for i > j
            for (size_type p = col_start; p < col_end; ++p) {
                size_type i = L.row_ind[p];
                if (i > j)
                    x[i] -= ljk_dk * L.values[p];
            }
        }

        // D(j) = x[j] (the accumulated diagonal value)
        Value dj = x[j];
        if (dj == Value{0}) {
            throw std::runtime_error(
                "sparse_ldlt_numeric: zero pivot at column " + std::to_string(j));
        }
        D[j] = dj;

        // Guarded write into column j of L (off-diagonal entries only)
        size_type col_capacity = (sym.col_counts[j] > 0) ? sym.col_counts[j] - 1 : 0;
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

        // Collect all rows i > j where x[i] != 0 (original entries + fill-in).
        // Use emitted[i] == j as epoch marker to avoid quadratic dedupe scans.
        // First: rows from original matrix C(:,j)
        for (size_type p = C.col_ptr[j]; p < C.col_ptr[j + 1]; ++p) {
            size_type i = C.row_ind[p];
            if (i > j && x[i] != Value{0}) {
                push_entry(i, x[i] / dj);
                emitted[i] = j;
            }
        }

        // Fill-in entries: rows touched by L(:,k) scatter but not in C(:,j)
        for (size_type col_k : reach_list) {
            size_type col_start = L.col_ptr[col_k];
            size_type col_end = L.col_ptr[col_k] + nz[col_k];
            for (size_type p = col_start; p < col_end; ++p) {
                size_type i = L.row_ind[p];
                if (i > j && x[i] != Value{0} && emitted[i] != j) {
                    push_entry(i, x[i] / dj);
                    emitted[i] = j;
                }
            }
        }

        // Clear workspace for rows we touched
        for (size_type p = C.col_ptr[j]; p < C.col_ptr[j + 1]; ++p)
            x[C.row_ind[p]] = Value{0};
        for (size_type col_k : reach_list) {
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
    result.symbolic = sym;
    // Install L and D and build the coupled solve schedules atomically.
    result.set_factor(std::move(L), std::move(D));
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
