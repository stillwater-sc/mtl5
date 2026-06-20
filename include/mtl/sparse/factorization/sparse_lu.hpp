#pragma once
// MTL5 -- Sparse LU factorization with partial pivoting
//
// Implements the left-looking sparse LU algorithm from Davis's
// "Direct Methods for Sparse Linear Systems", Chapter 6.
//
// Key features:
//   - Symbolic/numeric phase separation
//   - Column pre-ordering via pluggable fill-reducing ordering
//   - Threshold partial pivoting for numerical stability
//   - Produces L (unit lower triangular) and U (upper triangular) in CSC
//   - Row permutation tracked from pivoting
//
// Algorithm (left-looking LU for column k):
//   1. Solve L(1:k-1, 1:k-1) * x = A(:, q[k]) via sparse triangular solve
//   2. x now contains U(1:k-1, k) and candidate L(:, k) entries
//   3. Select pivot: row i >= k maximizing |x[i]| (threshold pivoting)
//   4. Swap pivot row with row k
//   5. U(1:k, k) = x[1:k], L(k+1:n, k) = x[k+1:n] / U(k,k)
//
// The factorization computes P*A*Q = L*U where:
//   P = row permutation (from pivoting)
//   Q = column permutation (from fill-reducing ordering)
//   L = unit lower triangular
//   U = upper triangular
//
// Reference: Davis, "Direct Methods for Sparse Linear Systems", SIAM, 2006.
//            CSparse: cs_lu, cs_sqr.
//            Gilbert & Peierls, "Sparse Partial Pivoting in Time Proportional
//            to Arithmetic Operations", SIAM J. Sci. Stat. Comput., 1988.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/concepts/scalar.hpp>
#include <mtl/sparse/util/csc.hpp>
#include <mtl/sparse/util/permutation.hpp>
#include <mtl/sparse/factorization/triangular_solve.hpp>

namespace mtl::sparse::factorization {

/// Result of symbolic LU analysis.
/// Captures the column ordering. For LU, the symbolic phase is lighter
/// than Cholesky because row pivoting during numeric factorization can
/// change the fill pattern.
struct lu_symbolic {
    std::vector<std::size_t> col_perm;   // column permutation q[new]=old
    std::vector<std::size_t> col_pinv;   // inverse column permutation
    std::size_t n{0};                    // matrix dimension
};

/// Result of numeric LU factorization.
/// Contains L (unit lower triangular) and U (upper triangular) in CSC,
/// plus the row and column permutations.
template <typename Value>
struct lu_numeric {
    util::csc_matrix<Value> L;           // unit lower triangular factor (CSC)
    util::csc_matrix<Value> U;           // upper triangular factor (CSC)
    std::vector<std::size_t> row_perm;   // row permutation from pivoting (p[new]=old)
    std::vector<std::size_t> row_pinv;   // inverse row permutation
    lu_symbolic symbolic;                // symbolic analysis used

    std::size_t num_rows() const { return symbolic.n; }
    std::size_t num_cols() const { return symbolic.n; }

    /// Solve A*x = b using the LU factorization.
    /// P*A*Q = L*U, so A = P^T * L * U * Q^T
    /// A*x = b => P^T * L * U * Q^T * x = b
    /// => L * U * Q^T * x = P * b
    /// Let y = Q^T * x, then L * U * y = P * b
    /// 1. w = P * b
    /// 2. L * z = w (forward solve)
    /// 3. U * y = z (back solve)
    /// 4. x = Q * y
    template <typename VecX, typename VecB>
    void solve(VecX& x, const VecB& b) const {
        std::size_t n = symbolic.n;
        if (static_cast<std::size_t>(x.size()) != n ||
            static_cast<std::size_t>(b.size()) != n) {
            throw std::invalid_argument(
                "lu_numeric::solve: vector size mismatch (expected "
                + std::to_string(n) + ")");
        }

        // Step 1: Apply row permutation: w = P * b
        std::vector<Value> w(n);
        for (std::size_t i = 0; i < n; ++i)
            w[i] = static_cast<Value>(b(row_perm[i]));

        // Step 2: Forward solve: L * z = w (L is unit lower triangular)
        dense_lower_solve(L, w);

        // Step 3: Back solve: U * y = z
        dense_upper_solve(U, w);

        // Step 4: Apply column permutation: x = Q * y
        // Q maps new col -> old col, so x[q[i]] = y[i]
        for (std::size_t i = 0; i < n; ++i)
            x(symbolic.col_perm[i]) = static_cast<typename VecX::value_type>(w[i]);
    }
};

/// Perform symbolic LU analysis on a square sparse matrix.
/// Computes the column permutation for fill reduction.
///
/// \param A        Square sparse matrix in CRS format
/// \param ordering Fill-reducing ordering functor
/// \return         Symbolic analysis result
template <typename Value, typename Parameters, typename Ordering>
lu_symbolic sparse_lu_symbolic(
    const mat::compressed2D<Value, Parameters>& A,
    const Ordering& ordering)
{
    if (A.num_rows() != A.num_cols()) {
        throw std::invalid_argument(
            "sparse_lu_symbolic: matrix must be square");
    }
    std::size_t n = A.num_rows();

    lu_symbolic sym;
    sym.n = n;
    sym.col_perm = ordering(A);
    if (!util::is_valid_permutation(sym.col_perm) || sym.col_perm.size() != n) {
        throw std::invalid_argument(
            "sparse_lu_symbolic: ordering returned invalid permutation");
    }
    sym.col_pinv = util::invert_permutation(sym.col_perm);
    return sym;
}

/// Overload without ordering: uses identity permutation (natural ordering).
template <typename Value, typename Parameters>
lu_symbolic sparse_lu_symbolic(
    const mat::compressed2D<Value, Parameters>& A)
{
    if (A.num_rows() != A.num_cols()) {
        throw std::invalid_argument(
            "sparse_lu_symbolic: matrix must be square");
    }
    std::size_t n = A.num_rows();

    lu_symbolic sym;
    sym.n = n;
    sym.col_perm = util::identity_permutation(n);
    sym.col_pinv = sym.col_perm;
    return sym;
}

/// Perform numeric LU factorization with threshold partial pivoting.
///
/// Uses the left-looking algorithm: for each column k, performs a sparse
/// triangular solve with the already-factored columns of L, then selects
/// a pivot from the remaining entries.
///
/// \param A         Square sparse matrix in CRS format
/// \param sym       Symbolic analysis from sparse_lu_symbolic()
/// \param threshold Pivoting threshold in (0, 1]. A pivot candidate at row i
///                  is acceptable if |a(i,k)| >= threshold * max|a(j,k)| for j>=k.
///                  Default 1.0 = full partial pivoting. Smaller values trade
///                  stability for sparsity preservation.
/// \return          Numeric factorization result containing L, U, and permutations
///
/// \throws std::runtime_error if a zero pivot is encountered (singular matrix)
template <typename Value, typename Parameters>
    requires OrderedField<Value>
lu_numeric<Value> sparse_lu_numeric(
    const mat::compressed2D<Value, Parameters>& A,
    const lu_symbolic& sym,
    Value threshold = Value{1})
{
    using size_type = std::size_t;
    using std::abs;  // ADL: also find abs() for custom number types (e.g. posit/cfloat)
    size_type n = sym.n;
    if (A.num_rows() != n || A.num_cols() != n) {
        throw std::invalid_argument(
            "sparse_lu_numeric: matrix dimensions ("
            + std::to_string(A.num_rows()) + "x" + std::to_string(A.num_cols())
            + ") do not match symbolic analysis (n=" + std::to_string(n) + ")");
    }
    if (sym.col_perm.size() != n || !util::is_valid_permutation(sym.col_perm)) {
        throw std::invalid_argument(
            "sparse_lu_numeric: symbolic permutation is invalid or wrong size");
    }

    // Column-permute A (apply the fill-reducing ordering) and convert to CSC.
    auto AQ = util::column_permute(A, sym.col_perm);
    auto C = util::crs_to_csc(AQ);

    // Left-looking Gilbert-Peierls LU with threshold partial pivoting
    // (Davis, "Direct Methods for Sparse Linear Systems", cs_lu). Each column
    // is computed by a sparse triangular solve over the *reach* of C(:,k) in
    // the partially-built L, so the total work is O(flops), not O(n^2).
    //
    //   pinv[old_row] = pivot position of that row, or -1 if not yet pivotal.
    //
    // L and U are accumulated in flat CSC arrays. L row indices are kept in
    // original-row space during factorization and remapped to pivot positions
    // at the end. L is unit lower (diagonal stored first per column, value 1);
    // U has its diagonal stored last per column -- the layouts dense_lower_solve
    // / dense_upper_solve expect.
    std::vector<std::ptrdiff_t> pinv(n, -1);
    std::vector<Value>          x(n, Value{0});   // dense numeric workspace
    std::vector<std::ptrdiff_t> xi(n);            // DFS stack + reach output
    std::vector<std::ptrdiff_t> pstack(n);        // DFS resume-pointer stack
    std::vector<char>           marked(n, 0);     // reach marks
    std::vector<std::ptrdiff_t> reach_nodes;      // marked nodes, for O(reach) reset
    reach_nodes.reserve(n);

    std::vector<size_type>      Lp(n + 1, 0), Up(n + 1, 0);
    std::vector<std::ptrdiff_t> Li;  std::vector<Value> Lx;
    std::vector<std::ptrdiff_t> Ui;  std::vector<Value> Ux;
    Li.reserve(C.values.size() * 2); Lx.reserve(C.values.size() * 2);
    Ui.reserve(C.values.size() * 2); Ux.reserve(C.values.size() * 2);

    for (size_type k = 0; k < n; ++k) {
        Lp[k] = Li.size();
        Up[k] = Ui.size();

        // --- reach(C(:,k)) in L: iterative DFS, topological order in xi[top..n) ---
        reach_nodes.clear();
        size_type top = n;
        for (size_type p = C.col_ptr[k]; p < C.col_ptr[k + 1]; ++p) {
            std::ptrdiff_t start = static_cast<std::ptrdiff_t>(C.row_ind[p]);
            if (marked[start]) continue;
            std::ptrdiff_t head = 0;
            xi[0] = start;
            while (head >= 0) {
                std::ptrdiff_t node = xi[head];
                std::ptrdiff_t jcol = pinv[node];          // column of L, if pivotal
                if (!marked[node]) {
                    marked[node] = 1;
                    reach_nodes.push_back(node);
                    pstack[head] = (jcol < 0) ? 0
                                 : static_cast<std::ptrdiff_t>(Lp[jcol]);
                }
                bool done = true;
                std::ptrdiff_t pend = (jcol < 0) ? 0
                                    : static_cast<std::ptrdiff_t>(Lp[jcol + 1]);
                for (std::ptrdiff_t p2 = pstack[head]; p2 < pend; ++p2) {
                    std::ptrdiff_t child = Li[p2];
                    if (marked[child]) continue;
                    pstack[head] = p2;                     // resume here on return
                    xi[++head] = child;
                    done = false;
                    break;
                }
                if (done) {
                    --head;
                    xi[--top] = node;
                }
            }
        }

        // --- scatter C(:,k) over the reach, then x = L \ C(:,k) ---
        for (size_type p = top; p < n; ++p) x[xi[p]] = Value{0};
        for (size_type p = C.col_ptr[k]; p < C.col_ptr[k + 1]; ++p)
            x[C.row_ind[p]] = C.values[p];

        for (size_type px = top; px < n; ++px) {
            std::ptrdiff_t j = xi[px];
            std::ptrdiff_t jcol = pinv[j];
            if (jcol < 0) continue;                        // column not yet formed
            // L(j,j) == 1 (unit lower, stored first) -> no divide needed.
            for (size_type p = Lp[jcol] + 1; p < Lp[jcol + 1]; ++p)
                x[Li[p]] -= Lx[p] * x[j];
        }

        // --- threshold partial pivoting + emit U(:,k) for already-pivotal rows ---
        Value a = Value{0};
        std::ptrdiff_t ipiv = -1;
        bool have = false;
        for (size_type px = top; px < n; ++px) {
            std::ptrdiff_t i = xi[px];
            if (pinv[i] < 0) {                             // pivot candidate / L entry
                Value t = abs(x[i]);
                if (!have || t > a) { a = t; ipiv = i; have = true; }
            } else {                                       // U(pinv[i], k)
                Ui.push_back(pinv[i]);
                Ux.push_back(x[i]);
            }
        }
        if (ipiv < 0 || a == Value{0}) {
            throw std::runtime_error(
                "sparse_lu_numeric: zero pivot at column " + std::to_string(k)
                + " (matrix is singular)");
        }
        // Prefer the natural diagonal (row k) if it meets the threshold.
        if (pinv[static_cast<std::ptrdiff_t>(k)] < 0
            && abs(x[k]) >= threshold * a) {
            ipiv = static_cast<std::ptrdiff_t>(k);
        }

        Value pivot = x[ipiv];
        Ui.push_back(static_cast<std::ptrdiff_t>(k));      // U(k,k) stored last
        Ux.push_back(pivot);
        pinv[ipiv] = static_cast<std::ptrdiff_t>(k);

        Li.push_back(ipiv);                                // L(k,k) = 1 stored first
        Lx.push_back(Value{1});
        for (size_type px = top; px < n; ++px) {
            std::ptrdiff_t i = xi[px];
            if (pinv[i] < 0) {                             // remaining -> L(i,k)
                Li.push_back(i);
                Lx.push_back(x[i] / pivot);
            }
            x[i] = Value{0};                               // clear workspace (reach only)
        }

        for (std::ptrdiff_t node : reach_nodes) marked[node] = 0;  // O(reach) reset
    }
    Lp[n] = Li.size();
    Up[n] = Ui.size();

    // Remap L row indices from original-row space to pivot positions.
    for (size_type p = 0; p < Li.size(); ++p)
        Li[p] = pinv[Li[p]];

    lu_numeric<Value> result;
    result.symbolic = sym;

    // Row permutation: pinv[old] = position; row_perm[position] = old.
    result.row_perm.resize(n);
    result.row_pinv.resize(n);
    for (size_type i = 0; i < n; ++i) {
        size_type pos = static_cast<size_type>(pinv[i]);
        result.row_pinv[i] = pos;
        result.row_perm[pos] = i;
    }

    auto to_csc = [&](const std::vector<size_type>& Mp,
                      const std::vector<std::ptrdiff_t>& Mi,
                      const std::vector<Value>& Mx) -> util::csc_matrix<Value> {
        util::csc_matrix<Value> M;
        M.nrows = n;
        M.ncols = n;
        M.col_ptr = Mp;
        M.row_ind.resize(Mi.size());
        for (size_type p = 0; p < Mi.size(); ++p)
            M.row_ind[p] = static_cast<size_type>(Mi[p]);
        M.values = Mx;
        return M;
    };
    result.L = to_csc(Lp, Li, Lx);
    result.U = to_csc(Up, Ui, Ux);
    return result;
}

/// One-shot sparse LU solve: factor and solve A*x = b.
template <typename Value, typename Parameters, typename VecX, typename VecB,
          typename Ordering>
    requires OrderedField<Value>
void sparse_lu_solve(
    const mat::compressed2D<Value, Parameters>& A,
    VecX& x, const VecB& b,
    const Ordering& ordering,
    Value threshold = Value{1})
{
    auto sym = sparse_lu_symbolic(A, ordering);
    auto num = sparse_lu_numeric(A, sym, threshold);
    num.solve(x, b);
}

/// One-shot sparse LU solve without ordering (natural ordering).
template <typename Value, typename Parameters, typename VecX, typename VecB>
void sparse_lu_solve(
    const mat::compressed2D<Value, Parameters>& A,
    VecX& x, const VecB& b)
{
    auto sym = sparse_lu_symbolic(A);
    auto num = sparse_lu_numeric(A, sym);
    num.solve(x, b);
}

} // namespace mtl::sparse::factorization
