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
#include <cstdint>
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

/// Accumulator policy for the numeric workspace of the left-looking solve.
///
/// The dense workspace column is held as `Acc` and the algorithm touches it only
/// through this trait, so the accumulation can be made exact/extended-precision
/// without MTL5 depending on any external number library: a caller supplies a
/// custom `Acc` (e.g. a Kahan/compensated accumulator, or a Universal `quire`
/// super-accumulator) by specializing `accumulator_traits<Acc, Value>`.
///
/// The default specialization makes `Acc == Value` (plain arithmetic, zero
/// overhead, identical results) -- the behavior unless a caller opts in.
///
/// `value()` rounds the accumulator to `Value` once, at the point the column
/// entry is consumed -- giving single-rounding ("exact dot product") semantics
/// when `Acc` is exact.
template <typename Acc, typename Value>
struct accumulator_traits {
    static void  clear(Acc& a)                                   { a = Acc{}; }
    static void  assign(Acc& a, const Value& v)                  { a = v; }
    static Value value(const Acc& a)                             { return static_cast<Value>(a); }
    // add_product(a, m, v) == a += m*v: the canonical accumulate-a-product
    // primitive (a dot product / quire is a sum of products). The caller passes
    // a negated multiplier for the elimination subtraction.
    static void  add_product(Acc& a, const Value& m, const Value& v) { a += m * v; }
};

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
template <typename Value, typename Parameters, typename Accumulator = Value>
    requires OrderedField<Value>
lu_numeric<Value> sparse_lu_numeric(
    const mat::compressed2D<Value, Parameters>& A,
    const lu_symbolic& sym,
    Value threshold = Value{1})
{
    using size_type = std::size_t;
    using std::abs;  // ADL: also find abs() for custom number types (e.g. posit/cfloat)
    using AT = accumulator_traits<Accumulator, Value>;  // numeric workspace policy
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
    // Row-index / node arrays use 32-bit indices to halve memory bandwidth in
    // the (memory-bound) numeric gather/scatter -- all values are node/row ids
    // in [0, n) with a -1 sentinel, and any in-memory block has n < 2^31 (KLU
    // uses int32 here too). Column pointers (Lp/Up/xprune) and the DFS resume
    // stack (pstack, which holds positions into Li up to the fill count) stay
    // 64-bit.
    using idx32 = std::int32_t;
    std::vector<idx32>          pinv(n, -1);
    std::vector<Accumulator>    x(n);             // dense numeric workspace (accumulator policy)
    std::vector<idx32>          xi(n);            // DFS stack + reach output
    std::vector<std::ptrdiff_t> pstack(n);        // DFS resume-pointer stack (Li positions)
    std::vector<char>           marked(n, 0);     // reach marks
    std::vector<idx32>          reach_nodes;      // marked nodes, for O(reach) reset
    reach_nodes.reserve(n);

    std::vector<size_type>      Lp(n + 1, 0), Up(n + 1, 0);
    // Eisenstat-Liu symmetric pruning: xprune[j] is the end of the *pruned*
    // prefix of column j of L used by the symbolic reach DFS (the numeric update
    // still uses the full column). It restricts the DFS to the pivotal-row
    // prefix once a symmetric entry (j,k)+(k,j) appears, which is the dominant
    // constant-factor win in scalar left-looking LU (SuperLU pruneL).
    std::vector<size_type>      xprune(n, 0);
    std::vector<idx32> Li;  std::vector<Value> Lx;
    std::vector<idx32> Ui;  std::vector<Value> Ux;
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
                // Symbolic DFS traverses only the pruned prefix of column jcol.
                std::ptrdiff_t pend = (jcol < 0) ? 0
                                    : static_cast<std::ptrdiff_t>(xprune[jcol]);
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
        for (size_type p = top; p < n; ++p) AT::clear(x[xi[p]]);
        for (size_type p = C.col_ptr[k]; p < C.col_ptr[k + 1]; ++p)
            AT::assign(x[C.row_ind[p]], C.values[p]);

        for (size_type px = top; px < n; ++px) {
            std::ptrdiff_t j = xi[px];
            std::ptrdiff_t jcol = pinv[j];
            if (jcol < 0) continue;                        // column not yet formed
            Value xj = AT::value(x[j]);                    // round U(j,k) once (single rounding)
            // L(j,j) == 1 (unit lower, stored first) -> no divide needed. The
            // elimination subtraction x[i] -= L(i,j)*xj is an accumulate of the
            // negated product (negating a Value is exact).
            for (size_type p = Lp[jcol] + 1; p < Lp[jcol + 1]; ++p)
                AT::add_product(x[Li[p]], -Lx[p], xj);
        }

        // --- threshold partial pivoting + emit U(:,k) for already-pivotal rows ---
        Value a = Value{0};
        std::ptrdiff_t ipiv = -1;
        bool have = false;
        for (size_type px = top; px < n; ++px) {
            std::ptrdiff_t i = xi[px];
            if (pinv[i] < 0) {                             // pivot candidate / L entry
                Value t = abs(AT::value(x[i]));
                if (!have || t > a) { a = t; ipiv = i; have = true; }
            } else {                                       // U(pinv[i], k)
                Ui.push_back(pinv[i]);
                Ux.push_back(AT::value(x[i]));
            }
        }
        if (ipiv < 0 || a == Value{0}) {
            throw std::runtime_error(
                "sparse_lu_numeric: zero pivot at column " + std::to_string(k)
                + " (matrix is singular)");
        }
        // Prefer the natural diagonal (row k) if it meets the threshold.
        if (pinv[static_cast<std::ptrdiff_t>(k)] < 0
            && abs(AT::value(x[k])) >= threshold * a) {
            ipiv = static_cast<std::ptrdiff_t>(k);
        }

        Value pivot = AT::value(x[ipiv]);
        Ui.push_back(static_cast<std::ptrdiff_t>(k));      // U(k,k) stored last
        Ux.push_back(pivot);
        pinv[ipiv] = static_cast<std::ptrdiff_t>(k);

        Li.push_back(ipiv);                                // L(k,k) = 1 stored first
        Lx.push_back(Value{1});
        for (size_type px = top; px < n; ++px) {
            std::ptrdiff_t i = xi[px];
            if (pinv[i] < 0) {                             // remaining -> L(i,k)
                Li.push_back(i);
                Lx.push_back(AT::value(x[i]) / pivot);
            }
            AT::clear(x[i]);                               // clear workspace (reach only)
        }
        xprune[k] = Li.size();                             // column k initially unpruned

        // --- symmetric pruning (Eisenstat-Liu / SuperLU pruneL) ---
        // For each column j contributing to U(:,k): if the symmetric entry is
        // present (pivot row of k appears in L(:,j)), partition L(:,j) so the
        // already-pivotal rows form a prefix, and prune the DFS to that prefix.
        {
            size_type u_diag = Ui.size() - 1;              // last U entry is diag k
            for (size_type up = Up[k]; up < u_diag; ++up) {
                size_type j = static_cast<size_type>(Ui[up]);
                if (xprune[j] != Lp[j + 1]) continue;      // already pruned
                bool sym = false;                          // pivot row ipiv in L(:,j)?
                for (size_type p = Lp[j]; p < Lp[j + 1]; ++p)
                    if (Li[p] == ipiv) { sym = true; break; }
                if (!sym) continue;
                size_type kmin = Lp[j] + 1, kmax = Lp[j + 1] - 1;  // skip diagonal
                while (kmin <= kmax) {
                    if (pinv[Li[kmax]] < 0)            { --kmax; }
                    else if (pinv[Li[kmin]] >= 0)      { ++kmin; }
                    else {
                        std::swap(Li[kmin], Li[kmax]);
                        std::swap(Lx[kmin], Lx[kmax]);
                        ++kmin; --kmax;
                    }
                }
                xprune[j] = kmin;                          // [Lp[j], kmin) = pivotal prefix
            }
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

    // Build CSC, sorting each column by row index (ascending). Sorting keeps the
    // diagonal first in L (rows >= k, min = k) and last in U (rows <= k, max = k)
    // -- the layout the triangular solves expect -- and gives each column a
    // topological order so sparse_lu_refactor() can replay the numeric values
    // without a reach DFS. The cost is O(fill * log(col_len)), negligible vs the
    // numeric flops.
    std::vector<size_type> ord;
    auto to_csc = [&](const std::vector<size_type>& Mp,
                      const std::vector<idx32>& Mi,
                      const std::vector<Value>& Mx) -> util::csc_matrix<Value> {
        util::csc_matrix<Value> M;
        M.nrows = n;
        M.ncols = n;
        M.col_ptr = Mp;
        M.row_ind.resize(Mi.size());
        M.values.resize(Mx.size());
        for (size_type c = 0; c < n; ++c) {
            size_type b = Mp[c], e = Mp[c + 1];
            ord.resize(e - b);
            for (size_type t = 0; t < e - b; ++t) ord[t] = b + t;
            std::sort(ord.begin(), ord.end(),
                      [&](size_type a, size_type c2) { return Mi[a] < Mi[c2]; });
            for (size_type t = 0; t < e - b; ++t) {
                M.row_ind[b + t] = static_cast<size_type>(Mi[ord[t]]);
                M.values[b + t]  = Mx[ord[t]];
            }
        }
        return M;
    };
    result.L = to_csc(Lp, Li, Lx);
    result.U = to_csc(Up, Ui, Ux);
    return result;
}

/// Refactorize a matrix with the SAME sparsity pattern as a prior factorization,
/// reusing that factorization's symbolic structure and pivot sequence. Only the
/// numeric values are recomputed -- no BTF/ordering, no reach DFS, and no pivot
/// search -- which is the fast path for repeated solves of a fixed pattern with
/// changing values (e.g. SPICE transient analysis: one factor, many refactor).
///
/// Preconditions: A has the same column ordering (prev.symbolic) and the same
/// nonzero pattern that produced prev. Relies on prev.L / prev.U columns being
/// stored in ascending row order (as sparse_lu_numeric produces).
///
/// \throws std::runtime_error if a reused pivot is numerically zero for the new
///         values (the prior pivot sequence is no longer valid).
template <typename Value, typename Parameters>
    requires OrderedField<Value>
lu_numeric<Value> sparse_lu_refactor(
    const mat::compressed2D<Value, Parameters>& A,
    const lu_numeric<Value>& prev)
{
    using size_type = std::size_t;
    size_type n = prev.symbolic.n;
    if (A.num_rows() != n || A.num_cols() != n) {
        throw std::invalid_argument(
            "sparse_lu_refactor: matrix dimensions do not match prior factorization");
    }

    auto AQ = util::column_permute(A, prev.symbolic.col_perm);
    auto C  = util::crs_to_csc(AQ);

    const auto& pinv = prev.row_pinv;    // original row -> pivot position (fixed)
    lu_numeric<Value> result = prev;     // reuse pattern + perms + symbolic
    auto& L = result.L;                  // overwrite values in place
    auto& U = result.U;

    std::vector<Value> x(n, Value{0});

    for (size_type k = 0; k < n; ++k) {
        // Scatter column k of the (column-permuted) A into the pivot-space
        // workspace.
        for (size_type p = C.col_ptr[k]; p < C.col_ptr[k + 1]; ++p)
            x[pinv[C.row_ind[p]]] = C.values[p];

        // Left-looking updates from contributing columns j < k. U(:,k) is sorted
        // ascending with the diagonal (row k) last, so the strict-upper entries
        // are exactly the contributing columns in topological order.
        size_type ubeg = U.col_ptr[k], uend = U.col_ptr[k + 1];
        for (size_type p = ubeg; p + 1 < uend; ++p) {   // exclude diagonal (last)
            size_type j = U.row_ind[p];
            Value xj = x[j];
            if (xj == Value{0}) continue;
            // L(:,j): diagonal (row j, value 1) is first -> skip it.
            for (size_type q = L.col_ptr[j] + 1; q < L.col_ptr[j + 1]; ++q)
                x[L.row_ind[q]] -= L.values[q] * xj;
        }

        Value pivot = x[k];
        if (pivot == Value{0}) {
            throw std::runtime_error(
                "sparse_lu_refactor: zero pivot at column " + std::to_string(k)
                + " (prior pivot sequence invalid for the new values)");
        }

        // Write U(:,k) values (all rows <= k, including the diagonal = pivot).
        for (size_type p = ubeg; p < uend; ++p)
            U.values[p] = x[U.row_ind[p]];

        // Write L(:,k): unit diagonal first, then x[row]/pivot below.
        for (size_type p = L.col_ptr[k]; p < L.col_ptr[k + 1]; ++p) {
            size_type r = L.row_ind[p];
            L.values[p] = (r == k) ? Value{1} : x[r] / pivot;
        }

        // Clear the workspace. Clear the U/L pattern AND the scattered A column:
        // for an exact same-pattern refactor these coincide, but if A carries an
        // entry outside the prior pattern (a pattern mismatch) it is scattered
        // here yet never consumed by the replay -- clearing the scatter positions
        // too keeps such an entry from polluting later columns.
        for (size_type p = ubeg; p < uend; ++p)               x[U.row_ind[p]] = Value{0};
        for (size_type p = L.col_ptr[k]; p < L.col_ptr[k + 1]; ++p) x[L.row_ind[p]] = Value{0};
        for (size_type p = C.col_ptr[k]; p < C.col_ptr[k + 1]; ++p) x[pinv[C.row_ind[p]]] = Value{0};
    }

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
