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

    // Apply column permutation and convert to CSC
    auto AQ = util::column_permute(A, sym.col_perm);
    auto C = util::crs_to_csc(AQ);

    // Row permutation tracking
    // piv[k] = original row that is now in position k
    std::vector<size_type> piv(n);
    std::vector<size_type> piv_inv(n);  // piv_inv[original_row] = current position
    for (size_type i = 0; i < n; ++i) {
        piv[i] = i;
        piv_inv[i] = i;
    }

    // Build L and U using dynamic storage (vectors of vectors)
    // then convert to CSC at the end
    std::vector<std::vector<size_type>> L_rows(n);   // row indices per column
    std::vector<std::vector<Value>>     L_vals(n);   // values per column
    std::vector<std::vector<size_type>> U_rows(n);   // row indices per column
    std::vector<std::vector<Value>>     U_vals(n);   // values per column

    // Dense workspace with sparse tracking
    std::vector<Value> x(n, Value{0});
    std::vector<size_type> touched;  // indices touched in current column
    touched.reserve(n);

    for (size_type k = 0; k < n; ++k) {
        touched.clear();

        // Scatter column k of permuted-row C into workspace x
        // We need to apply the current row permutation
        for (size_type p = C.col_ptr[k]; p < C.col_ptr[k + 1]; ++p) {
            size_type orig_row = C.row_ind[p];
            size_type cur_pos = piv_inv[orig_row];
            x[cur_pos] = C.values[p];
            touched.push_back(cur_pos);
        }

        // Left-looking: subtract contributions from previously factored columns
        // Only visit columns j < k where x[j] != 0 (sparse traversal)
        for (size_type j = 0; j < k; ++j) {
            // U(j,k) is in x[j] at this point (entries above diagonal)
            Value ujk = x[j];
            if (ujk == Value{0}) continue;

            // Subtract ujk * L(i,j) from x[i] for all i > j in column j of L
            for (size_type idx = 0; idx < L_rows[j].size(); ++idx) {
                size_type i = L_rows[j][idx];
                if (i == j) continue;  // skip diagonal (L has unit diagonal)
                if (x[i] == Value{0}) touched.push_back(i);
                x[i] -= L_vals[j][idx] * ujk;
            }
        }

        // Threshold partial pivoting: find the best pivot in x[k..n-1]
        Value max_val = Value{0};
        size_type pivot_pos = k;
        for (size_type ti = 0; ti < touched.size(); ++ti) {
            size_type i = touched[ti];
            if (i < k) continue;
            Value abs_val = std::abs(x[i]);
            if (abs_val > max_val) {
                max_val = abs_val;
                pivot_pos = i;
            }
        }
        // Also check x[k] itself (may not be in touched if it was set to 0)
        {
            Value abs_k = std::abs(x[k]);
            if (abs_k > max_val) {
                max_val = abs_k;
                pivot_pos = k;
            }
        }

        if (max_val == Value{0}) {
            for (size_type i : touched) x[i] = Value{0};
            throw std::runtime_error(
                "sparse_lu_numeric: zero pivot at column " + std::to_string(k)
                + " (matrix is singular)");
        }

        // Apply threshold: if x[k] is large enough relative to max, keep it
        // Otherwise swap with the row that has the maximum
        if (std::abs(x[k]) < threshold * max_val && pivot_pos != k) {
            // Swap rows k and pivot_pos in workspace
            std::swap(x[k], x[pivot_pos]);

            // Update permutation tracking
            size_type orig_k = piv[k];
            size_type orig_p = piv[pivot_pos];
            std::swap(piv[k], piv[pivot_pos]);
            piv_inv[orig_k] = pivot_pos;
            piv_inv[orig_p] = k;

            // Swap rows k and pivot_pos in all previously stored L columns
            for (size_type j = 0; j < k; ++j) {
                for (size_type idx = 0; idx < L_rows[j].size(); ++idx) {
                    if (L_rows[j][idx] == k) L_rows[j][idx] = pivot_pos;
                    else if (L_rows[j][idx] == pivot_pos) L_rows[j][idx] = k;
                }
            }
        }

        // Store U(:,k): entries x[0..k] (upper triangular including diagonal)
        for (size_type i = 0; i <= k; ++i) {
            if (x[i] != Value{0}) {
                U_rows[k].push_back(i);
                U_vals[k].push_back(x[i]);
            }
        }

        // Store L(:,k): unit diagonal + entries x[k+1..n-1] / U(k,k)
        Value ukk = x[k];
        L_rows[k].push_back(k);
        L_vals[k].push_back(Value{1});  // unit diagonal

        for (size_type i = k + 1; i < n; ++i) {
            if (x[i] != Value{0}) {
                L_rows[k].push_back(i);
                L_vals[k].push_back(x[i] / ukk);
            }
        }

        // Clear only touched workspace entries
        for (size_type i : touched)
            x[i] = Value{0};
        // Also clear any entries in 0..k that weren't in touched
        for (size_type i = 0; i <= k; ++i)
            x[i] = Value{0};
    }

    // Convert L and U from vectors-of-vectors to CSC
    auto build_csc = [&](const std::vector<std::vector<size_type>>& rows,
                         const std::vector<std::vector<Value>>& vals,
                         size_type ncols) -> util::csc_matrix<Value> {
        util::csc_matrix<Value> M;
        M.nrows = n;
        M.ncols = ncols;
        M.col_ptr.resize(ncols + 1);

        // Count nnz
        size_type total = 0;
        for (size_type j = 0; j < ncols; ++j) {
            M.col_ptr[j] = total;
            total += rows[j].size();
        }
        M.col_ptr[ncols] = total;

        M.row_ind.resize(total);
        M.values.resize(total);

        for (size_type j = 0; j < ncols; ++j) {
            // Sort entries by row index within each column
            std::vector<size_type> order(rows[j].size());
            std::iota(order.begin(), order.end(), size_type{0});
            std::sort(order.begin(), order.end(),
                [&](size_type a, size_type b) {
                    return rows[j][a] < rows[j][b];
                });

            size_type pos = M.col_ptr[j];
            for (size_type idx : order) {
                M.row_ind[pos] = rows[j][idx];
                M.values[pos] = vals[j][idx];
                ++pos;
            }
        }

        return M;
    };

    lu_numeric<Value> result;
    result.L = build_csc(L_rows, L_vals, n);
    result.U = build_csc(U_rows, U_vals, n);
    result.row_perm = piv;
    result.row_pinv = piv_inv;
    result.symbolic = sym;
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
