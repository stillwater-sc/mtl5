#pragma once
// MTL5 -- Sparse QR factorization via Householder reflections
//
// Implements a column-by-column Householder QR using a dense workspace,
// inspired by Davis's "Direct Methods for Sparse Linear Systems", Ch. 5.
//
// Key features:
//   - Symbolic/numeric phase separation
//   - Column pre-ordering via pluggable fill-reducing ordering
//   - Householder reflections stored compactly (V vectors + beta scalars)
//   - Handles rectangular matrices (m x n, m >= n) for least-squares
//   - Solves min ||Ax - b||_2 via Q^T b then back-substitution on R
//
// Factorization: A*Q = Q_h * R where Q_h = H_0 * H_1 * ... * H_{n-1},
// with H_k = I - beta_k * v_k * v_k^T.
//
// Reference: Davis, "Direct Methods for Sparse Linear Systems", SIAM, 2006.
//            Golub & Van Loan, "Matrix Computations", 4th ed., Ch. 5.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/sparse/util/csc.hpp>
#include <mtl/sparse/util/permutation.hpp>

namespace mtl::sparse::factorization {

/// Result of symbolic QR analysis.
struct qr_symbolic {
    std::vector<std::size_t> col_perm;   // column permutation q[new]=old
    std::vector<std::size_t> col_pinv;   // inverse column permutation
    std::size_t nrows{0};
    std::size_t ncols{0};
};

/// Result of numeric QR factorization.
/// R is upper triangular (n x n, CSC). Householder vectors and betas
/// encode Q implicitly.
template <typename Value>
struct qr_numeric {
    util::csc_matrix<Value> R;                     // upper triangular (n x n)
    std::vector<std::vector<Value>> V;             // Householder vectors (dense below k)
    std::vector<std::vector<std::size_t>> V_idx;   // row indices for V
    std::vector<Value> beta;                       // Householder coefficients
    qr_symbolic symbolic;

    std::size_t num_rows() const { return symbolic.nrows; }
    std::size_t num_cols() const { return symbolic.ncols; }

    /// Solve min ||A*x - b||_2.
    /// A*Q = Q_h * R => A = Q_h * R * Q^T
    /// min ||Ax - b|| => R * Q^T * x = (Q_h^T * b)[0:n-1]
    template <typename VecX, typename VecB>
    void solve(VecX& x, const VecB& b) const {
        std::size_t m = symbolic.nrows;
        std::size_t nc = symbolic.ncols;
        if (static_cast<std::size_t>(b.size()) != m) {
            throw std::invalid_argument(
                "qr_numeric::solve: b size mismatch (expected "
                + std::to_string(m) + ")");
        }
        if (static_cast<std::size_t>(x.size()) != nc) {
            throw std::invalid_argument(
                "qr_numeric::solve: x size mismatch (expected "
                + std::to_string(nc) + ")");
        }

        // Step 1: w = Q_h^T * b = H_{n-1} * ... * H_1 * H_0 * b
        // (same order as factorization since H_k is symmetric)
        std::vector<Value> w(m);
        for (std::size_t i = 0; i < m; ++i)
            w[i] = static_cast<Value>(b(i));

        for (std::size_t k = 0; k < nc; ++k) {
            if (beta[k] == Value{0}) continue;

            // dot = v_k^T * w
            Value dot = Value{0};
            for (std::size_t idx = 0; idx < V_idx[k].size(); ++idx)
                dot += V[k][idx] * w[V_idx[k][idx]];

            // w -= beta_k * dot * v_k
            Value scale = beta[k] * dot;
            for (std::size_t idx = 0; idx < V_idx[k].size(); ++idx)
                w[V_idx[k][idx]] -= scale * V[k][idx];
        }

        // Step 2: Solve R * y = w[0:n-1]
        std::vector<Value> y(w.begin(), w.begin() + nc);

        for (std::size_t j = nc; j > 0; --j) {
            std::size_t col = j - 1;
            // Diagonal is last entry in column
            std::size_t diag_pos = R.col_ptr[col + 1] - 1;
            if (R.col_ptr[col] > diag_pos) continue;

            Value diag = R.values[diag_pos];
            if (std::abs(diag) < std::numeric_limits<Value>::min()) {
                throw std::runtime_error(
                    "qr_numeric::solve: zero diagonal in R at column "
                    + std::to_string(col) + " (rank deficient)");
            }

            y[col] /= diag;
            for (std::size_t p = R.col_ptr[col]; p < diag_pos; ++p)
                y[R.row_ind[p]] -= R.values[p] * y[col];
        }

        // Step 3: x = Q * y => x[col_perm[i]] = y[i]
        for (std::size_t i = 0; i < nc; ++i)
            x(symbolic.col_perm[i]) = static_cast<typename VecX::value_type>(y[i]);
    }
};

/// Symbolic QR analysis with ordering.
template <typename Value, typename Parameters, typename Ordering>
qr_symbolic sparse_qr_symbolic(
    const mat::compressed2D<Value, Parameters>& A,
    const Ordering& ordering)
{
    if (A.num_rows() < A.num_cols()) {
        throw std::invalid_argument(
            "sparse_qr_symbolic: requires m >= n (overdetermined or square)");
    }

    qr_symbolic sym;
    sym.nrows = A.num_rows();
    sym.ncols = A.num_cols();

    if (A.num_rows() == A.num_cols()) {
        sym.col_perm = ordering(A);
    } else {
        sym.col_perm = util::identity_permutation(A.num_cols());
    }

    if (!util::is_valid_permutation(sym.col_perm) || sym.col_perm.size() != sym.ncols) {
        throw std::invalid_argument(
            "sparse_qr_symbolic: ordering returned invalid permutation");
    }
    sym.col_pinv = util::invert_permutation(sym.col_perm);
    return sym;
}

/// Symbolic QR analysis without ordering.
template <typename Value, typename Parameters>
qr_symbolic sparse_qr_symbolic(
    const mat::compressed2D<Value, Parameters>& A)
{
    if (A.num_rows() < A.num_cols()) {
        throw std::invalid_argument(
            "sparse_qr_symbolic: requires m >= n (overdetermined or square)");
    }

    qr_symbolic sym;
    sym.nrows = A.num_rows();
    sym.ncols = A.num_cols();
    sym.col_perm = util::identity_permutation(A.num_cols());
    sym.col_pinv = sym.col_perm;
    return sym;
}

/// Numeric QR factorization.
///
/// Dense-workspace column-by-column Householder QR on the column-permuted
/// matrix. For each column k:
///   1. Scatter column k into dense workspace
///   2. Apply H_0, ..., H_{k-1} to transform it
///   3. Extract R(0:k, k) from rows 0..k
///   4. Compute Householder reflector for rows k..m-1
///
/// Householder convention (Golub & Van Loan):
///   Given x = w[k:m-1], define:
///     alpha = -sign(x[0]) * ||x||
///     v = x;  v[0] -= alpha
///     beta = 2 / (v^T * v)
///   Then H = I - beta * v * v^T satisfies H*x = alpha * e_1.
template <typename Value, typename Parameters>
qr_numeric<Value> sparse_qr_numeric(
    const mat::compressed2D<Value, Parameters>& A,
    const qr_symbolic& sym)
{
    using size_type = std::size_t;
    size_type m = sym.nrows;
    size_type nc = sym.ncols;
    if (A.num_rows() != m || A.num_cols() != nc) {
        throw std::invalid_argument(
            "sparse_qr_numeric: matrix dimensions ("
            + std::to_string(A.num_rows()) + "x" + std::to_string(A.num_cols())
            + ") do not match symbolic analysis ("
            + std::to_string(m) + "x" + std::to_string(nc) + ")");
    }

    // Column-permute A and convert to CSC
    auto AQ = util::column_permute(A, sym.col_perm);
    auto C = util::crs_to_csc(AQ);

    // Dense workspace
    std::vector<Value> w(m, Value{0});

    // Output
    std::vector<std::vector<size_type>> R_rows(nc);
    std::vector<std::vector<Value>>     R_vals(nc);
    std::vector<std::vector<size_type>> V_idx(nc);
    std::vector<std::vector<Value>>     V_vals(nc);
    std::vector<Value> betas(nc, Value{0});

    for (size_type k = 0; k < nc; ++k) {
        // Scatter column k into workspace
        for (size_type p = C.col_ptr[k]; p < C.col_ptr[k + 1]; ++p)
            w[C.row_ind[p]] = C.values[p];

        // Apply previous reflectors H_0, ..., H_{k-1}
        for (size_type j = 0; j < k; ++j) {
            if (betas[j] == Value{0}) continue;

            Value dot = Value{0};
            for (size_type idx = 0; idx < V_idx[j].size(); ++idx)
                dot += V_vals[j][idx] * w[V_idx[j][idx]];

            if (dot == Value{0}) continue;

            Value scale = betas[j] * dot;
            for (size_type idx = 0; idx < V_idx[j].size(); ++idx)
                w[V_idx[j][idx]] -= scale * V_vals[j][idx];
        }

        // Extract R entries above diagonal: w[0..k-1]
        for (size_type i = 0; i < k; ++i) {
            if (w[i] != Value{0}) {
                R_rows[k].push_back(i);
                R_vals[k].push_back(w[i]);
            }
        }

        // Compute Householder for w[k:m-1]
        // sigma = sum(w[i]^2 for i = k+1..m-1)
        Value sigma = Value{0};
        for (size_type i = k + 1; i < m; ++i)
            sigma += w[i] * w[i];

        Value x0 = w[k];

        if (sigma == Value{0} && x0 >= Value{0}) {
            // Column is already reduced; no reflection needed
            betas[k] = Value{0};
            R_rows[k].push_back(k);
            R_vals[k].push_back(x0);
            V_idx[k].push_back(k);
            V_vals[k].push_back(Value{1});
        } else if (sigma == Value{0} && x0 < Value{0}) {
            // Just flip sign
            betas[k] = Value{2};
            R_rows[k].push_back(k);
            R_vals[k].push_back(-x0);
            V_idx[k].push_back(k);
            V_vals[k].push_back(Value{1});
        } else {
            Value norm_x = std::sqrt(x0 * x0 + sigma);

            // alpha = -sign(x0) * ||x||
            Value alpha = (x0 >= Value{0}) ? -norm_x : norm_x;

            // v = x; v[0] -= alpha  =>  v[0] = x0 - alpha
            Value v0 = x0 - alpha;

            // beta = 2 / (v^T * v) = 2 / (v0^2 + sigma)
            Value vtv = v0 * v0 + sigma;
            Value beta_k = Value{2} / vtv;
            betas[k] = beta_k;

            // R(k,k) = alpha
            R_rows[k].push_back(k);
            R_vals[k].push_back(alpha);

            // Store Householder vector v = [v0, w[k+1], ..., w[m-1]]
            V_idx[k].push_back(k);
            V_vals[k].push_back(v0);
            for (size_type i = k + 1; i < m; ++i) {
                if (w[i] != Value{0}) {
                    V_idx[k].push_back(i);
                    V_vals[k].push_back(w[i]);
                }
            }
        }

        // Clear workspace
        for (size_type i = 0; i < m; ++i)
            w[i] = Value{0};
    }

    // Build R in CSC (sorted by row within each column)
    util::csc_matrix<Value> R;
    R.nrows = nc;
    R.ncols = nc;
    R.col_ptr.resize(nc + 1);
    size_type total = 0;
    for (size_type j = 0; j < nc; ++j) {
        R.col_ptr[j] = total;
        total += R_rows[j].size();
    }
    R.col_ptr[nc] = total;
    R.row_ind.resize(total);
    R.values.resize(total);

    for (size_type j = 0; j < nc; ++j) {
        std::vector<size_type> order(R_rows[j].size());
        std::iota(order.begin(), order.end(), size_type{0});
        std::sort(order.begin(), order.end(),
            [&](size_type a, size_type b) {
                return R_rows[j][a] < R_rows[j][b];
            });

        size_type pos = R.col_ptr[j];
        for (size_type idx : order) {
            R.row_ind[pos] = R_rows[j][idx];
            R.values[pos] = R_vals[j][idx];
            ++pos;
        }
    }

    qr_numeric<Value> result;
    result.R = std::move(R);
    result.V = std::move(V_vals);
    result.V_idx = std::move(V_idx);
    result.beta = std::move(betas);
    result.symbolic = sym;
    return result;
}

/// One-shot sparse QR solve with ordering.
template <typename Value, typename Parameters, typename VecX, typename VecB,
          typename Ordering>
void sparse_qr_solve(
    const mat::compressed2D<Value, Parameters>& A,
    VecX& x, const VecB& b,
    const Ordering& ordering)
{
    auto sym = sparse_qr_symbolic(A, ordering);
    auto num = sparse_qr_numeric(A, sym);
    num.solve(x, b);
}

/// One-shot sparse QR solve without ordering.
template <typename Value, typename Parameters, typename VecX, typename VecB>
void sparse_qr_solve(
    const mat::compressed2D<Value, Parameters>& A,
    VecX& x, const VecB& b)
{
    auto sym = sparse_qr_symbolic(A);
    auto num = sparse_qr_numeric(A, sym);
    num.solve(x, b);
}

} // namespace mtl::sparse::factorization
