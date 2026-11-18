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
#include <functional>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>
#include <type_traits>

#include <mtl/concepts/scalar.hpp>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/sparse/util/csc.hpp>
#include <mtl/sparse/util/permutation.hpp>
#include <mtl/math/accumulator_traits.hpp>
#include <mtl/math/pow2_scaling.hpp>

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
    ///
    /// `Accumulator` (default `void`, leading as in sparse_qr_numeric) selects the
    /// accumulator policy of the solve-time reductions (#535): the `v_k^T w` dot
    /// of the Q_h^T b step and each row's `w[i] - sum_k R(i,k) y[k]` of the R
    /// back-substitution. `void` is the raw Value path, byte for byte. A non-void
    /// `Acc` holds one accumulator per row of R (O(ncols) workspace, see
    /// math/accumulator_traits.hpp) so that each y[i] is rounded once, when its
    /// column is reached, and the terms are added in the same order as the
    /// default scatter.
    template <typename Accumulator = void, typename VecX, typename VecB>
    void solve(VecX& x, const VecB& b) const {
        using std::abs;  // ADL: also find abs() for custom number types
        constexpr bool raw = std::is_void_v<Accumulator>;
        using AT = mtl::math::accumulator_traits<
            std::conditional_t<raw, Value, Accumulator>, Value>;
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
            if constexpr (raw) {
                for (std::size_t idx = 0; idx < V_idx[k].size(); ++idx)
                    dot += V[k][idx] * w[V_idx[k][idx]];
            } else {
                Accumulator dot_acc{};
                AT::clear(dot_acc);
                for (std::size_t idx = 0; idx < V_idx[k].size(); ++idx)
                    AT::add_product(dot_acc, V[k][idx], w[V_idx[k][idx]]);
                dot = AT::value(dot_acc);
            }

            // w -= beta_k * dot * v_k
            Value scale = beta[k] * dot;
            for (std::size_t idx = 0; idx < V_idx[k].size(); ++idx)
                w[V_idx[k][idx]] -= scale * V[k][idx];
        }

        // Step 2: Solve R * y = w[0:n-1]
        std::vector<Value> y(w.begin(), w.begin() + nc);

        // Accumulator path: ws[i] carries w[i] minus the R(i,col)*y[col] terms
        // scattered so far; y[col] is rounded out of ws[col] when col is reached.
        // (Unused, and empty, on the raw path.)
        using ws_type = std::conditional_t<raw, Value, Accumulator>;
        std::vector<ws_type> ws;
        if constexpr (!raw) {
            ws.resize(nc);
            for (std::size_t i = 0; i < nc; ++i) AT::assign(ws[i], y[i]);
        }

        for (std::size_t j = nc; j > 0; --j) {
            std::size_t col = j - 1;
            if constexpr (!raw) y[col] = AT::value(ws[col]);   // round once, on consume
            // An empty column is tested before the diagonal's position is
            // formed: col_ptr[col + 1] - 1 wraps when col_ptr[col + 1] == 0
            // (issue 907, as in dense_upper_solve).
            if (R.col_ptr[col] >= R.col_ptr[col + 1]) continue;
            // Diagonal is last entry in column
            std::size_t diag_pos = R.col_ptr[col + 1] - 1;

            Value diag = R.values[diag_pos];
            // Rank deficiency means an exactly zero (or subnormal) R(j,j); a
            // small but normal one is solved as is, so a nearly singular A
            // gives a large x rather than an exception. A relative tolerance
            // (n eps max|R(j,j)|) was considered and not adopted (issue 864):
            // it would reject systems that solve today.
            if (abs(diag) < std::numeric_limits<Value>::min()) {
                throw std::runtime_error(
                    "qr_numeric::solve: zero diagonal in R at column "
                    + std::to_string(col) + " (rank deficient)");
            }

            y[col] /= diag;
            if constexpr (raw) {
                for (std::size_t p = R.col_ptr[col]; p < diag_pos; ++p)
                    y[R.row_ind[p]] -= R.values[p] * y[col];
            } else {
                for (std::size_t p = R.col_ptr[col]; p < diag_pos; ++p)
                    AT::add_product(ws[R.row_ind[p]], -R.values[p], y[col]);
            }
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

    // Every shape goes to the ordering (issue 907): a rectangular A used to
    // get the identity without a word, although COLAMD orders it. An ordering
    // that needs a square matrix (AMD, RCM) now says so by throwing.
    sym.col_perm = ordering(A);

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
template <typename Accumulator = void, typename Value, typename Parameters>
    requires OrderedField<Value>   // magnitudes are compared (issue 907), as in sparse_lu_numeric
qr_numeric<Value> sparse_qr_numeric(
    const mat::compressed2D<Value, Parameters>& A,
    const qr_symbolic& sym)
{
    // void (the default) accumulates in Value, as the former `Accumulator = Value` did.
    using Acc = std::conditional_t<std::is_void_v<Accumulator>, Value, Accumulator>;
    using size_type = std::size_t;
    using std::sqrt;  // ADL: also find sqrt() for custom number types
    using AT = mtl::math::accumulator_traits<Acc, Value>;
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

    // Column magnitudes outside [safe_lo, safe_hi] are scaled before their
    // squares are summed (see the Householder step below). The largest square
    // formed is v^T v = v0^2 + sigma <= (|x0| + ||x||)^2 + sigma <= 4 ||x||^2
    // <= 4 m amax^2, so amax <= sqrt(max / 4m) keeps it finite: a column
    // [2^511; 2^511] passed sqrt(max / m), overflowed v0^2 and got beta = 0
    // (#868 review).
    using std::abs;
    using limits = std::numeric_limits<Value>;
    const Value safe_hi = sqrt(limits::max() / static_cast<Value>(4.0 * static_cast<double>(m > 0 ? m : 1)));
    const Value safe_lo = sqrt(limits::min() / limits::epsilon());

    // Output
    std::vector<std::vector<size_type>> R_rows(nc);
    std::vector<std::vector<Value>>     R_vals(nc);
    std::vector<std::vector<size_type>> V_idx(nc);
    std::vector<std::vector<Value>>     V_vals(nc);
    std::vector<Value> betas(nc, Value{0});

    // Column k touches only the rows in its pattern, and only the reflectors
    // whose vectors meet that pattern (issue 907): applying every earlier
    // reflector, and scanning and clearing all m rows of w, made QR quadratic
    // in n even for a tridiagonal A. `pat` lists the rows of w that may be
    // nonzero; `refl_of_row[r]` the reflectors whose vectors hold row r.
    //
    // The arithmetic is unchanged. A reflector H_j that the old loop applied
    // had a vector meeting w's pattern when its turn came (otherwise its dot
    // product is a sum of products with zeros: zero for finite entries, and
    // the old loop skipped it): that row was either in
    // column k or brought in by an earlier reflector, and either way H_j was
    // queued then. The queue is applied in ascending j, as before; the sums
    // over rows run in ascending row order, as before, and leave out only rows
    // that are exactly zero.
    std::vector<size_type> pat;
    std::vector<size_type> row_stamp(m, 0);          // row r is in pat iff row_stamp[r] == k + 1
    std::vector<size_type> refl_stamp(nc, 0);        // H_j is queued iff refl_stamp[j] == k + 1
    std::vector<std::vector<size_type>> refl_of_row(m);
    std::priority_queue<size_type, std::vector<size_type>, std::greater<size_type>> queue;

    for (size_type k = 0; k < nc; ++k) {
        const size_type stamp = k + 1;
        pat.clear();
        // Row r enters w's pattern: queue the reflectors holding it that come
        // after `after` (the reflector being applied; none at the start).
        auto add_row = [&](size_type r, bool any, size_type after) {
            if (row_stamp[r] == stamp) return;
            row_stamp[r] = stamp;
            pat.push_back(r);
            for (const size_type j : refl_of_row[r])
                if ((any || j > after) && refl_stamp[j] != stamp) {
                    refl_stamp[j] = stamp;
                    queue.push(j);
                }
        };

        // Scatter column k into workspace
        for (size_type p = C.col_ptr[k]; p < C.col_ptr[k + 1]; ++p) {
            w[C.row_ind[p]] = C.values[p];
            add_row(C.row_ind[p], true, 0);
        }

        // Apply the previous reflectors that meet w, in ascending order
        while (!queue.empty()) {
            const size_type j = queue.top();
            queue.pop();
            if (betas[j] == Value{0}) continue;

            Acc dot_acc;
            AT::clear(dot_acc);
            for (size_type idx = 0; idx < V_idx[j].size(); ++idx)
                AT::add_product(dot_acc, V_vals[j][idx], w[V_idx[j][idx]]);
            Value dot = AT::value(dot_acc);

            if (dot == Value{0}) continue;

            Value scale = betas[j] * dot;
            for (size_type idx = 0; idx < V_idx[j].size(); ++idx) {
                w[V_idx[j][idx]] -= scale * V_vals[j][idx];
                add_row(V_idx[j][idx], false, j);
            }
        }
        std::sort(pat.begin(), pat.end());
        const auto pat_k = std::lower_bound(pat.begin(), pat.end(), k);   // first row >= k

        // Extract R entries above diagonal: w[0..k-1]
        for (auto it = pat.begin(); it != pat_k; ++it) {
            const size_type i = *it;
            if (w[i] != Value{0}) {
                R_rows[k].push_back(i);
                R_vals[k].push_back(w[i]);
            }
        }

        // Compute Householder for w[k:m-1].
        // The sum of squares overflows for entries near sqrt(max()/4m) and loses
        // its digits (or underflows to 0) near sqrt(min()/eps): (1e200, 1e200)
        // gave R(k,k) = -inf (issue 864). Such a column is scaled by 2^-e into
        // [1/2, 1): exact, so the reflector is the same, and v and beta are
        // stored scaled (H = I - beta v v^T does not change). Every other
        // column takes the unscaled path, bit for bit as before.
        Value amax = Value{0};
        for (auto it = pat_k; it != pat.end(); ++it) {
            const Value a = abs(w[*it]);
            if (amax < a) amax = a;
        }
        int e = 0;
        if (amax > Value{0} && (amax > safe_hi || amax < safe_lo))
            e = math::pow2_scaling<Value>::exponent(amax) + 1;
        auto down = [e](const Value& v) {
            return e == 0 ? v : math::pow2_scaling<Value>::scale(v, -e);
        };

        // sigma = sum(w[i]^2 for i = k+1..m-1), scaled when e != 0
        Acc sigma_acc;
        AT::clear(sigma_acc);
        for (auto it = pat_k; it != pat.end(); ++it) {
            if (*it == k) continue;   // rows k+1..m-1
            const Value wi = down(w[*it]);
            AT::add_product(sigma_acc, wi, wi);
        }
        Value sigma = AT::value(sigma_acc);

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
            const Value x0s = down(x0);
            Value norm_x = sqrt(x0s * x0s + sigma);

            // alpha = -sign(x0) * ||x||
            Value alpha = (x0s >= Value{0}) ? -norm_x : norm_x;

            // v = x; v[0] -= alpha  =>  v[0] = x0 - alpha
            Value v0 = x0s - alpha;

            // beta = 2 / (v^T * v) = 2 / (v0^2 + sigma)
            Value vtv = v0 * v0 + sigma;
            Value beta_k = Value{2} / vtv;
            betas[k] = beta_k;

            // R(k,k) = alpha, in A's units
            R_rows[k].push_back(k);
            R_vals[k].push_back(e == 0 ? alpha : math::pow2_scaling<Value>::scale(alpha, e));

            // Store Householder vector v = [v0, w[k+1], ..., w[m-1]] (scaled by 2^-e)
            V_idx[k].push_back(k);
            V_vals[k].push_back(v0);
            for (auto it = pat_k; it != pat.end(); ++it) {
                const size_type i = *it;
                if (i != k && w[i] != Value{0}) {
                    V_idx[k].push_back(i);
                    V_vals[k].push_back(down(w[i]));
                }
            }
        }

        // Index the new reflector by its rows (when it acts), then clear the
        // rows of w this column touched
        if (betas[k] != Value{0})
            for (const size_type r : V_idx[k]) refl_of_row[r].push_back(k);
        for (const size_type r : pat)
            w[r] = Value{0};
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
/// `Accumulator` (default `void`) is forwarded to both the numeric factorization
/// and the solve, so one policy covers the whole solve path (#535).
template <typename Accumulator = void, typename Value, typename Parameters, typename VecX, typename VecB,
          typename Ordering>
    requires OrderedField<Value>
void sparse_qr_solve(
    const mat::compressed2D<Value, Parameters>& A,
    VecX& x, const VecB& b,
    const Ordering& ordering)
{
    auto sym = sparse_qr_symbolic(A, ordering);
    auto num = sparse_qr_numeric<Accumulator>(A, sym);
    num.template solve<Accumulator>(x, b);
}

/// One-shot sparse QR solve without ordering.
/// `Accumulator` (default `void`) is forwarded to both the numeric factorization
/// and the solve, so one policy covers the whole solve path (#535).
template <typename Accumulator = void, typename Value, typename Parameters, typename VecX, typename VecB>
    requires OrderedField<Value>
void sparse_qr_solve(
    const mat::compressed2D<Value, Parameters>& A,
    VecX& x, const VecB& b)
{
    auto sym = sparse_qr_symbolic(A);
    auto num = sparse_qr_numeric<Accumulator>(A, sym);
    num.template solve<Accumulator>(x, b);
}

} // namespace mtl::sparse::factorization
