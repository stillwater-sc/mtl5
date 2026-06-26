#pragma once
// MTL5 -- Supernodal LU factorization with threshold partial pivoting.
// Phase 2 of the native-SuperLU effort (#182).
//
// Left-looking Gilbert-Peierls LU (PA = LU) that groups columns of L into
// SUPERNODES so the dominant elimination work becomes a dense block update
// (TRSV + GEMV) -- the granularity where mixed precision pays off. Supernodes
// are formed DYNAMICALLY during the pivoted factorization (pivoting precludes a
// fully-static structure), capped by the column-etree relaxed-supernode
// boundaries from Phase 1 (analysis/column_etree.hpp).
//
// The per-column reach DFS, Eisenstat-Liu symmetric pruning, threshold partial
// pivoting, CSC L/U layout, and solve() are taken from the scalar sparse_lu.hpp;
// the only upgrade is the inner update, which applies each already-closed
// supernode as one dense block through accumulator_traits (the current, still
// open supernode -- a few columns -- uses the proven scalar SAXPY). After the
// Phase-1 postorder, increasing column index is a valid topological order, so a
// supernode's columns are contiguous and a block update is correct.
//
// Scope (Phase 2): column-at-a-time target; the block update uses generic dense
// loops via accumulator_traits. Multi-column panels, BLAS-3-tuned kernels, and
// relaxed-supernode size tuning are Phase 3 (#183).
//
// References: Demmel/Eisenstat/Gilbert/Li/Liu, "A Supernodal Approach to Sparse
// Partial Pivoting", SIAM J. Matrix Anal. Appl. 20(3), 1999; Davis, "Direct
// Methods for Sparse Linear Systems", SIAM 2006.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/concepts/scalar.hpp>
#include <mtl/math/accumulator_traits.hpp>
#include <mtl/sparse/util/csc.hpp>
#include <mtl/sparse/util/permutation.hpp>
#include <mtl/sparse/analysis/column_etree.hpp>
#include <mtl/sparse/factorization/triangular_solve.hpp>
#include <mtl/sparse/factorization/sparse_lu.hpp>   // lu_symbolic conventions, accumulator_traits
#include <mtl/sparse/iterative_refine.hpp>

namespace mtl::sparse::factorization {

/// Symbolic analysis for supernodal LU: a fill-reducing column order fused with
/// the column-etree postorder (so supernode columns are contiguous), plus the
/// relaxed-supernode boundaries that cap dynamic supernode growth.
struct supernodal_lu_symbolic {
    std::size_t              n{0};
    std::vector<std::size_t> col_perm;     // q[new]=old (ordering then column-etree postorder)
    std::vector<std::size_t> col_pinv;     // inverse
    std::vector<std::size_t> col_super;    // col(final space) -> relaxed supernode id (cap)
};

/// Result of numeric supernodal LU factorization (same factor layout as
/// lu_numeric, plus the dynamic L-supernode boundaries).
template <typename Value>
struct supernodal_lu_factor {
    util::csc_matrix<Value>  L;            // unit lower (CSC), diagonal first
    util::csc_matrix<Value>  U;            // upper (CSC), diagonal last
    std::vector<std::size_t> row_perm;     // pivoting row order (p[new]=old)
    std::vector<std::size_t> row_pinv;     // inverse
    std::vector<std::size_t> lsuper_first; // dynamic L-supernode boundaries (size nsuper+1)
    std::vector<Value>       row_scale;    // r[orig row]; empty = unscaled (factored R*A)
    std::size_t              num_perturbed = 0; // pivots replaced by perturbation (0 = clean factor)
    supernodal_lu_symbolic   symbolic;

    std::size_t num_rows() const { return symbolic.n; }
    std::size_t num_cols() const { return symbolic.n; }
    std::size_t nsuper()   const { return lsuper_first.empty() ? 0 : lsuper_first.size() - 1; }

    /// Solve A*x = b.  P*A*Q = L*U (identical to lu_numeric::solve).
    template <typename VecX, typename VecB>
    void solve(VecX& x, const VecB& b) const {
        const std::size_t n = symbolic.n;
        if (static_cast<std::size_t>(x.size()) != n ||
            static_cast<std::size_t>(b.size()) != n) {
            throw std::invalid_argument(
                "supernodal_lu_factor::solve: vector size mismatch (expected "
                + std::to_string(n) + ")");
        }
        // If R*A was factored (row equilibration), the RHS is row-scaled by the
        // same R; x is unchanged.
        std::vector<Value> w(n);
        const bool scaled = !row_scale.empty();
        for (std::size_t i = 0; i < n; ++i) {
            const std::size_t orow = row_perm[i];
            w[i] = static_cast<Value>(b(orow)) * (scaled ? row_scale[orow] : Value{1});
        }
        dense_lower_solve(L, w);                               // L z = P (R b)
        dense_upper_solve(U, w);                               // U y = z
        for (std::size_t i = 0; i < n; ++i)
            x(symbolic.col_perm[i]) = static_cast<typename VecX::value_type>(w[i]);
    }
};

/// Symbolic supernodal LU analysis with a fill-reducing column ordering.
template <typename Value, typename Parameters, typename Ordering>
supernodal_lu_symbolic supernodal_lu_symbolic_analyze(
    const mat::compressed2D<Value, Parameters>& A,
    const Ordering& ordering)
{
    if (A.num_rows() != A.num_cols())
        throw std::invalid_argument("supernodal_lu_symbolic_analyze: matrix must be square");

    supernodal_lu_symbolic sym;
    sym.n = A.num_cols();

    auto cperm = ordering(A);                                  // e.g. COLAMD
    if (!util::is_valid_permutation(cperm) || cperm.size() != sym.n)
        throw std::invalid_argument("supernodal_lu_symbolic_analyze: ordering returned invalid permutation");

    // Column-etree postorder of the ordered matrix => contiguous supernodes.
    auto Aq = util::column_permute(A, cperm);
    auto sa = analysis::analyze_unsymmetric(Aq);
    sym.col_perm = util::compose_permutations(cperm, sa.col_perm);
    sym.col_pinv = util::invert_permutation(sym.col_perm);
    sym.col_super = std::move(sa.super);                       // relaxed-supernode id per final column
    return sym;
}

namespace detail {

/// One closed (finished) L-supernode kept densely for use as an update source.
template <typename Value>
struct lu_snode {
    std::size_t              sf = 0, w = 0, m = 0;  // first col, width, height
    std::vector<std::size_t> rows;                  // orig-row indices: [w pivots] ++ off-diagonal
    std::vector<Value>       block;                 // dense column-major m x w; diag block unit-lower
};

} // namespace detail

/// Numeric supernodal LU factorization with threshold partial pivoting.
///
/// \param threshold pivoting threshold in (0,1]; 1 = full partial pivoting.
/// \param max_super maximum supernode width (relaxation cap).
/// \param scale     opt-in row equilibration (factor R*A; RHS scaled in solve()).
/// \param pivot_perturb opt-in zero-pivot perturbation (default 0 = off). When
///                  > 0, a chosen pivot whose magnitude is below
///                  `pivot_perturb * ||column||_inf` is replaced (sign-preserving)
///                  by that floor and `num_perturbed` is incremented, instead of
///                  throwing -- keeping a low-precision factorization going when
///                  cancellation drives a structurally-nonzero pivot numerically
///                  to zero (clean up with iterative_refine). Default 0 is
///                  byte-identical to before (hard throw). Mirrors sparse_lu.
/// \throws std::runtime_error on a singular (zero) pivot when perturbation is off,
///                  or when the whole pivot column is numerically zero.
template <typename Value, typename Parameters, typename Accumulator = Value>
    requires OrderedField<Value>
supernodal_lu_factor<Value> supernodal_lu_numeric(
    const mat::compressed2D<Value, Parameters>& A,
    const supernodal_lu_symbolic& sym,
    Value threshold = Value{1},
    std::size_t max_super = 64,
    bool scale = false,
    Value pivot_perturb = Value{0})
{
    using size_type = std::size_t;
    using std::abs;
    using AT = accumulator_traits<Accumulator, Value>;
    const size_type n = sym.n;
    if (A.num_rows() != n || A.num_cols() != n)
        throw std::invalid_argument("supernodal_lu_numeric: dimensions do not match symbolic analysis");
    if (!(threshold > Value{0} && threshold <= Value{1}))
        throw std::invalid_argument("supernodal_lu_numeric: threshold must be in (0,1]");
    if (sym.col_perm.size() != n || sym.col_pinv.size() != n ||
        sym.col_super.size() != n || !util::is_valid_permutation(sym.col_perm))
        throw std::invalid_argument("supernodal_lu_numeric: invalid symbolic analysis");
    // Row/node indices are stored as int32 (memory-bandwidth choice inherited
    // from sparse_lu); fail safely rather than silently overflow on huge inputs.
    if (n > static_cast<size_type>(std::numeric_limits<std::int32_t>::max()))
        throw std::invalid_argument("supernodal_lu_numeric: dimension exceeds 32-bit index range");

    auto C = util::crs_to_csc(util::column_permute(A, sym.col_perm));

    // Optional row equilibration: factor R*A with r[row] = 1/max|A(row,:)|.
    // Improves pivot stability (notably for low-precision factorization); the
    // RHS is row-scaled by the same R in solve(), so x is unchanged.
    std::vector<Value> row_scale;
    if (scale) {
        std::vector<Value> rmax(n, Value{0});
        for (size_type p = 0; p < C.values.size(); ++p) {
            Value a = abs(C.values[p]);
            if (a > rmax[C.row_ind[p]]) rmax[C.row_ind[p]] = a;
        }
        row_scale.assign(n, Value{1});
        for (size_type r = 0; r < n; ++r)
            if (rmax[r] > Value{0}) row_scale[r] = Value{1} / rmax[r];
        for (size_type p = 0; p < C.values.size(); ++p)
            C.values[p] *= row_scale[C.row_ind[p]];
    }

    using idx32 = std::int32_t;
    std::vector<idx32>          pinv(n, -1);
    std::vector<Accumulator>    x(n);
    std::vector<idx32>          xi(n);
    std::vector<std::ptrdiff_t> pstack(n);
    std::vector<char>           marked(n, 0);
    std::vector<idx32>          reach_nodes; reach_nodes.reserve(n);
    std::vector<size_type>      Lp(n + 1, 0), Up(n + 1, 0), xprune(n, 0);
    std::vector<idx32> Li; std::vector<Value> Lx;
    std::vector<idx32> Ui; std::vector<Value> Ux;
    Li.reserve(C.values.size() * 2); Lx.reserve(C.values.size() * 2);
    Ui.reserve(C.values.size() * 2); Ux.reserve(C.values.size() * 2);

    std::vector<size_type> piv_row(n, 0);          // piv_row[col] = orig pivot row of that column

    // Supernode bookkeeping.
    std::vector<detail::lu_snode<Value>> snodes;   // closed supernodes
    std::vector<idx32> col_csuper(n, -1);          // col -> closed supernode id (or -1 if open/unset)
    std::vector<size_type> lsuper_first;           // boundaries, recorded as supernodes close
    std::size_t num_perturbed = 0;                 // count of perturbed pivots (pivot_perturb > 0)
    size_type open_sf = 0;                         // first column of the currently-open supernode
    std::vector<size_type> prev_cand;              // sorted orig candidate rows of column k-1

    std::vector<idx32> contrib_super;              // closed supernodes hitting column k (dedup)
    std::vector<char>  super_seen;                 // marker per closed supernode
    std::vector<Accumulator> segacc;               // supernode segment, accumulator precision
    std::vector<Value>       segval;               // pivots rounded to Value (multipliers)

    // Close the open supernode [open_sf, sl): build its dense block from the CSC
    // columns (pivots now known), using `offdiag` (orig rows below the diagonal
    // block) = the candidate rows of its last column.
    auto close_supernode = [&](size_type sl, const std::vector<size_type>& offdiag) {
        const size_type sf = open_sf;
        const size_type w = sl - sf;
        detail::lu_snode<Value> S;
        S.sf = sf; S.w = w;
        S.rows.reserve(w + offdiag.size());
        for (size_type c = sf; c < sl; ++c) S.rows.push_back(piv_row[c]);  // diagonal-block rows
        for (size_type r : offdiag) S.rows.push_back(r);                   // shared off-diagonal rows
        S.m = S.rows.size();
        S.block.assign(S.m * w, Value{0});
        // Local orig-row -> local index for this supernode.
        // (m is small per supernode; a flat scan keeps it allocation-free.)
        for (size_type c = 0; c < w; ++c) {
            const size_type col = sf + c;
            for (size_type p = Lp[col]; p < Lp[col + 1]; ++p) {
                const size_type orig = static_cast<size_type>(Li[p]);
                // find orig in S.rows (rows >= this column's diagonal only)
                for (size_type r = c; r < S.m; ++r)
                    if (S.rows[r] == orig) { S.block[c * S.m + r] = Lx[p]; break; }
            }
        }
        const idx32 id = static_cast<idx32>(snodes.size());
        for (size_type c = sf; c < sl; ++c) col_csuper[c] = id;
        snodes.push_back(std::move(S));
        super_seen.push_back(0);
        lsuper_first.push_back(sf);
    };

    for (size_type k = 0; k < n; ++k) {
        Lp[k] = Li.size();
        Up[k] = Ui.size();

        // --- reach(C(:,k)) in L: iterative DFS, topological order in xi[top..n) ---
        reach_nodes.clear();
        size_type top = n;
        for (size_type p = C.col_ptr[k]; p < C.col_ptr[k + 1]; ++p) {
            std::ptrdiff_t start = static_cast<std::ptrdiff_t>(C.row_ind[p]);
            if (marked[start]) continue;
            std::ptrdiff_t head = 0; xi[0] = start;
            while (head >= 0) {
                std::ptrdiff_t node = xi[head];
                std::ptrdiff_t jcol = pinv[node];
                if (!marked[node]) {
                    marked[node] = 1; reach_nodes.push_back(node);
                    pstack[head] = (jcol < 0) ? 0 : static_cast<std::ptrdiff_t>(Lp[jcol]);
                }
                bool done = true;
                std::ptrdiff_t pend = (jcol < 0) ? 0 : static_cast<std::ptrdiff_t>(xprune[jcol]);
                for (std::ptrdiff_t p2 = pstack[head]; p2 < pend; ++p2) {
                    std::ptrdiff_t child = Li[p2];
                    if (marked[child]) continue;
                    pstack[head] = p2; xi[++head] = child; done = false; break;
                }
                if (done) { --head; xi[--top] = node; }
            }
        }

        // --- scatter C(:,k) ---
        for (size_type p = top; p < n; ++p) AT::clear(x[xi[p]]);
        for (size_type p = C.col_ptr[k]; p < C.col_ptr[k + 1]; ++p)
            AT::assign(x[C.row_ind[p]], C.values[p]);

        // --- supernodal solve x = L \ C(:,k) ---
        // (a) closed supernodes (cols < open_sf): dense block update, in increasing
        //     supernode order. Collect distinct contributing supernodes first.
        contrib_super.clear();
        for (size_type px = top; px < n; ++px) {
            std::ptrdiff_t j = xi[px];
            std::ptrdiff_t jcol = pinv[j];
            if (jcol < 0) continue;
            idx32 sid = col_csuper[jcol];
            if (sid >= 0 && !super_seen[sid]) { super_seen[sid] = 1; contrib_super.push_back(sid); }
        }
        std::sort(contrib_super.begin(), contrib_super.end(),
                  [&](idx32 a, idx32 b) { return snodes[a].sf < snodes[b].sf; });
        for (idx32 sid : contrib_super) {
            const auto& S = snodes[sid];
            const size_type w = S.w, m = S.m;
            segacc.resize(w); segval.resize(w);
            for (size_type c = 0; c < w; ++c) segacc[c] = x[S.rows[c]];   // copy accumulator state
            // TRSV against the unit-lower diagonal block, accumulating in Acc;
            // each pivot is rounded to Value once (it then serves as a multiplier,
            // which is what accumulator_traits::add_product consumes).
            for (size_type c = 0; c < w; ++c) {
                segval[c] = AT::template value<Value>(segacc[c]);
                for (size_type r = c + 1; r < w; ++r)
                    AT::add_product(segacc[r], S.block[c * m + r], -segval[c]);
            }
            for (size_type c = 0; c < w; ++c) x[S.rows[c]] = segacc[c];   // write back (still Acc)
            // GEMV: off-diagonal rows (the dominant work, accumulated in Accumulator).
            for (size_type r = w; r < m; ++r) {
                Accumulator& dst = x[S.rows[r]];
                for (size_type c = 0; c < w; ++c)
                    AT::add_product(dst, -S.block[c * m + r], segval[c]);
            }
            super_seen[sid] = 0;
        }
        // (b) open supernode columns [open_sf, k): proven scalar SAXPY, reach order.
        for (size_type px = top; px < n; ++px) {
            std::ptrdiff_t j = xi[px];
            std::ptrdiff_t jcol = pinv[j];
            if (jcol < 0 || static_cast<size_type>(jcol) < open_sf) continue;  // closed -> handled above
            Value xj = AT::value(x[j]);
            for (size_type p = Lp[jcol] + 1; p < Lp[jcol + 1]; ++p)
                AT::add_product(x[Li[p]], -Lx[p], xj);
        }

        // --- threshold partial pivoting + emit U(:,k) for pivotal rows ---
        Value a = Value{0};                          // max |candidate|
        Value colnorm = Value{0};                    // ||column k||_inf over the reach
        std::ptrdiff_t ipiv = -1; bool have = false;
        for (size_type px = top; px < n; ++px) {
            std::ptrdiff_t i = xi[px];
            Value mag = abs(AT::value(x[i]));
            if (mag > colnorm) colnorm = mag;
            if (pinv[i] < 0) {
                if (!have || mag > a) { a = mag; ipiv = i; have = true; }
            } else {
                Ui.push_back(pinv[i]); Ux.push_back(AT::value(x[i]));
            }
        }
        const bool perturb = pivot_perturb > Value{0};
        if (ipiv < 0)                                // no candidate row: nothing to perturb
            throw std::runtime_error(
                "supernodal_lu_numeric: zero pivot at column " + std::to_string(k) + " (singular)");
        if (!perturb && a == Value{0})               // default: hard throw, unchanged
            throw std::runtime_error(
                "supernodal_lu_numeric: zero pivot at column " + std::to_string(k) + " (singular)");
        if (pinv[static_cast<std::ptrdiff_t>(k)] < 0
            && abs(AT::value(x[k])) >= threshold * a)
            ipiv = static_cast<std::ptrdiff_t>(k);

        Value pivot = AT::value(x[ipiv]);
        // Opt-in zero-pivot perturbation: replace a collapsed pivot (below
        // pivot_perturb * ||column||) sign-preservingly rather than failing. A
        // wholly-zero column (colnorm == 0) is genuinely singular -> still throw.
        if (perturb) {
            const Value floor = pivot_perturb * colnorm;
            if (abs(pivot) < floor) {
                pivot = (pivot < Value{0}) ? -floor : floor;
                ++num_perturbed;
            } else if (abs(pivot) == Value{0}) {
                throw std::runtime_error(
                    "supernodal_lu_numeric: zero pivot at column " + std::to_string(k)
                    + " (pivot column is numerically zero)");
            }
        }
        Ui.push_back(static_cast<idx32>(k)); Ux.push_back(pivot);
        pinv[ipiv] = static_cast<idx32>(k);
        piv_row[k] = static_cast<size_type>(ipiv);

        Li.push_back(static_cast<idx32>(ipiv)); Lx.push_back(Value{1});   // L(k,k)=1 first
        std::vector<size_type> cand;                                      // orig rows of L(:,k) below diag
        for (size_type px = top; px < n; ++px) {
            std::ptrdiff_t i = xi[px];
            if (pinv[i] < 0) {
                Li.push_back(static_cast<idx32>(i));
                Lx.push_back(AT::value(x[i]) / pivot);
                cand.push_back(static_cast<size_type>(i));
            }
            AT::clear(x[i]);
        }
        xprune[k] = Li.size();
        std::sort(cand.begin(), cand.end());

        // --- symmetric pruning (Eisenstat-Liu) ---
        {
            size_type u_diag = Ui.size() - 1;
            for (size_type up = Up[k]; up < u_diag; ++up) {
                size_type j = static_cast<size_type>(Ui[up]);
                if (xprune[j] != Lp[j + 1]) continue;
                bool symm = false;
                for (size_type p = Lp[j]; p < Lp[j + 1]; ++p)
                    if (Li[p] == ipiv) { symm = true; break; }
                if (!symm) continue;
                size_type kmin = Lp[j] + 1, kmax = Lp[j + 1] - 1;
                while (kmin <= kmax) {
                    if (pinv[Li[kmax]] < 0)        { --kmax; }
                    else if (pinv[Li[kmin]] >= 0)  { ++kmin; }
                    else { std::swap(Li[kmin], Li[kmax]); std::swap(Lx[kmin], Lx[kmax]); ++kmin; --kmax; }
                }
                xprune[j] = kmin;
            }
        }

        for (std::ptrdiff_t node : reach_nodes) marked[node] = 0;

        // --- dynamic supernode formation ---
        // Column k extends the open supernode iff its below-diagonal structure
        // nests under column k-1's (a true supernode), it stays inside the
        // Phase-1 relaxed boundary, and the width cap holds. Otherwise close
        // [open_sf, k) (its off-diagonal rows == column k-1's candidate rows).
        // In a supernode, column k's below-diagonal rows are column k-1's with
        // column k's OWN pivot removed: cand_k == cand_{k-1} \ {piv_k}.
        const size_type cur_pivot = static_cast<size_type>(ipiv);
        if (k > open_sf) {
            bool join =
                (k - open_sf) < max_super &&
                sym.col_super[k] == sym.col_super[k - 1] &&
                prev_cand.size() == cand.size() + 1;
            if (join) {
                size_type ci = 0;
                for (size_type r : prev_cand) {
                    if (r == cur_pivot) continue;        // the new diagonal
                    if (ci >= cand.size() || cand[ci] != r) { join = false; break; }
                    ++ci;
                }
                if (ci != cand.size()) join = false;     // cur_pivot wasn't in prev_cand
            }
            if (!join) { close_supernode(k, prev_cand); open_sf = k; }
        }
        prev_cand = std::move(cand);
    }
    Lp[n] = Li.size();
    Up[n] = Ui.size();
    if (n > 0) {
        close_supernode(n, prev_cand);             // close the final open supernode
        lsuper_first.push_back(n);
    } else {
        lsuper_first.push_back(0);                 // empty matrix: zero supernodes
    }

    // Remap L row indices to pivot positions.
    for (size_type p = 0; p < Li.size(); ++p) Li[p] = pinv[Li[p]];

    supernodal_lu_factor<Value> result;
    result.symbolic = sym;
    result.row_scale = std::move(row_scale);
    result.num_perturbed = num_perturbed;
    result.lsuper_first.assign(lsuper_first.begin(), lsuper_first.end());
    result.row_perm.resize(n); result.row_pinv.resize(n);
    for (size_type i = 0; i < n; ++i) {
        size_type pos = static_cast<size_type>(pinv[i]);
        result.row_pinv[i] = pos; result.row_perm[pos] = i;
    }

    std::vector<size_type> ord;
    auto to_csc = [&](const std::vector<size_type>& Mp, const std::vector<idx32>& Mi,
                      const std::vector<Value>& Mx) -> util::csc_matrix<Value> {
        util::csc_matrix<Value> M; M.nrows = n; M.ncols = n; M.col_ptr = Mp;
        M.row_ind.resize(Mi.size()); M.values.resize(Mx.size());
        for (size_type c = 0; c < n; ++c) {
            size_type b = Mp[c], e = Mp[c + 1];
            ord.resize(e - b);
            for (size_type t = 0; t < e - b; ++t) ord[t] = b + t;
            std::sort(ord.begin(), ord.end(), [&](size_type p, size_type q) { return Mi[p] < Mi[q]; });
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
/// reusing its analyze + pivot/supernode pattern. Only numeric values are
/// recomputed: no ordering, no reach DFS, no pivot search, no supernode
/// detection -- the fast path for repeated solves of a fixed pattern with
/// changing values (transient SPICE: one analyze+factor, many refactors). The
/// stored L/U columns are in topological (ascending-row) order, so the values
/// replay directly. Mirrors sparse_lu_refactor.
///
/// Preconditions: A uses prev's column ordering and the same nonzero pattern.
/// \throws std::runtime_error if a reused pivot is numerically zero for the new
///         values (the prior pivot sequence is no longer valid).
template <typename Value, typename Parameters>
    requires OrderedField<Value>
supernodal_lu_factor<Value> supernodal_lu_refactor(
    const mat::compressed2D<Value, Parameters>& A,
    const supernodal_lu_factor<Value>& prev)
{
    using size_type = std::size_t;
    const size_type n = prev.symbolic.n;
    if (A.num_rows() != n || A.num_cols() != n)
        throw std::invalid_argument(
            "supernodal_lu_refactor: matrix dimensions do not match prior factorization");

    using std::abs;
    auto C = util::crs_to_csc(util::column_permute(A, prev.symbolic.col_perm));
    const auto& pinv = prev.row_pinv;        // original row -> pivot position (fixed)
    supernodal_lu_factor<Value> result = prev;   // reuse pattern + perms + symbolic + supernodes
    result.num_perturbed = 0;                    // refactor replays values without perturbing (throws on zero pivot)

    // If the prior factorization was row-equilibrated, re-equilibrate the new
    // values (same pattern) and factor R*A; solve() row-scales the RHS by R.
    if (!prev.row_scale.empty()) {
        std::vector<Value> rmax(n, Value{0});
        for (size_type p = 0; p < C.values.size(); ++p) {
            Value a = abs(C.values[p]);
            if (a > rmax[C.row_ind[p]]) rmax[C.row_ind[p]] = a;
        }
        result.row_scale.assign(n, Value{1});
        for (size_type r = 0; r < n; ++r)
            if (rmax[r] > Value{0}) result.row_scale[r] = Value{1} / rmax[r];
        for (size_type p = 0; p < C.values.size(); ++p)
            C.values[p] *= result.row_scale[C.row_ind[p]];
    }

    auto& L = result.L;                      // overwrite values in place
    auto& U = result.U;

    std::vector<Value> x(n, Value{0});
    for (size_type k = 0; k < n; ++k) {
        // Scatter column k of the column-permuted A into the pivot-space workspace.
        for (size_type p = C.col_ptr[k]; p < C.col_ptr[k + 1]; ++p)
            x[pinv[C.row_ind[p]]] = C.values[p];

        // Left-looking updates from contributing columns j < k (U(:,k) is sorted
        // ascending with the diagonal last -> strict-upper entries are the
        // contributing columns in topological order).
        const size_type ubeg = U.col_ptr[k], uend = U.col_ptr[k + 1];
        for (size_type p = ubeg; p + 1 < uend; ++p) {
            const size_type j = U.row_ind[p];
            const Value xj = x[j];
            if (xj == Value{0}) continue;
            for (size_type q = L.col_ptr[j] + 1; q < L.col_ptr[j + 1]; ++q)  // skip unit diagonal
                x[L.row_ind[q]] -= L.values[q] * xj;
        }

        const Value pivot = x[k];
        if (pivot == Value{0})
            throw std::runtime_error(
                "supernodal_lu_refactor: zero pivot at column " + std::to_string(k)
                + " (prior pivot sequence invalid for the new values)");

        for (size_type p = ubeg; p < uend; ++p) U.values[p] = x[U.row_ind[p]];
        for (size_type p = L.col_ptr[k]; p < L.col_ptr[k + 1]; ++p) {
            const size_type r = L.row_ind[p];
            L.values[p] = (r == k) ? Value{1} : x[r] / pivot;
        }

        // Clear the workspace over the touched positions (U/L pattern and the
        // scattered A column, in case A has an entry outside the prior pattern).
        for (size_type p = ubeg; p < uend; ++p)                 x[U.row_ind[p]] = Value{0};
        for (size_type p = L.col_ptr[k]; p < L.col_ptr[k + 1]; ++p) x[L.row_ind[p]] = Value{0};
        for (size_type p = C.col_ptr[k]; p < C.col_ptr[k + 1]; ++p) x[pinv[C.row_ind[p]]] = Value{0};
    }
    return result;
}

/// One-shot supernodal LU solve with a fill-reducing column ordering.
template <typename Value, typename Parameters, typename VecX, typename VecB, typename Ordering>
    requires OrderedField<Value>
void supernodal_lu_solve(const mat::compressed2D<Value, Parameters>& A,
                         VecX& x, const VecB& b, const Ordering& ordering,
                         Value threshold = Value{1})
{
    auto sym = supernodal_lu_symbolic_analyze(A, ordering);
    auto fac = supernodal_lu_numeric(A, sym, threshold);
    fac.solve(x, b);
}

/// Mixed-precision solve with iterative refinement: factor in `FactorValue`
/// (e.g. float) with accumulation in `Accumulator`, then refine the residual in
/// `Residual` (e.g. double) through the low-precision factor. `FactorValue` is
/// the only required template argument.
template <typename FactorValue, typename Accumulator = FactorValue,
          typename Residual, typename Parameters, typename Ordering>
refine_result supernodal_lu_solve_refined(
    const mat::compressed2D<Residual, Parameters>& A,
    vec::dense_vector<Residual>& x,
    const vec::dense_vector<Residual>& b,
    const Ordering& ordering,
    const refine_options& opt = {})
{
    using size_type = typename mat::compressed2D<Residual, Parameters>::size_type;
    // Re-type the sparse matrix into the factor precision, preserving sparsity
    // (mtl::convert is for dense tensors and would densify a compressed2D).
    const std::size_t nnz = A.nnz();
    std::vector<size_type> starts(A.ref_major().begin(), A.ref_major().end());
    std::vector<size_type> idx(A.ref_minor().begin(), A.ref_minor().end());
    std::vector<FactorValue> dat(nnz);
    const auto& src = A.ref_data();
    for (std::size_t kk = 0; kk < nnz; ++kk) dat[kk] = static_cast<FactorValue>(src[kk]);
    mat::compressed2D<FactorValue, Parameters> Af(A.num_rows(), A.num_cols(), nnz,
                                                  starts.data(), idx.data(), dat.data());

    auto sym = supernodal_lu_symbolic_analyze(Af, ordering);
    auto fac = supernodal_lu_numeric<FactorValue, Parameters, Accumulator>(Af, sym);
    for (std::size_t i = 0; i < static_cast<std::size_t>(x.size()); ++i)
        x(static_cast<int>(i)) = Residual{0};
    return iterative_refine(A, fac, b, x, opt);
}

} // namespace mtl::sparse::factorization
