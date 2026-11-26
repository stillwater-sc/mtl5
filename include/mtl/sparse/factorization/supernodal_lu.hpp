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
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <type_traits>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/concepts/scalar.hpp>
#include <mtl/math/accumulator_traits.hpp>
#include <mtl/sparse/util/csc.hpp>
#include <mtl/sparse/util/permutation.hpp>
#include <mtl/sparse/analysis/column_etree.hpp>
#include <mtl/sparse/factorization/triangular_solve.hpp>
#include <mtl/sparse/factorization/level_schedule.hpp>
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

/// The default `panel` of supernodal_lu_numeric (#195): the columns whose
/// updates from the supernodes closed before them are applied together, so
/// each supernode block is read once per panel instead of once per column.
/// 1 is the column-at-a-time update; at most 64.
inline constexpr std::size_t supernodal_lu_default_panel = 16;


/// Result of numeric supernodal LU factorization (same factor layout as
/// lu_numeric, plus the dynamic L-supernode boundaries).
template <typename Value>
struct supernodal_lu_factor {
    std::vector<std::size_t> row_perm;     // pivoting row order (p[new]=old)
    std::vector<std::size_t> row_pinv;     // inverse
    std::vector<std::size_t> lsuper_first; // dynamic L-supernode boundaries (size nsuper+1)
    std::vector<Value>       row_scale;    // r[orig row]; empty = unscaled (factored R*A)
    std::size_t              num_perturbed = 0; // pivots replaced by perturbation (0 = clean factor)
    /// Supernode updates the panel pre-passes applied (#195): one per closed
    /// supernode per panel that took it. 0 when panels never ran (panel = 1,
    /// or the gate kept them off). A diagnostic: the factor is the same either
    /// way, and tests use it to know the panel path ran (issue 926).
    std::size_t              panel_updates = 0;
    supernodal_lu_symbolic   symbolic;

    std::size_t num_rows() const { return symbolic.n; }
    std::size_t num_cols() const { return symbolic.n; }
    std::size_t nsuper()   const { return lsuper_first.empty() ? 0 : lsuper_first.size() - 1; }

    /// Read-only access to the unit lower factor L and the upper factor U (CSC).
    const util::csc_matrix<Value>& factorL() const { return L_; }
    const util::csc_matrix<Value>& factorU() const { return U_; }

    /// Install the factors (unit lower L diagonal-first, upper U diagonal-last)
    /// and (re)build the coupled forward (L z = w) and upper (U y = z) solve
    /// schedules from them. L, U and the schedules are always set together, so the
    /// schedules' cached structure always matches the factors' patterns. Strongly
    /// exception safe: the schedules are built into locals before any member is
    /// replaced. Requires `symbolic` installed first (validates against symbolic.n).
    void set_factor(util::csc_matrix<Value> L, util::csc_matrix<Value> U) {
        const std::size_t n = symbolic.n;
        if (static_cast<std::size_t>(L.ncols) != n || static_cast<std::size_t>(L.nrows) != n ||
            static_cast<std::size_t>(U.ncols) != n || static_cast<std::size_t>(U.nrows) != n)
            throw std::invalid_argument(
                "supernodal_lu_factor::set_factor: factor dimensions do not match "
                "the symbolic analysis");
        lower_solve_schedule fsched = build_lower_solve_schedule(L);   // may throw
        upper_solve_schedule usched = build_upper_solve_schedule(U);   // may throw
        // The diagonals are checked here, once, so the solves skip the pass
        // (diagonal_check::done_by_caller, issue 926).
        detail::require_nonzero_diagonal(L, false, L.values, "supernodal_lu_factor::set_factor");   // may throw
        detail::require_nonzero_diagonal(U, true, U.values, "supernodal_lu_factor::set_factor");    // may throw
        L_ = std::move(L);            // noexcept
        U_ = std::move(U);            // noexcept
        fwd_sched_ = std::move(fsched);  // noexcept
        up_sched_  = std::move(usched);  // noexcept
        has_factor_ = true;              // commit last: solve() is now valid
    }

    /// True once factors (and their coupled schedules) have been installed.
    bool has_factor() const { return has_factor_; }

    /// Install recomputed VALUES on the already-installed factor pattern, keeping
    /// the existing solve schedules (see lu_numeric::set_factor_values). The
    /// schedules cache positions into L_.values / U_.values, not the values, so a
    /// same-pattern refactorization skips the O(nnz) rebuild set_factor pays.
    /// Requires a factor already installed and value arrays of exactly the
    /// installed nnz; the schedules' built_nnz staleness guards are re-checked, so
    /// a pattern change cannot slip through (use set_factor for that).
    /// Strongly exception safe: all validation precedes any member replacement.
    void set_factor_values(std::vector<Value> Lvals, std::vector<Value> Uvals) {
        if (!has_factor_)
            throw std::logic_error(
                "supernodal_lu_factor::set_factor_values: no factor installed "
                "(call set_factor first)");
        if (Lvals.size() != L_.values.size() || Uvals.size() != U_.values.size() ||
            fwd_sched_.built_nnz != Lvals.size() || up_sched_.built_nnz != Uvals.size())
            throw std::invalid_argument(
                "supernodal_lu_factor::set_factor_values: value count does not match the "
                "installed factor pattern (use set_factor for a pattern change)");
        detail::require_nonzero_diagonal(L_, false, Lvals, "supernodal_lu_factor::set_factor_values");   // may throw
        detail::require_nonzero_diagonal(U_, true, Uvals, "supernodal_lu_factor::set_factor_values");    // may throw
        L_.values = std::move(Lvals);   // noexcept
        U_.values = std::move(Uvals);   // noexcept
    }

    /// Solve A*x = b.  P*A*Q = L*U (identical to lu_numeric::solve).
    /// `Accumulator` (default `void`, leading as in the numeric phase) selects
    /// the accumulator policy of the triangular-solve row reductions (#535):
    /// `void` is the raw Value loop; see level_schedule.hpp for the contract.
    template <typename Accumulator = void, typename VecX, typename VecB>
    void solve(VecX& x, const VecB& b) const {
        // A caller can set `symbolic` (n > 0) yet never install factors; the
        // triangular solves below would then read empty factor arrays. Reject
        // that explicitly instead of walking off the end.
        if (!has_factor_)
            throw std::logic_error(
                "supernodal_lu_factor::solve: no factor installed (call set_factor "
                "or supernodal_lu_numeric first)");
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
        // L z = P (R b), then U y = z. Level-scheduled (parallel, bit-identical to
        // dense_lower_solve / dense_upper_solve); serial at MTL5_NUM_THREADS=1.
        level_scheduled_lower_solve<Accumulator>(L_, fwd_sched_, w, diagonal_check::done_by_caller);
        level_scheduled_upper_solve<Accumulator>(U_, up_sched_, w, diagonal_check::done_by_caller);
        for (std::size_t i = 0; i < n; ++i)
            x(symbolic.col_perm[i]) = static_cast<typename VecX::value_type>(w[i]);
    }

private:
    util::csc_matrix<Value> L_;          // unit lower (CSC), diagonal first
    util::csc_matrix<Value> U_;          // upper (CSC), diagonal last
    lower_solve_schedule    fwd_sched_;  // forward (L z = w) schedule, bound to L_
    upper_solve_schedule    up_sched_;   // upper  (U y = z) schedule, bound to U_
    bool                    has_factor_ = false;  // set by set_factor; solve() requires it
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
template <typename Accumulator = void, typename Value, typename Parameters>
    requires OrderedField<Value>
supernodal_lu_factor<Value> supernodal_lu_numeric(
    const mat::compressed2D<Value, Parameters>& A,
    const supernodal_lu_symbolic& sym,
    Value threshold = Value{1},
    std::size_t max_super = 64,
    bool scale = false,
    Value pivot_perturb = Value{0},
    std::size_t panel = supernodal_lu_default_panel)
{
    // void (the default) accumulates in Value, as the former `Accumulator = Value` did.
    using Acc = std::conditional_t<std::is_void_v<Accumulator>, Value, Accumulator>;
    using size_type = std::size_t;
    using std::abs;
    using AT = accumulator_traits<Acc, Value>;
    const size_type n = sym.n;
    // Row/node indices are stored as int32 (memory-bandwidth choice inherited
    // from sparse_lu); fail safely rather than silently overflow on huge inputs.
    // Checked first, so a test reaches it without an A of that size (issue 926).
    if (n > static_cast<size_type>(std::numeric_limits<std::int32_t>::max()))
        throw std::invalid_argument("supernodal_lu_numeric: dimension exceeds 32-bit index range");
    if (A.num_rows() != n || A.num_cols() != n)
        throw std::invalid_argument("supernodal_lu_numeric: dimensions do not match symbolic analysis");
    if (!(threshold > Value{0} && threshold <= Value{1}))
        throw std::invalid_argument("supernodal_lu_numeric: threshold must be in (0,1]");
    if (sym.col_perm.size() != n || sym.col_pinv.size() != n ||
        sym.col_super.size() != n || !util::is_valid_permutation(sym.col_perm))
        throw std::invalid_argument("supernodal_lu_numeric: invalid symbolic analysis");
    if (panel < 1 || panel > 64)
        throw std::invalid_argument("supernodal_lu_numeric: panel must be in [1, 64]");

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
    std::vector<Acc>    x(n);
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
    std::size_t panel_updates = 0;                 // supernode updates applied by panel pre-passes
    size_type open_sf = 0;                         // first column of the currently-open supernode
    std::vector<size_type> prev_cand;              // sorted orig candidate rows of column k-1

    std::vector<idx32> contrib_super;              // closed supernodes hitting column k (dedup)
    std::vector<char>  super_seen;                 // marker per closed supernode
    std::vector<Acc> segacc;               // supernode segment, accumulator precision
    std::vector<Acc> gemv_tmp;             // a supernode's off-diagonal rows, gathered (#195)
    std::vector<idx32> row_local(n, -1);   // orig row -> its index in the supernode being closed
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
        // Local orig-row -> local index for this supernode, through row_local
        // (#195): it searched S.rows linearly for every entry, O(w * nnz * m)
        // per supernode, 17-18% of the factorization of wang3 and raefsky3.
        // The rows of a supernode are distinct; an entry lands at its row's
        // index when that is at or below its column's diagonal, as before.
        for (size_type r = 0; r < S.m; ++r) row_local[S.rows[r]] = static_cast<idx32>(r);
        for (size_type c = 0; c < w; ++c) {
            const size_type col = sf + c;
            for (size_type p = Lp[col]; p < Lp[col + 1]; ++p) {
                const idx32 r = row_local[static_cast<size_type>(Li[p])];
                if (r >= 0 && static_cast<size_type>(r) >= c) S.block[c * S.m + static_cast<size_type>(r)] = Lx[p];
            }
        }
        for (size_type r = 0; r < S.m; ++r) row_local[S.rows[r]] = -1;
        const idx32 id = static_cast<idx32>(snodes.size());
        for (size_type c = sf; c < sl; ++c) col_csuper[c] = id;
        snodes.push_back(std::move(S));
        super_seen.push_back(0);
        lsuper_first.push_back(sf);
    };

    // reach(C(:,col)) in L: iterative DFS, topological order in xi[top..n);
    // returns top. The caller unmarks reach_nodes.
    auto dfs_reach = [&](size_type col) -> size_type {
        reach_nodes.clear();
        size_type top = n;
        for (size_type p = C.col_ptr[col]; p < C.col_ptr[col + 1]; ++p) {
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
        return top;
    };

    // Panel state (#195). For the panel [panel_k0, panel_k1), panel_x holds
    // each column's accumulator (column t at t * n), with C(:,col) scattered
    // and the updates from every supernode closed before the panel (those
    // with sf < panel_open_sf) applied. Paths in L go only to later columns,
    // so the part of a panel column's reach before the panel is known before
    // any panel column is factored, and is the reach a DFS finds now.
    const size_type W = panel;
    // Allocated by the first pre-pass, not up front: 16 n accumulators (128 MB
    // for double at n = 1e6) were allocated and cleared even when the gate kept
    // panels off for the whole factorization (issue 926).
    std::vector<Acc> panel_x;
    std::vector<std::uint64_t> super_mask;          // per closed supernode: panel columns it updates
    std::vector<idx32> panel_super;
    std::vector<Acc> panel_tmp;                     // gathered off-diagonal rows, column t at t * mo
    std::vector<Acc> panel_seg;                     // TRSV state, column t at t * w
    std::vector<Value> panel_segval;
    size_type panel_k0 = 0, panel_k1 = 0, panel_open_sf = 0;
    // Panels pay only when the block reads they save outweigh the pre-pass:
    // a second DFS per column and the panel scatter. On the
    // convection-diffusion grids, with small supernodes and short reaches,
    // they made the factorization 15-20% slower (#195). So each pre-pass
    // compares the block entries it saved, sum (columns - 1) * w * m over its
    // supernodes, with its reach, sum of the panel columns' reach sizes; when
    // the saving is below panel_min_gain times the reach, the next panel_skip
    // panels run column at a time, then it probes again. Either way a column's
    // factor is the same. Measured on the i7-12700K: 128 and 16 kept the gains
    // on wang3, wang4 and raefsky3 and cut the grids' cost to 3-5% (the probes);
    // 1024 lost wang3's gain, and probing every 32 panels lost raefsky3's.
    constexpr double panel_min_gain = 128;
    constexpr size_type panel_skip = 16;
    double panel_saved = 0.0, panel_reach = 0.0;   // the last pre-pass's
    size_type panels_off = 0;        // panels left to run column at a time
    bool panel_active = false;
    auto panel_prepass = [&](size_type k0) {
        if (panel_x.empty()) {
            panel_x.resize(n * W);
            for (auto& v : panel_x) AT::clear(v);
        }
        panel_k0 = k0; panel_k1 = std::min(n, k0 + W); panel_open_sf = open_sf;
        panel_saved = 0.0; panel_reach = 0.0;
        super_mask.resize(snodes.size(), 0);
        panel_super.clear();
        for (size_type t = 0; t < panel_k1 - panel_k0; ++t) {
            const size_type col = panel_k0 + t;
            const size_type top = dfs_reach(col);
            panel_reach += double(n - top);
            for (size_type px = top; px < n; ++px) {
                const std::ptrdiff_t jcol = pinv[xi[px]];
                if (jcol < 0 || static_cast<size_type>(jcol) >= panel_open_sf) continue;
                const idx32 sid = col_csuper[jcol];
                if (super_mask[sid] == 0) panel_super.push_back(sid);
                super_mask[sid] |= std::uint64_t{1} << t;
            }
            for (std::ptrdiff_t node : reach_nodes) marked[node] = 0;
            Acc* xt = panel_x.data() + t * n;
            for (size_type p = C.col_ptr[col]; p < C.col_ptr[col + 1]; ++p)
                AT::assign(xt[C.row_ind[p]], C.values[p]);
        }
        // Each supernode in ascending sf, as the column-at-a-time update takes
        // them; per column the same TRSV, and a GEMV fused over the panel
        // columns it updates, so each block element is read once per panel.
        // Every accumulator takes the same terms in the same order as before.
        std::sort(panel_super.begin(), panel_super.end(),
                  [&](idx32 a, idx32 b) { return snodes[a].sf < snodes[b].sf; });
        std::size_t cols[64];
        for (idx32 sid : panel_super) {
            ++panel_updates;
            const auto& S = snodes[sid];
            const size_type w = S.w, m = S.m, mo = m - w;
            size_type nt = 0;
            for (std::uint64_t mask = super_mask[sid]; mask; mask &= mask - 1)
                cols[nt++] = static_cast<std::size_t>(std::countr_zero(mask));
            panel_saved += double(nt - 1) * double(w) * double(m);
            super_mask[sid] = 0;
            panel_seg.resize(nt * w); panel_segval.resize(nt * w); panel_tmp.resize(nt * mo);
            for (size_type q = 0; q < nt; ++q) {
                Acc* xt = panel_x.data() + cols[q] * n;
                Acc* sa = panel_seg.data() + q * w;
                Value* sv = panel_segval.data() + q * w;
                for (size_type c = 0; c < w; ++c) sa[c] = xt[S.rows[c]];
                for (size_type c = 0; c < w; ++c) {
                    sv[c] = AT::template value<Value>(sa[c]);
                    for (size_type r = c + 1; r < w; ++r)
                        AT::add_product(sa[r], S.block[c * m + r], -sv[c]);
                }
                for (size_type c = 0; c < w; ++c) xt[S.rows[c]] = sa[c];
                Acc* tq = panel_tmp.data() + q * mo;
                for (size_type r = 0; r < mo; ++r) tq[r] = xt[S.rows[w + r]];
            }
            // tmp(:, q) -= block(w:m, :) * segval(:, q), 8 rows of a column at a
            // time held across all w terms and stored once, as a GEMM tile does:
            // the accumulators were loaded and stored once per block column.
            // Each accumulator still adds its terms in ascending c.
            constexpr size_type R = 8;
            const Value* blk = S.block.data() + w;
            size_type r0 = 0;
            for (; r0 + R <= mo; r0 += R)
                for (size_type q = 0; q < nt; ++q) {
                    Acc* tq = panel_tmp.data() + q * mo + r0;
                    const Value* sv = panel_segval.data() + q * w;
                    Acc acc[R];
                    for (size_type i = 0; i < R; ++i) acc[i] = tq[i];
                    for (size_type c = 0; c < w; ++c) {
                        const Value sc = sv[c];
                        const Value* col = blk + c * m + r0;
                        for (size_type i = 0; i < R; ++i) AT::add_product(acc[i], -col[i], sc);
                    }
                    for (size_type i = 0; i < R; ++i) tq[i] = acc[i];
                }
            for (; r0 < mo; ++r0)
                for (size_type q = 0; q < nt; ++q) {
                    Acc a = panel_tmp[q * mo + r0];
                    const Value* sv = panel_segval.data() + q * w;
                    for (size_type c = 0; c < w; ++c) AT::add_product(a, -blk[c * m + r0], sv[c]);
                    panel_tmp[q * mo + r0] = a;
                }
            for (size_type q = 0; q < nt; ++q) {
                Acc* xt = panel_x.data() + cols[q] * n;
                const Acc* tq = panel_tmp.data() + q * mo;
                for (size_type r = 0; r < mo; ++r) xt[S.rows[w + r]] = tq[r];
            }
        }
    };

    for (size_type k = 0; k < n; ++k) {
        Lp[k] = Li.size();
        Up[k] = Ui.size();
        if (W > 1 && k == panel_k1) {
            if (k == 0) {
                // No supernode is closed before column 0, so a pre-pass there
                // can apply nothing, and its second DFS per column and the
                // gate's verdict on it (panels off for the next 16) were pure
                // cost (issue 926). The first probe is at k = W.
                panel_active = false;
                panel_k0 = 0; panel_k1 = std::min(n, W);
            } else if (panels_off == 0) {
                panel_prepass(k);
                panel_active = true;
                if (panel_saved < panel_min_gain * panel_reach) panels_off = panel_skip;
            } else {
                --panels_off;
                panel_active = false;
                panel_k0 = k; panel_k1 = std::min(n, k + W);
            }
        }
        const bool in_panel = W > 1 && panel_active;

        // --- reach(C(:,k)) in L: iterative DFS, topological order in xi[top..n) ---
        const size_type top = dfs_reach(k);

        // --- scatter C(:,k) --- or, in a panel, take the panel's accumulator,
        // C(:,k) with the pre-panel supernodes applied, and clear it there.
        if (in_panel) {
            Acc* xt = panel_x.data() + (k - panel_k0) * n;
            for (size_type p = top; p < n; ++p) {
                const auto i = static_cast<size_type>(xi[p]);
                x[i] = xt[i];
                AT::clear(xt[i]);
            }
        } else {
            for (size_type p = top; p < n; ++p) AT::clear(x[xi[p]]);
            for (size_type p = C.col_ptr[k]; p < C.col_ptr[k + 1]; ++p)
                AT::assign(x[C.row_ind[p]], C.values[p]);
        }

        // --- supernodal solve x = L \ C(:,k) ---
        // (a) closed supernodes (cols < open_sf): dense block update, in increasing
        //     supernode order. Collect distinct contributing supernodes first.
        contrib_super.clear();
        for (size_type px = top; px < n; ++px) {
            std::ptrdiff_t j = xi[px];
            std::ptrdiff_t jcol = pinv[j];
            if (jcol < 0) continue;
            idx32 sid = col_csuper[jcol];
            if (in_panel && static_cast<size_type>(jcol) < panel_open_sf) continue;   // applied in the pre-pass
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
            // Gather the m - w destination rows into a dense temporary, sweep the
            // column-major block one column at a time (contiguous, vectorizable),
            // and scatter back (#195). It walked the block a row at a time, w
            // terms at stride m, a cache line per term. Each row still adds its
            // terms in ascending c, so the result is the same.
            const size_type mo = m - w;
            gemv_tmp.resize(mo);
            for (size_type r = 0; r < mo; ++r) gemv_tmp[r] = x[S.rows[w + r]];
            for (size_type c = 0; c < w; ++c) {
                const Value sc = segval[c];
                const Value* col = S.block.data() + c * m + w;
                for (size_type r = 0; r < mo; ++r) AT::add_product(gemv_tmp[r], -col[r], sc);
            }
            for (size_type r = 0; r < mo; ++r) x[S.rows[w + r]] = gemv_tmp[r];
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
        // Prefer the diagonal of C(:,k) = A(:, q[k]), which is row q[k] in the
        // original numbering, not row k (issue 916).
        // A zero diagonal never displaces a nonzero maximum, even when
        // threshold * a underflows to zero.
        const auto kd = static_cast<std::ptrdiff_t>(sym.col_perm[k]);
        if (pinv[kd] < 0) {
            const Value dmag = abs(AT::value(x[kd]));
            if (dmag >= threshold * a && (dmag != Value{0} || a == Value{0}))
                ipiv = kd;
        }

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
    result.panel_updates = panel_updates;
    result.lsuper_first.assign(lsuper_first.begin(), lsuper_first.end());
    result.row_perm.resize(n); result.row_pinv.resize(n);
    for (size_type i = 0; i < n; ++i) {
        size_type pos = static_cast<size_type>(pinv[i]);
        result.row_pinv[i] = pos; result.row_perm[pos] = i;
    }

    // The columns of L and U, in reach order, into CSC with ascending rows: a
    // transpose to row-major and back, O(nnz) (#195). It sorted each column
    // through an index array, 10-21% of the factorization of wang3 and
    // raefsky3. A column's rows are distinct, so the order is the same.
    std::vector<size_type> tp, tcol, cnt;
    std::vector<Value> tval;
    auto to_csc = [&](const std::vector<size_type>& Mp, const std::vector<idx32>& Mi,
                      const std::vector<Value>& Mx) -> util::csc_matrix<Value> {
        const size_type nz = Mi.size();
        // Column-major (Mp, Mi) -> row-major (tp, tcol, tval): row r's entries in
        // ascending column order.
        tp.assign(n + 1, 0);
        for (size_type q = 0; q < nz; ++q) ++tp[static_cast<size_type>(Mi[q]) + 1];
        for (size_type r = 0; r < n; ++r) tp[r + 1] += tp[r];
        tcol.resize(nz); tval.resize(nz);
        cnt.assign(tp.begin(), tp.end() - 1);
        for (size_type c = 0; c < n; ++c)
            for (size_type q = Mp[c]; q < Mp[c + 1]; ++q) {
                const size_type dst = cnt[static_cast<size_type>(Mi[q])]++;
                tcol[dst] = c; tval[dst] = Mx[q];
            }
        // And back: column c's entries in ascending row order.
        util::csc_matrix<Value> M; M.nrows = n; M.ncols = n; M.col_ptr = Mp;
        M.row_ind.resize(nz); M.values.resize(nz);
        cnt.assign(Mp.begin(), Mp.end() - 1);
        for (size_type r = 0; r < n; ++r)
            for (size_type q = tp[r]; q < tp[r + 1]; ++q) {
                const size_type dst = cnt[tcol[q]]++;
                M.row_ind[dst] = r; M.values[dst] = tval[q];
            }
        return M;
    };
    // Install L and U and build the coupled solve schedules atomically.
    result.set_factor(to_csc(Lp, Li, Lx), to_csc(Up, Ui, Ux));
    return result;
}

/// Refactorize IN PLACE: recompute `fac`'s numeric values from a matrix with the
/// SAME sparsity pattern, reusing its analyze + pivot/supernode pattern -- no
/// ordering, no reach DFS, no pivot search, no supernode detection -- and install
/// them while KEEPING the factor pattern and the coupled solve schedules (#310).
/// The allocation-light fast path for repeated solves of a fixed pattern with
/// changing values (transient SPICE: one analyze+factor, many refactors); nothing
/// is copied but the recomputed values. The stored L/U columns are in topological
/// (ascending-row) order, so the values replay directly. Mirrors
/// sparse_lu_refactor_in_place.
///
/// Preconditions: A uses fac's column ordering, and its nonzero pattern lies
/// inside the factor's (A's at factor time, plus fill). Checked: an entry outside
/// it throws std::invalid_argument.
/// Strong exception guarantee: values are computed into scratch arrays and
/// installed only after the last column succeeds, so a throw leaves `fac` exactly
/// as it was.
/// \throws std::runtime_error if a reused pivot is numerically zero for the new
///         values (the prior pivot sequence is no longer valid).
template <typename Value, typename Parameters>
    requires OrderedField<Value>
void supernodal_lu_refactor_in_place(
    const mat::compressed2D<Value, Parameters>& A,
    supernodal_lu_factor<Value>& fac)
{
    using size_type = std::size_t;
    const size_type n = fac.symbolic.n;
    if (A.num_rows() != n || A.num_cols() != n)
        throw std::invalid_argument(
            "supernodal_lu_refactor: matrix dimensions do not match prior factorization");
    // The replay below indexes fac's factor pattern; with no factor installed
    // those arrays are empty and the walk would run off the end.
    if (!fac.has_factor())
        throw std::logic_error(
            "supernodal_lu_refactor: prior factorization has no factor installed");

    using std::abs;
    auto C = util::crs_to_csc(util::column_permute(A, fac.symbolic.col_perm));
    const auto& pinv = fac.row_pinv;         // original row -> pivot position (fixed)

    // If the prior factorization was row-equilibrated, re-equilibrate the new
    // values (same pattern) and factor R*A; solve() row-scales the RHS by R.
    std::vector<Value> row_scale;            // empty unless fac was equilibrated
    if (!fac.row_scale.empty()) {
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

    // The installed factors' PATTERN is reused verbatim (read-only); only the
    // values are recomputed, into scratch arrays laid out on that pattern. Every
    // position is written by the column that owns it before any later column reads
    // it, so the zero-initialization is just placeholder storage. Installing these
    // via set_factor_values keeps the existing solve schedules (value-agnostic,
    // and valid for an unchanged pattern) instead of rebuilding them. Throws on a
    // zero pivot; no perturbation.
    const util::csc_matrix<Value>& L = fac.factorL();
    const util::csc_matrix<Value>& U = fac.factorU();
    std::vector<Value> Lx(L.values.size(), Value{0});
    std::vector<Value> Ux(U.values.size(), Value{0});

    std::vector<Value> x(n, Value{0});
    std::vector<size_type> mark(n, n);   // mark[i] == k: row i is in L(:,k) or U(:,k)
    for (size_type k = 0; k < n; ++k) {
        // An entry of A outside the factor's pattern would be scattered and never
        // consumed, a wrong factor with no error (issue 864). Fill positions are
        // fine: U(:,k)'s pattern is closed under the reach in L.
        for (size_type p = U.col_ptr[k]; p < U.col_ptr[k + 1]; ++p) mark[U.row_ind[p]] = k;
        for (size_type p = L.col_ptr[k]; p < L.col_ptr[k + 1]; ++p) mark[L.row_ind[p]] = k;
        for (size_type p = C.col_ptr[k]; p < C.col_ptr[k + 1]; ++p) {
            if (mark[pinv[C.row_ind[p]]] != k) {
                throw std::invalid_argument(
                    "supernodal_lu_refactor: A has an entry in row "
                    + std::to_string(C.row_ind[p]) + ", column "
                    + std::to_string(fac.symbolic.col_perm[k])
                    + " outside the prior factor's pattern; factor it afresh");
            }
        }

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
            // j < k, so Lx(:,j) already holds this refactorization's values.
            for (size_type q = L.col_ptr[j] + 1; q < L.col_ptr[j + 1]; ++q)  // skip unit diagonal
                x[L.row_ind[q]] -= Lx[q] * xj;
        }

        const Value pivot = x[k];
        if (pivot == Value{0})
            throw std::runtime_error(
                "supernodal_lu_refactor: zero pivot at column " + std::to_string(k)
                + " (prior pivot sequence invalid for the new values)");

        for (size_type p = ubeg; p < uend; ++p) Ux[p] = x[U.row_ind[p]];
        for (size_type p = L.col_ptr[k]; p < L.col_ptr[k + 1]; ++p) {
            const size_type r = L.row_ind[p];
            Lx[p] = (r == k) ? Value{1} : x[r] / pivot;
        }

        // Clear the workspace over the touched positions (U/L pattern and the
        // scattered A column, in case A has an entry outside the prior pattern).
        for (size_type p = ubeg; p < uend; ++p)                 x[U.row_ind[p]] = Value{0};
        for (size_type p = L.col_ptr[k]; p < L.col_ptr[k + 1]; ++p) x[L.row_ind[p]] = Value{0};
        for (size_type p = C.col_ptr[k]; p < C.col_ptr[k + 1]; ++p) x[pinv[C.row_ind[p]]] = Value{0};
    }

    // Install only the values: the symbolic analysis, permutations, supernode
    // boundaries, factor pattern and both solve schedules stay as they were. The
    // schedules are value-agnostic and the pattern is unchanged, so rebuilding
    // them here would reproduce them entry for entry at O(nnz) cost per refactor.
    // The validating install goes first: everything after it is noexcept, so a
    // rejected install leaves `fac` wholly unchanged (row scaling included).
    fac.set_factor_values(std::move(Lx), std::move(Ux));
    fac.row_scale = std::move(row_scale);   // recomputed above (empty if unscaled)
    fac.num_perturbed = 0;   // refactor replays values without perturbing (throws on zero pivot)
}

/// Refactorize into a NEW factor object, leaving `prev` untouched: a copy of prev
/// (symbolic analysis, permutations, supernode boundaries, factor pattern and
/// solve schedules) refactorized in place. Prefer supernodal_lu_refactor_in_place
/// in a transient loop that overwrites its factorization anyway -- it skips this
/// copy. Same preconditions and exceptions as supernodal_lu_refactor_in_place; on
/// a throw `prev` is untouched (the copy is discarded).
template <typename Value, typename Parameters>
    requires OrderedField<Value>
supernodal_lu_factor<Value> supernodal_lu_refactor(
    const mat::compressed2D<Value, Parameters>& A,
    const supernodal_lu_factor<Value>& prev)
{
    supernodal_lu_factor<Value> result = prev;
    supernodal_lu_refactor_in_place(A, result);
    return result;
}

/// One-shot supernodal LU solve with a fill-reducing column ordering.
/// `Accumulator` (default `void`) is forwarded to both the numeric factorization
/// and the solve, so one policy covers the whole solve path (#535).
template <typename Accumulator = void, typename Value, typename Parameters, typename VecX, typename VecB, typename Ordering>
    requires OrderedField<Value>
void supernodal_lu_solve(const mat::compressed2D<Value, Parameters>& A,
                         VecX& x, const VecB& b, const Ordering& ordering,
                         Value threshold = Value{1})
{
    auto sym = supernodal_lu_symbolic_analyze(A, ordering);
    auto fac = supernodal_lu_numeric<Accumulator>(A, sym, threshold);
    fac.template solve<Accumulator>(x, b);
}

/// Mixed-precision solve with iterative refinement: factor in `FactorValue`
/// (e.g. float) with accumulation in `Accumulator`, then refine the residual in
/// `Residual` (e.g. double) through the low-precision factor. `FactorValue` is
/// the only required template argument.
///
/// `pivot_perturb` (default 0 = off, a hard throw as before) is forwarded to
/// supernodal_lu_numeric: a pivot that cancellation in `FactorValue` drives
/// below `pivot_perturb * ||column||_inf` is replaced by that floor instead of
/// throwing, and `refine_result::num_perturbed` counts them (#185). This is the
/// path where it matters: a matrix nonsingular in `Residual` can have an exact
/// zero pivot in a low-precision factor. The solve then completes, and the
/// refinement does what it can. A pivot cancelled exactly to zero in precision
/// u means a condition number of order 1/u or more, so full `Residual` accuracy
/// is not promised then; check `rel_residual`.
template <typename FactorValue, typename Accumulator = FactorValue,
          typename Residual, typename Parameters, typename Ordering>
refine_result supernodal_lu_solve_refined(
    const mat::compressed2D<Residual, Parameters>& A,
    vec::dense_vector<Residual>& x,
    const vec::dense_vector<Residual>& b,
    const Ordering& ordering,
    const refine_options& opt = {},
    FactorValue pivot_perturb = FactorValue{0})
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
    auto fac = supernodal_lu_numeric<Accumulator>(Af, sym, FactorValue{1}, 64, false, pivot_perturb);
    for (std::size_t i = 0; i < static_cast<std::size_t>(x.size()); ++i)
        x(static_cast<int>(i)) = Residual{0};
    refine_result res = iterative_refine(A, fac, b, x, opt);
    res.num_perturbed = fac.num_perturbed;
    return res;
}

} // namespace mtl::sparse::factorization
