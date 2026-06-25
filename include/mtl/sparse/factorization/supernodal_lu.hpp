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
#include <mtl/interface/blas.hpp>                    // blas::gemv / blas::trsv (MTL5_HAS_BLAS)
#include <mtl/interface/dispatch_traits.hpp>         // is_blas_scalar_v
#ifdef MTL5_NATIVE_FAST_GEMM
#include <mtl/detail/gemm_blocked.hpp>                // native-SIMD blocked GEMM (panel update)
#endif
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
        std::vector<Value> w(n);
        for (std::size_t i = 0; i < n; ++i) w[i] = static_cast<Value>(b(row_perm[i]));
        dense_lower_solve(L, w);                               // L z = P b
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
/// \throws std::runtime_error on a singular (zero) pivot.
template <typename Value, typename Parameters, typename Accumulator = Value>
    requires OrderedField<Value>
supernodal_lu_factor<Value> supernodal_lu_numeric(
    const mat::compressed2D<Value, Parameters>& A,
    const supernodal_lu_symbolic& sym,
    Value threshold = Value{1},
    std::size_t max_super = 64)
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

    using idx32 = std::int32_t;
    std::vector<idx32>          pinv(n, -1);
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
    size_type open_sf = 0;                         // first column of the currently-open supernode
    std::vector<size_type> prev_cand;              // sorted orig candidate rows of column k-1

    std::vector<idx32> contrib_super;              // closed supernodes hitting column k (dedup)
    std::vector<char>  super_seen;                 // marker per closed supernode
    std::vector<Accumulator> segacc;               // supernode segment, accumulator precision (generic path)
    std::vector<Value>       segval;               // segment / pivots in Value (generic path)

    // Fast dense block update when no custom accumulator is requested and the
    // value type is BLAS-eligible (float/double): route the supernode TRSV+GEMV
    // through BLAS-2 / native-SIMD instead of the generic accumulator loops.
    constexpr bool fast_block =
        std::is_same_v<Accumulator, Value> && interface::is_blas_scalar_v<Value>;

    // Panel width: columns are processed in panels of `panelW`, and supernodes
    // are aligned to panel boundaries so that, at each panel start, every earlier
    // column lives in a *closed* supernode (enables the batched BLAS-3 update).
    constexpr size_type panelW = 16;

    // Panel workspace: n x panelW dense columns (column-major), reused across
    // panels. `x` is repointed to column j of this within the per-column body, so
    // the per-column code keeps its `x[...]` syntax unchanged.
    std::vector<Accumulator> Xpanel(static_cast<std::size_t>(n) * panelW);
    std::vector<Value>       pseg, pY;             // batched TRSM/GEMM buffers (fast path)
    // (closed supernode id, panel column) pairs: which columns each closed
    // supernode contributes to (subset tracking -> GEMM only over reached cols).
    std::vector<std::pair<idx32, idx32>> panel_contrib;

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

    // Gilbert-Peierls reach of column k over the current L (EL-pruned DFS). Fills
    // xi[top..n) in topological order and records visited nodes in reach_nodes
    // (caller resets marked[] via reach_nodes). Returns top.
    auto compute_reach = [&](size_type k) -> size_type {
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
        return top;
    };

    for (size_type kp = 0; kp < n; kp += panelW) {
        const size_type wpan = std::min(panelW, n - kp);

        // Close any supernode straddling the panel boundary, so every column < kp
        // is in a CLOSED supernode (precondition for the batched update below).
        // Lp[kp] must be set first: close_supernode reads Lp[kp] as the end of the
        // last column's L entries when building the supernode's dense block.
        if (open_sf < kp) { Lp[kp] = Li.size(); close_supernode(kp, prev_cand); open_sf = kp; }

        // ===== panel start: scatter each panel column into X, and (fast path)
        //       record (closed supernode, column) pairs for the panel =====
        panel_contrib.clear();
        for (size_type j = 0; j < wpan; ++j) {
            const size_type kk = kp + j;
            Accumulator* xj = &Xpanel[j * n];          // X(:,j), clean from the prior panel
            for (size_type p = C.col_ptr[kk]; p < C.col_ptr[kk + 1]; ++p)
                AT::assign(xj[C.row_ind[p]], C.values[p]);
            if constexpr (fast_block) {
                size_type top = compute_reach(kk);     // reach over current L (cols < kp)
                for (size_type px = top; px < n; ++px) {
                    std::ptrdiff_t jc = pinv[xi[px]];
                    if (jc < 0) continue;
                    idx32 sid = col_csuper[jc];
                    if (sid >= 0)
                        panel_contrib.emplace_back(sid, static_cast<idx32>(j));
                }
                for (std::ptrdiff_t node : reach_nodes) marked[node] = 0;
            }
        }

        // ===== batched closed-supernode update: each contributing supernode is
        //       applied ONLY to the columns that reach it (subset), in increasing
        //       supernode order, as one TRSM + GEMM (BLAS-3). =====
        if constexpr (fast_block) {
            // Sort by (supernode sf, column); dedup (sid,col) pairs from a column
            // reaching a supernode through several of its pivotal rows.
            std::sort(panel_contrib.begin(), panel_contrib.end(),
                      [&](const auto& a, const auto& b) {
                          if (snodes[a.first].sf != snodes[b.first].sf)
                              return snodes[a.first].sf < snodes[b.first].sf;
                          if (a.first != b.first) return a.first < b.first;
                          return a.second < b.second;
                      });
            panel_contrib.erase(std::unique(panel_contrib.begin(), panel_contrib.end()),
                                panel_contrib.end());
            size_type gi = 0;
            while (gi < panel_contrib.size()) {
                const idx32 sid = panel_contrib[gi].first;
                size_type ge = gi;
                while (ge < panel_contrib.size() && panel_contrib[ge].first == sid) ++ge;
                const size_type ns = ge - gi;                  // reached columns in this panel
                const auto& S = snodes[sid];
                const size_type w = S.w, m = S.m, mo = m - w;
                pseg.assign(w * ns, Value{0});
                for (size_type t = 0; t < ns; ++t) {           // gather segments
                    const Accumulator* xj = &Xpanel[panel_contrib[gi + t].second * n];
                    for (size_type c = 0; c < w; ++c) pseg[t * w + c] = xj[S.rows[c]];
                }
                for (size_type t = 0; t < ns; ++t) {           // TRSM (per-column trsv)
#ifdef MTL5_HAS_BLAS
                    interface::blas::trsv('L', 'N', 'U', static_cast<int>(w),
                                          S.block.data(), static_cast<int>(m), pseg.data() + t * w, 1);
#else
                    for (size_type c = 0; c < w; ++c)
                        for (size_type r = c + 1; r < w; ++r)
                            pseg[t * w + r] -= S.block[c * m + r] * pseg[t * w + c];
#endif
                }
                for (size_type t = 0; t < ns; ++t) {           // write back diagonal rows
                    Accumulator* xj = &Xpanel[panel_contrib[gi + t].second * n];
                    for (size_type c = 0; c < w; ++c) xj[S.rows[c]] = pseg[t * w + c];
                }
                if (mo > 0) {                                  // GEMM: Y = Loff * seg
                    pY.assign(mo * ns, Value{0});
                    const Value* Loff = S.block.data() + w;    // off-diagonal block (mo x w, lda=m)
#ifdef MTL5_HAS_BLAS
                    interface::blas::gemm('N', 'N', static_cast<int>(mo), static_cast<int>(ns),
                                          static_cast<int>(w), Value{1}, Loff, static_cast<int>(m),
                                          pseg.data(), static_cast<int>(w), Value{0},
                                          pY.data(), static_cast<int>(mo));
#elif defined(MTL5_NATIVE_FAST_GEMM)
                    mtl::detail::gemm_blocked<Value>(mo, ns, w, Value{1},
                        Loff, 1, static_cast<std::ptrdiff_t>(m),
                        pseg.data(), 1, static_cast<std::ptrdiff_t>(w),
                        Value{0}, pY.data(), mo);
#else
                    for (size_type t = 0; t < ns; ++t)
                        for (size_type r = 0; r < mo; ++r) {
                            Value s = Value{0};
                            for (size_type c = 0; c < w; ++c) s += Loff[c * m + r] * pseg[t * w + c];
                            pY[t * mo + r] = s;
                        }
#endif
                    for (size_type t = 0; t < ns; ++t) {       // scatter-subtract
                        Accumulator* xj = &Xpanel[panel_contrib[gi + t].second * n];
                        for (size_type r = 0; r < mo; ++r) xj[S.rows[w + r]] -= pY[t * mo + r];
                    }
                }
                gi = ge;
            }
        }

        // ===== per-column inner factorization =====
        for (size_type j = 0; j < wpan; ++j) {
            const size_type k = kp + j;
            Accumulator* x = &Xpanel[j * n];           // C(:,k) already scattered into X(:,j)
            Lp[k] = Li.size();
            Up[k] = Ui.size();

            size_type top = compute_reach(k);          // full reach (closed + intra-panel)

            // --- apply updates from earlier columns ---
            if constexpr (fast_block) {
                // Closed supernodes (cols < kp) were applied by the batched GEMM at
                // panel start; apply intra-panel columns [kp, k) as scalar SAXPY.
                for (size_type px = top; px < n; ++px) {
                    std::ptrdiff_t jn = xi[px];
                    std::ptrdiff_t jcol = pinv[jn];
                    if (jcol < 0 || static_cast<size_type>(jcol) < kp) continue;
                    Value xj = x[jn];
                    for (size_type p = Lp[jcol] + 1; p < Lp[jcol + 1]; ++p)
                        x[Li[p]] -= Lx[p] * xj;
                }
            } else {
                // Generic accumulator path: Phase-2 per-column closed block update +
                // open scalar update (no panel batching for custom accumulators).
                contrib_super.clear();
                for (size_type px = top; px < n; ++px) {
                    std::ptrdiff_t jn = xi[px];
                    std::ptrdiff_t jcol = pinv[jn];
                    if (jcol < 0) continue;
                    idx32 sid = col_csuper[jcol];
                    if (sid >= 0 && !super_seen[sid]) { super_seen[sid] = 1; contrib_super.push_back(sid); }
                }
                std::sort(contrib_super.begin(), contrib_super.end(),
                          [&](idx32 a, idx32 b) { return snodes[a].sf < snodes[b].sf; });
                for (idx32 sid : contrib_super) {
                    const auto& S = snodes[sid];
                    const size_type w = S.w, m = S.m;
                    if (w == 1) {
                        const Value xv = static_cast<Value>(AT::value(x[S.rows[0]]));
                        for (size_type r = 1; r < m; ++r)
                            AT::add_product(x[S.rows[r]], S.block[r], -xv);
                    } else {
                        segacc.resize(w); segval.resize(w);
                        for (size_type c = 0; c < w; ++c) segacc[c] = x[S.rows[c]];
                        for (size_type c = 0; c < w; ++c) {
                            segval[c] = AT::template value<Value>(segacc[c]);
                            for (size_type r = c + 1; r < w; ++r)
                                AT::add_product(segacc[r], S.block[c * m + r], -segval[c]);
                        }
                        for (size_type c = 0; c < w; ++c) x[S.rows[c]] = segacc[c];
                        for (size_type r = w; r < m; ++r) {
                            Accumulator& dst = x[S.rows[r]];
                            for (size_type c = 0; c < w; ++c)
                                AT::add_product(dst, -S.block[c * m + r], segval[c]);
                        }
                    }
                    super_seen[sid] = 0;
                }
                for (size_type px = top; px < n; ++px) {
                    std::ptrdiff_t jn = xi[px];
                    std::ptrdiff_t jcol = pinv[jn];
                    if (jcol < 0 || static_cast<size_type>(jcol) < open_sf) continue;
                    Value xj = AT::value(x[jn]);
                    for (size_type p = Lp[jcol] + 1; p < Lp[jcol + 1]; ++p)
                        AT::add_product(x[Li[p]], -Lx[p], xj);
                }
            }

        // --- threshold partial pivoting + emit U(:,k) for pivotal rows ---
        Value a = Value{0}; std::ptrdiff_t ipiv = -1; bool have = false;
        for (size_type px = top; px < n; ++px) {
            std::ptrdiff_t i = xi[px];
            if (pinv[i] < 0) {
                Value t = abs(AT::value(x[i]));
                if (!have || t > a) { a = t; ipiv = i; have = true; }
            } else {
                Ui.push_back(pinv[i]); Ux.push_back(AT::value(x[i]));
            }
        }
        if (ipiv < 0 || a == Value{0})
            throw std::runtime_error(
                "supernodal_lu_numeric: zero pivot at column " + std::to_string(k) + " (singular)");
        if (pinv[static_cast<std::ptrdiff_t>(k)] < 0
            && abs(AT::value(x[k])) >= threshold * a)
            ipiv = static_cast<std::ptrdiff_t>(k);

        Value pivot = AT::value(x[ipiv]);
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
                (k % panelW != 0) &&                  // supernodes don't cross panel boundaries
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
        }   // per-column loop
    }       // panel loop
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
