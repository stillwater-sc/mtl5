#pragma once
// MTL5 -- Native supernodal LDL^T factorization for symmetric matrices.
//
// A = P^T L D L^T P, with L unit lower triangular and D diagonal. Columns of L
// sharing a nonzero structure are grouped into dense panels (supernodes); the
// bulk of the arithmetic is then a dense block update of one panel by another --
// the natural granularity for mixed precision. The factor is stored low (Value)
// while every accumulation is carried in a higher-precision Accumulator and
// rounded to Value exactly once when the entry is finalized.
//
// Why LDL^T first (vs LL^T): it is square-root free, which suits low-precision
// and custom number types (posit/LNS) where sqrt is costly or less accurate, and
// the diagonal D is a natural place to study precision. Symmetric => no pivoting,
// so supernodes follow directly from the elimination tree.
//
// The numeric kernel emits a standard CSC factor (unit lower L, diagonal implicit)
// plus the diagonal vector D, so the solve and the generic iterative-refinement
// loop are reused unchanged. A blocked supernodal solve is a later optimization.
//
// Accumulator policy: customize mtl::math::accumulator_traits<Acc, Value> to pick
// the accumulation precision (a wider IEEE type, a Kahan/compensated sum, or a
// quire). The panel is touched ONLY through add_product / value / clear, so an
// add-only super-accumulator works. The default Acc == Value is byte-identical to
// a plain Value-precision factorization.
//
// References: Davis, "Direct Methods for Sparse Linear Systems", SIAM 2006;
//             Ng & Peyton, "Block sparse Cholesky algorithms ...", SIAM 1993.

#include <cassert>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <mtl/concepts/scalar.hpp>          // OrderedField
#include <mtl/math/accumulator_traits.hpp>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/convert.hpp>
#include <mtl/sparse/util/csc.hpp>
#include <mtl/sparse/util/permutation.hpp>
#include <mtl/sparse/analysis/elimination_tree.hpp>
#include <mtl/sparse/analysis/postorder.hpp>
#include <mtl/sparse/analysis/supernodes.hpp>
#include <mtl/sparse/factorization/triangular_solve.hpp>
#include <mtl/sparse/factorization/sparse_cholesky.hpp>  // sparse_cholesky_symbolic
#include <mtl/sparse/iterative_refine.hpp>

namespace mtl::sparse::factorization {

/// Symbolic analysis for supernodal LDL^T. Combines the fill-reducing ordering
/// with a postorder of the elimination tree (so supernode columns are
/// contiguous) into a single permutation `sperm`, and records the supernode
/// partition computed in that permuted space.
struct supernodal_symbolic {
    std::size_t                   n{0};
    std::vector<std::size_t>      sperm;   // sperm[new] = old (fill-reducing then postorder)
    std::vector<std::size_t>      spinv;   // inverse of sperm
    std::vector<std::size_t>      parent;  // elimination tree in permuted space
    std::size_t                   nnz_L{0};
    analysis::supernode_partition snodes;
};

/// Result of numeric supernodal LDL^T factorization. L is unit lower triangular
/// in CSC (diagonal implicit) in the permuted space; D is the diagonal.
template <typename Value>
struct supernodal_ldlt_factor {
    util::csc_matrix<Value> L;          // unit lower triangular (CSC, no diagonal)
    std::vector<Value>      D;          // diagonal entries
    supernodal_symbolic     symbolic;

    std::size_t num_rows() const { return symbolic.n; }
    std::size_t num_cols() const { return symbolic.n; }

    /// Solve A*x = b.  A = P^T L D L^T P, so x = P^T L^{-T} D^{-1} L^{-1} P b.
    template <typename VecX, typename VecB>
    void solve(VecX& x, const VecB& b) const {
        const std::size_t n = symbolic.n;
        if (static_cast<std::size_t>(x.size()) != n ||
            static_cast<std::size_t>(b.size()) != n) {
            throw std::invalid_argument(
                "supernodal_ldlt_factor::solve: vector size mismatch (expected "
                + std::to_string(n) + ")");
        }
        const auto& p = symbolic.sperm;

        std::vector<Value> w(n);
        for (std::size_t i = 0; i < n; ++i)
            w[i] = static_cast<Value>(b(p[i]));            // w = P b
        dense_unit_lower_solve(L, w);                       // L y = w
        for (std::size_t i = 0; i < n; ++i)
            w[i] /= D[i];                                   // D z = y
        dense_unit_lower_transpose_solve(L, w);             // L^T u = z
        for (std::size_t i = 0; i < n; ++i)
            x(p[i]) = static_cast<typename VecX::value_type>(w[i]);  // x = P^T u
    }
};

namespace detail {
/// Shared back end of the two symbolic overloads: fuse the fill-reducing
/// ordering with a postorder of the elimination tree (so supernode columns are
/// contiguous), then detect supernodes in that permuted space.
template <typename Value, typename Parameters>
inline supernodal_symbolic build_supernodal_symbolic(
    const mat::compressed2D<Value, Parameters>& A,
    const cholesky_symbolic& base,
    std::size_t relax)
{
    supernodal_symbolic sym;
    sym.n = base.n;
    sym.sperm = util::compose_permutations(base.perm, base.post);  // sperm[f]=perm[post[f]]
    sym.spinv = util::invert_permutation(sym.sperm);

    auto C = util::crs_to_csc(util::symmetric_permute(A, sym.sperm));
    sym.parent = analysis::elimination_tree(C);
    sym.snodes = analysis::find_supernodes(C, sym.parent, relax);
    sym.nnz_L = analysis::total_nnz(sym.snodes.col_counts);
    return sym;
}
} // namespace detail

/// Symbolic supernodal analysis with a fill-reducing ordering.
template <typename Value, typename Parameters, typename Ordering>
supernodal_symbolic supernodal_ldlt_symbolic(
    const mat::compressed2D<Value, Parameters>& A,
    const Ordering& ordering,
    std::size_t relax = 0)
{
    if (A.num_rows() != A.num_cols())
        throw std::invalid_argument("supernodal_ldlt_symbolic: matrix must be square");
    return detail::build_supernodal_symbolic(A, sparse_cholesky_symbolic(A, ordering), relax);
}

/// Symbolic supernodal analysis using the natural ordering (with postorder only).
template <typename Value, typename Parameters>
supernodal_symbolic supernodal_ldlt_symbolic(
    const mat::compressed2D<Value, Parameters>& A)
{
    if (A.num_rows() != A.num_cols())
        throw std::invalid_argument("supernodal_ldlt_symbolic: matrix must be square");
    return detail::build_supernodal_symbolic(A, sparse_cholesky_symbolic(A), 0);
}

/// Numeric supernodal LDL^T factorization.
///
/// Left-looking over supernodes: each supernode's dense panel is assembled from
/// A, updated by every earlier supernode whose off-diagonal rows reach into its
/// columns (the dense block update -- the mixed-precision hot spot), then
/// factored in place. The panel lives in `Accumulator`; it is read out and
/// rounded to `Value` exactly once per entry on store.
///
/// \throws std::runtime_error on a zero pivot (D(j) == 0).
template <typename Value, typename Parameters, typename Accumulator = Value>
    requires OrderedField<Value>
supernodal_ldlt_factor<Value> supernodal_ldlt_numeric(
    const mat::compressed2D<Value, Parameters>& A,
    const supernodal_symbolic& sym)
{
    using AT = mtl::math::accumulator_traits<Accumulator, Value>;
    const std::size_t n = sym.n;
    if (A.num_rows() != n || A.num_cols() != n) {
        throw std::invalid_argument(
            "supernodal_ldlt_numeric: matrix dimensions ("
            + std::to_string(A.num_rows()) + "x" + std::to_string(A.num_cols())
            + ") do not match symbolic analysis (n=" + std::to_string(n) + ")");
    }

    auto C = util::crs_to_csc(util::symmetric_permute(A, sym.sperm));

    const auto& snodes   = sym.snodes;
    const auto& sn_first = snodes.sn_first;
    const auto& lnz_ptr  = snodes.lnz_ptr;
    const auto& row_idx  = snodes.row_idx;
    const auto& super    = snodes.super;
    const std::size_t nsuper = snodes.nsuper;

    // Allocate L (unit lower, diagonal not stored) from the column counts.
    util::csc_matrix<Value> L;
    L.nrows = n;
    L.ncols = n;
    L.col_ptr.assign(n + 1, 0);
    for (std::size_t j = 0; j < n; ++j) {
        std::size_t cc = snodes.col_counts[j];
        L.col_ptr[j + 1] = L.col_ptr[j] + (cc > 0 ? cc - 1 : 0);
    }
    L.row_ind.resize(L.col_ptr[n]);
    L.values.resize(L.col_ptr[n]);

    std::vector<Value> D(n, Value{0});

    // Off-diagonal panel of each supernode, kept in Value for use as an update
    // source: panel_off[s][kk*moff + t] = L(row_idx[s][w+t], f+kk).
    std::vector<std::vector<Value>> panel_off(nsuper);

    // For each supernode, the earlier supernodes that update it.
    std::vector<std::vector<std::size_t>> updates(nsuper);
    for (std::size_t K = 0; K < nsuper; ++K) {
        const std::size_t fK = sn_first[K];
        const std::size_t wK = sn_first[K + 1] - fK;
        const std::size_t baseK = lnz_ptr[K];
        const std::size_t mK = lnz_ptr[K + 1] - baseK;
        std::size_t last = analysis::no_parent;
        for (std::size_t t = wK; t < mK; ++t) {           // off-diagonal rows of K
            std::size_t J = super[row_idx[baseK + t]];
            if (J != last) { updates[J].push_back(K); last = J; }
        }
    }

    constexpr std::size_t NPOS = std::numeric_limits<std::size_t>::max();
    std::vector<std::size_t> rel(n, NPOS);   // global row -> local panel row
    std::vector<Accumulator> P;              // dense panel (column-major), in Accumulator

    for (std::size_t J = 0; J < nsuper; ++J) {
        const std::size_t f = sn_first[J];
        const std::size_t l = sn_first[J + 1];
        const std::size_t w = l - f;
        const std::size_t baseJ = lnz_ptr[J];
        const std::size_t m = lnz_ptr[J + 1] - baseJ;
        const std::size_t* rows = &row_idx[baseJ];

        for (std::size_t t = 0; t < m; ++t) rel[rows[t]] = t;

        // --- assemble: scatter lower triangle of A's columns [f,l) into P (=0) ---
        P.resize(m * w);
        for (std::size_t a = 0; a < m * w; ++a) AT::clear(P[a]);
        for (std::size_t c = 0; c < w; ++c) {
            const std::size_t gcol = f + c;
            for (std::size_t p = C.col_ptr[gcol]; p < C.col_ptr[gcol + 1]; ++p) {
                const std::size_t i = static_cast<std::size_t>(C.row_ind[p]);
                if (i >= gcol)                              // lower triangle incl. diagonal
                    AT::add_product(P[c * m + rel[i]], C.values[p], Value{1});
            }
        }

        // --- update: subtract the dense block contribution of each earlier K ---
        for (std::size_t K : updates[J]) {
            const std::size_t fK = sn_first[K];
            const std::size_t wK = sn_first[K + 1] - fK;
            const std::size_t baseK = lnz_ptr[K];
            const std::size_t mK = lnz_ptr[K + 1] - baseK;
            const std::size_t moff = mK - wK;              // off-diagonal rows of K
            const std::size_t* rowsK = &row_idx[baseK + wK];
            const std::vector<Value>& UK = panel_off[K];

            // Off-diagonal rows of K at/below f participate; among them, those in
            // [f,l) name the target columns hit (B), the rest extend below (U).
            std::size_t s0 = 0;
            while (s0 < moff && rowsK[s0] < f) ++s0;
            for (std::size_t bi = s0; bi < moff && rowsK[bi] < l; ++bi) {
                const std::size_t tc = rowsK[bi] - f;       // target column in [0,w)
                for (std::size_t ui = bi; ui < moff; ++ui) {  // ui>=bi => lower triangle
                    const std::size_t tr = rel[rowsK[ui]];
                    assert(tr != NPOS && "update row must lie in the target panel");
                    Accumulator* dst = &P[tc * m + tr];
                    for (std::size_t kk = 0; kk < wK; ++kk) {
                        // coefficient D(k)*L(row_b,k) formed in Value; product with
                        // L(row_u,k) accumulated in Accumulator (negated -> subtract).
                        const Value coef = D[fK + kk] * UK[kk * moff + bi];
                        AT::add_product(*dst, UK[kk * moff + ui], -coef);
                    }
                }
            }
        }

        // --- factor the panel in place (dense unit-lower LDL^T over m rows) ---
        // P column-major; P[c*m + r] for row r, column c (rows index row_idx).
        std::vector<Value> Lcol(m);  // finalized multipliers L(row_r, f+c) in Value
        panel_off[J].assign((m - w) * w, Value{0});
        for (std::size_t c = 0; c < w; ++c) {
            const Value dcc = AT::template value<Value>(P[c * m + c]);
            // Reject a singular (exact-zero) or non-finite (NaN/Inf) pivot: either
            // would silently propagate garbage through the divisions below. LDL^T
            // does no pivoting and targets indefinite / custom number types, so we
            // do NOT reject merely small pivots (those can be legitimate). The
            // (dcc - dcc != 0) test is true for NaN and Inf and false for any
            // finite value, staying dependency-free for custom types.
            if (dcc == Value{0} || dcc - dcc != Value{0}) {
                for (std::size_t t = 0; t < m; ++t) rel[rows[t]] = NPOS;
                throw std::runtime_error(
                    "supernodal_ldlt_numeric: zero or non-finite pivot at column "
                    + std::to_string(f + c));
            }
            D[f + c] = dcc;

            // L(row_r, f+c) = P(r,c) / dcc for r > c.
            for (std::size_t r = c + 1; r < m; ++r)
                Lcol[r] = AT::template value<Value>(P[c * m + r]) / dcc;

            // Rank-1 update of the remaining diagonal-block columns cc in (c,w):
            //   P(r,cc) -= L(r,c) * dcc * L(cc,c)   for r >= cc.
            for (std::size_t cc = c + 1; cc < w; ++cc) {
                const Value coef = dcc * Lcol[cc];          // dcc * L(cc,c), in Value
                Accumulator* col = &P[cc * m];
                for (std::size_t r = cc; r < m; ++r)
                    AT::add_product(col[r], Lcol[r], -coef);
            }

            // Store column f+c: off-diagonal entries (rows > c) into CSC L, and the
            // strictly-below-block entries (r >= w) into the source panel.
            std::size_t pos = L.col_ptr[f + c];
            for (std::size_t r = c + 1; r < m; ++r) {
                L.row_ind[pos] = rows[r];
                L.values[pos]  = Lcol[r];
                ++pos;
                if (r >= w)
                    panel_off[J][c * (m - w) + (r - w)] = Lcol[r];
            }
            assert(pos == L.col_ptr[f + c + 1]);
        }

        for (std::size_t t = 0; t < m; ++t) rel[rows[t]] = NPOS;
    }

    supernodal_ldlt_factor<Value> result;
    result.L = std::move(L);
    result.D = std::move(D);
    result.symbolic = sym;
    return result;
}

/// One-shot supernodal LDL^T solve with a fill-reducing ordering.
template <typename Value, typename Parameters, typename VecX, typename VecB,
          typename Ordering>
void supernodal_ldlt_solve(
    const mat::compressed2D<Value, Parameters>& A,
    VecX& x, const VecB& b,
    const Ordering& ordering)
{
    auto sym = supernodal_ldlt_symbolic(A, ordering);
    auto num = supernodal_ldlt_numeric(A, sym);
    num.solve(x, b);
}

/// One-shot supernodal LDL^T solve using the natural ordering.
template <typename Value, typename Parameters, typename VecX, typename VecB>
void supernodal_ldlt_solve(
    const mat::compressed2D<Value, Parameters>& A,
    VecX& x, const VecB& b)
{
    auto sym = supernodal_ldlt_symbolic(A);
    auto num = supernodal_ldlt_numeric(A, sym);
    num.solve(x, b);
}

/// Mixed-precision solve with iterative refinement: factor in `FactorValue`
/// (e.g. float) with accumulation in `Accumulator`, then refine the residual in
/// `Residual` (e.g. double) through the low-precision factor. Reuses the generic
/// iterative_refine loop. `FactorValue` is the only required template argument:
///   supernodal_ldlt_solve_refined<float>(A, x, b, ordering);
///   supernodal_ldlt_solve_refined<float, double>(A, x, b, ordering);  // wider Acc
template <typename FactorValue, typename Accumulator = FactorValue,
          typename Residual, typename Parameters, typename Ordering>
refine_result supernodal_ldlt_solve_refined(
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
    for (std::size_t k = 0; k < nnz; ++k) dat[k] = static_cast<FactorValue>(src[k]);
    mat::compressed2D<FactorValue> Af(A.num_rows(), A.num_cols(), nnz,
                                      starts.data(), idx.data(), dat.data());

    auto sym = supernodal_ldlt_symbolic(Af, ordering);
    auto fac = supernodal_ldlt_numeric<FactorValue, typename decltype(Af)::param_type,
                                       Accumulator>(Af, sym);
    for (std::size_t i = 0; i < static_cast<std::size_t>(x.size()); ++i)
        x(static_cast<int>(i)) = Residual{0};
    return iterative_refine(A, fac, b, x, opt);
}

} // namespace mtl::sparse::factorization
