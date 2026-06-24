#pragma once
// MTL5 -- Column elimination tree and unsymmetric symbolic analysis for
// supernodal LU (SuperLU-style). Phase 1 of the native-SuperLU effort (#181).
//
// For unsymmetric A with partial pivoting (PA = LU), the structure of L and U is
// contained in the structure of the Cholesky factor of A^T A, whatever the row
// permutation P (George & Ng 1987; Gilbert 1994). So the symbolic analysis is
// driven by the *column* elimination tree -- the elimination tree of A^T A --
// and the column counts of chol(A^T A) give a static fill upper bound valid for
// any pivoting choice. Both are computed WITHOUT forming A^T A.
//
// References:
//   - Davis, "Direct Methods for Sparse Linear Systems", SIAM 2006
//     (cs_etree with ata=1, cs_counts with ata=1, cs_leaf).
//   - Demmel, Eisenstat, Gilbert, Li, Liu, "A Supernodal Approach to Sparse
//     Partial Pivoting", SIAM J. Matrix Anal. Appl. 20(3), 1999 (SuperLU).
//   - Gilbert, Ng, Peyton, column/row count algorithm.

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/sparse/util/csc.hpp>
#include <mtl/sparse/util/permutation.hpp>           // column_permute
#include <mtl/sparse/analysis/elimination_tree.hpp>  // no_parent
#include <mtl/sparse/analysis/postorder.hpp>         // tree_postorder
#include <mtl/sparse/analysis/supernodes.hpp>        // supernode_partition

namespace mtl::sparse::analysis {

/// Column elimination tree of A: the elimination tree of A^T A, computed without
/// forming the product (CSparse cs_etree with ata=1). `parent[j]` is the parent
/// of column j in that tree, or `no_parent` if j is a root.
template <typename Value, typename SizeType>
inline std::vector<std::size_t> column_elimination_tree(
    const util::csc_matrix<Value, SizeType>& A)
{
    const std::size_t m = static_cast<std::size_t>(A.nrows);
    const std::size_t n = static_cast<std::size_t>(A.ncols);
    std::vector<std::size_t> parent(n, no_parent);
    std::vector<std::size_t> ancestor(n, no_parent);
    std::vector<std::size_t> prev(m, no_parent);   // prev[row] = last column seen with that row

    for (std::size_t k = 0; k < n; ++k) {
        parent[k] = no_parent;
        ancestor[k] = no_parent;
        for (SizeType p = A.col_ptr[k]; p < A.col_ptr[k + 1]; ++p) {
            std::size_t row = static_cast<std::size_t>(A.row_ind[p]);
            // ata: start from the previous column that had a nonzero in this row.
            std::size_t i = prev[row];
            while (i != no_parent && i < k) {
                std::size_t inext = ancestor[i];
                ancestor[i] = k;                    // path compression
                if (inext == no_parent) parent[i] = k;
                i = inext;
            }
            prev[row] = k;
        }
    }
    return parent;
}

/// CRS overload: convert to CSC and compute the column elimination tree.
template <typename Value, typename Parameters>
inline std::vector<std::size_t> column_elimination_tree(
    const mat::compressed2D<Value, Parameters>& A)
{
    return column_elimination_tree(util::crs_to_csc(A));
}

namespace detail {

/// CSparse cs_leaf: is column j a leaf of the i-th row subtree of the column
/// etree, and if so where? Returns the least common ancestor `q` and sets
/// `jleaf` to 0 (not a leaf), 1 (first leaf), or 2 (subsequent leaf).
inline std::size_t etree_leaf(std::size_t i, std::size_t j,
                              const std::vector<std::size_t>& first,
                              std::vector<std::size_t>& maxfirst,
                              std::vector<std::size_t>& prevleaf,
                              std::vector<std::size_t>& ancestor,
                              int& jleaf)
{
    jleaf = 0;
    // i <= j: not in the strictly-lower structure. first[j] <= maxfirst[i]:
    // already counted a later leaf for this row, so j is not a new leaf.
    if (i <= j ||
        (maxfirst[i] != no_parent && first[j] != no_parent && first[j] <= maxfirst[i]))
        return no_parent;
    maxfirst[i] = first[j];
    std::size_t jprev = prevleaf[i];
    prevleaf[i] = j;
    jleaf = (jprev == no_parent) ? 1 : 2;
    if (jleaf == 1) return i;                 // first leaf of row i: lca is i itself
    // Find the least common ancestor of jprev and j: the root of jprev's set.
    std::size_t q = jprev;
    while (q != ancestor[q]) q = ancestor[q];
    // Path-compress jprev..q onto q.
    for (std::size_t s = jprev; s != q; ) {
        std::size_t sparent = ancestor[s];
        ancestor[s] = q;
        s = sparent;
    }
    return q;
}

} // namespace detail

/// Column counts of the Cholesky factor of A^T A given the column elimination
/// tree `parent` and its postorder `post`, computed without forming A^T A
/// (CSparse cs_counts with ata=1). `counts[j]` = nnz in column j of chol(A^T A)
/// including the diagonal -- the static upper bound on column j of L (and of
/// U^T) for PA = LU under any partial-pivoting permutation P.
template <typename Value, typename SizeType>
inline std::vector<std::size_t> column_counts_ata(
    const util::csc_matrix<Value, SizeType>& A,
    const std::vector<std::size_t>& parent,
    const std::vector<std::size_t>& post)
{
    const std::size_t m = static_cast<std::size_t>(A.nrows);
    const std::size_t n = static_cast<std::size_t>(A.ncols);

    std::vector<std::size_t> delta(n, 0);       // = colcount, accumulated in place
    std::vector<std::size_t> ancestor(n), maxfirst(n, no_parent),
        prevleaf(n, no_parent), first(n, no_parent);

    // AT = pattern of A^T (CSC of A^T): AT.col_ptr indexes rows of A, AT.row_ind
    // gives the columns of A present in that row.
    auto AT = util::transpose_csc(A);

    // first[j] = smallest postorder index of any descendant of j (incl. j).
    for (std::size_t k = 0; k < n; ++k) {
        std::size_t j = post[k];
        delta[j] = (first[j] == no_parent) ? 1 : 0;     // j is a leaf if unseen
        for (; j != no_parent && first[j] == no_parent; j = parent[j])
            first[j] = k;
    }

    // ata linked lists: head[k] -> rows of A whose earliest (postordered)
    // column is the kth postordered node; next[] chains them.
    std::vector<std::size_t> invpost(n);
    for (std::size_t k = 0; k < n; ++k) invpost[post[k]] = k;
    std::vector<std::size_t> head(n, no_parent), next(m, no_parent);
    for (std::size_t row = 0; row < m; ++row) {
        std::size_t kmin = n;                            // n = "+infinity" sentinel
        for (SizeType p = AT.col_ptr[row]; p < AT.col_ptr[row + 1]; ++p) {
            std::size_t col = static_cast<std::size_t>(AT.row_ind[p]);
            kmin = std::min(kmin, invpost[col]);
        }
        if (kmin < n) { next[row] = head[kmin]; head[kmin] = row; }
    }

    for (std::size_t i = 0; i < n; ++i) ancestor[i] = i;

    // delta is unsigned, but cs_counts lets a count dip negative before it is
    // rolled up to the parent. Modular (wraparound) arithmetic makes this safe
    // here: every final count is a small non-negative integer, which equals the
    // mathematically-correct value mod 2^64. We just never compare an
    // intermediate delta against zero.
    for (std::size_t k = 0; k < n; ++k) {
        std::size_t j = post[k];
        if (parent[j] != no_parent) --delta[parent[j]];  // j is not a root
        for (std::size_t J = head[k]; J != no_parent; J = next[J]) {
            for (SizeType p = AT.col_ptr[J]; p < AT.col_ptr[J + 1]; ++p) {
                std::size_t i = static_cast<std::size_t>(AT.row_ind[p]);
                int jleaf = 0;
                std::size_t q = detail::etree_leaf(i, j, first, maxfirst,
                                                   prevleaf, ancestor, jleaf);
                if (jleaf >= 1) ++delta[j];
                if (jleaf == 2) --delta[q];
            }
        }
        if (parent[j] != no_parent) ancestor[j] = parent[j];
    }

    // Roll child counts up to parents.
    for (std::size_t j = 0; j < n; ++j)
        if (parent[j] != no_parent) delta[parent[j]] += delta[j];

    return delta;
}

/// Result of unsymmetric (column-etree) symbolic analysis.
struct lu_symbolic_analysis {
    std::size_t              n = 0;
    std::vector<std::size_t> col_perm;     // column reorder applied (postorder); col_perm[new]=old
    std::vector<std::size_t> col_parent;   // column elimination tree (of A^T A), in col_perm space
    std::vector<std::size_t> col_post;     // postorder of the column etree
    std::vector<std::size_t> col_counts;   // nnz per column of chol(A^T A) (the bound)
    std::size_t              nsuper = 0;
    std::vector<std::size_t> super;        // super[col] -> supernode id
    std::vector<std::size_t> sn_first;     // size nsuper+1: first column of each supernode
    std::size_t              fill_chol_ata = 0;  // sum(col_counts) = nnz of chol(A^T A)
    std::size_t              fill_lu_bound = 0;  // 2*fill_chol_ata - n : nnz(L)+nnz(U) upper bound
};

/// Detect column-etree supernodes: maximal runs of consecutive columns where
/// each column is the only child of the next in the column etree and the counts
/// nest (col_counts[j] == col_counts[j+1] + 1) -- the unsymmetric analogue of
/// the fundamental supernodes from #178. Requires the columns to be in a
/// postorder of the column etree so a supernode's columns are contiguous.
inline void column_supernodes(const std::vector<std::size_t>& parent,
                              const std::vector<std::size_t>& col_counts,
                              std::size_t n,
                              std::size_t& nsuper,
                              std::vector<std::size_t>& super,
                              std::vector<std::size_t>& sn_first)
{
    super.assign(n, 0);
    sn_first.clear();
    if (n == 0) { nsuper = 0; sn_first.assign(1, 0); return; }

    std::vector<std::size_t> nchild(n, 0);
    for (std::size_t j = 0; j < n; ++j)
        if (parent[j] != no_parent) ++nchild[parent[j]];

    sn_first.push_back(0);
    std::size_t cur = 0;
    super[0] = 0;
    for (std::size_t j = 1; j < n; ++j) {
        const bool merge =
            parent[j - 1] == j &&
            nchild[j] == 1 &&
            col_counts[j - 1] == col_counts[j] + 1;
        if (!merge) { ++cur; sn_first.push_back(j); }
        super[j] = cur;
    }
    sn_first.push_back(n);
    nsuper = sn_first.size() - 1;
}

/// Full unsymmetric symbolic analysis on a square matrix already in the desired
/// column order (apply a fill-reducing column ordering, e.g. COLAMD, beforehand
/// and pass the permuted matrix). Computes the column elimination tree, its
/// postorder, the chol(A^T A) column counts, the supernode partition, and the
/// static L/U fill upper bound.
template <typename Value, typename Parameters>
inline lu_symbolic_analysis analyze_unsymmetric(
    const mat::compressed2D<Value, Parameters>& A)
{
    if (A.num_rows() != A.num_cols())
        throw std::invalid_argument("analyze_unsymmetric: matrix must be square");

    lu_symbolic_analysis r;
    r.n = A.num_cols();

    // 1. Column etree + postorder of the matrix as given.
    auto parent0 = column_elimination_tree(util::crs_to_csc(A));
    r.col_perm = tree_postorder(parent0);

    // 2. Reorder columns into that postorder so a supernode's columns are
    //    contiguous, then recompute the (relabeled) column etree and counts in
    //    that space. Postordering does not change the fill counts, only labels.
    auto Apost = util::column_permute(A, r.col_perm);
    auto C = util::crs_to_csc(Apost);
    r.col_parent = column_elimination_tree(C);
    r.col_post = tree_postorder(r.col_parent);
    r.col_counts = column_counts_ata(C, r.col_parent, r.col_post);
    column_supernodes(r.col_parent, r.col_counts, r.n, r.nsuper, r.super, r.sn_first);
    r.fill_chol_ata = total_nnz(r.col_counts);
    // nnz(L) and nnz(U) are each bounded by nnz(chol(A^T A)); the diagonal is
    // shared, so nnz(L)+nnz(U) <= 2*nnz(chol(A^T A)) - n.
    r.fill_lu_bound = (2 * r.fill_chol_ata >= r.n) ? 2 * r.fill_chol_ata - r.n : 0;
    return r;
}

} // namespace mtl::sparse::analysis
