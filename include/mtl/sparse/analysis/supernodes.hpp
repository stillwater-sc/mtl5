#pragma once
// MTL5 -- Supernode detection for sparse Cholesky / LDL^T factorization
//
// A supernode is a maximal range of consecutive columns [f, l) of the factor L
// that share the same off-diagonal nonzero structure and whose diagonal block is
// dense. Treating such a range as a single dense panel turns the bulk of the
// factorization's work into dense block operations (the natural place to apply
// mixed precision: store the panel low, accumulate the block update high).
//
// This header computes, for a symmetric matrix already permuted into a postorder
// of its elimination tree (so that columns of a supernode are contiguous):
//   1. the column-wise nonzero pattern of L (symbolic Cholesky, via ereach), and
//   2. the partition of columns into *fundamental* supernodes.
//
// References:
//   - Davis, "Direct Methods for Sparse Linear Systems", SIAM 2006 (ch. on
//     supernodal methods); CSparse cs_ereach for the symbolic pattern.
//   - Liu, Ng, Peyton, "On finding supernodes for sparse matrix computations",
//     SIAM J. Matrix Anal. Appl., 14(1), 1993.

#include <cassert>
#include <cstddef>
#include <vector>

#include <mtl/sparse/util/csc.hpp>
#include <mtl/sparse/analysis/elimination_tree.hpp>  // no_parent

namespace mtl::sparse::analysis {

/// Partition of the columns of L into fundamental supernodes.
///
/// All indices are in the *permuted* (postordered) space in which the columns of
/// a supernode are contiguous. For supernode s, its columns are
/// [sn_first[s], sn_first[s+1]); the first (l-f) entries of its row list are the
/// dense diagonal block rows f..l-1, followed by the shared off-diagonal rows.
struct supernode_partition {
    std::size_t              nsuper = 0;
    std::vector<std::size_t> super;       // super[col] -> supernode id
    std::vector<std::size_t> sn_first;    // size nsuper+1: first column of each supernode
    std::vector<std::size_t> lnz_ptr;     // size nsuper+1: offsets into row_idx
    std::vector<std::size_t> row_idx;     // per-supernode full row-index lists (diag block first)
    std::vector<std::size_t> col_counts;  // per-column nnz of L (incl. diagonal)
};

/// Compute the column-wise nonzero pattern of the Cholesky factor L.
///
/// `C` is a symmetric matrix in CSC format (both triangles present) and `parent`
/// is its elimination tree. Returns the pattern as (col_ptr, row_ind): for each
/// column j, the sorted row indices i >= j with L(i,j) != 0 (diagonal included).
///
/// Standard symbolic Cholesky: the nonzero columns of row k of L are the etree
/// paths from each i <= k with C(i,k) != 0, up to k (CSparse cs_ereach). We walk
/// those paths and append row k to each visited column's list; processing rows in
/// increasing k keeps every column list sorted ascending with the diagonal first.
template <typename Value, typename SizeType>
inline void symbolic_cholesky_pattern(
    const util::csc_matrix<Value, SizeType>& C,
    const std::vector<std::size_t>& parent,
    std::vector<std::size_t>& col_ptr,
    std::vector<std::size_t>& row_ind)
{
    const std::size_t n = static_cast<std::size_t>(C.ncols);
    assert(parent.size() == n);

    std::vector<std::vector<std::size_t>> colpat(n);
    std::vector<std::size_t> mark(n, no_parent);  // mark[node] == k while column k walks it

    for (std::size_t k = 0; k < n; ++k) {
        // Diagonal first (keeps each column list sorted with the diagonal leading).
        colpat[k].push_back(k);
        for (SizeType p = C.col_ptr[k]; p < C.col_ptr[k + 1]; ++p) {
            std::size_t i = static_cast<std::size_t>(C.row_ind[p]);
            if (i >= k) continue;  // only the upper triangle (i < k)
            // Walk i up the etree toward k, stopping at the first node already
            // visited for this k (the union of these paths is row k's pattern).
            std::size_t node = i;
            while (node != no_parent && node < k && mark[node] != k) {
                mark[node] = k;
                colpat[node].push_back(k);  // row k belongs to column `node` of L
                node = parent[node];
            }
        }
    }

    col_ptr.assign(n + 1, 0);
    for (std::size_t j = 0; j < n; ++j)
        col_ptr[j + 1] = col_ptr[j] + colpat[j].size();
    row_ind.resize(col_ptr[n]);
    std::size_t pos = 0;
    for (std::size_t j = 0; j < n; ++j)
        for (std::size_t r : colpat[j])
            row_ind[pos++] = r;
}

/// Detect fundamental supernodes of a symmetric matrix `C` (CSC, both triangles)
/// with elimination tree `parent`, both in postordered space.
///
/// Columns j and j+1 join the same supernode iff j is the *only* child of j+1 in
/// the elimination tree and the column counts satisfy
/// col_counts[j] == col_counts[j+1] + 1 (i.e. identical off-diagonal structure).
/// This yields fundamental supernodes; the `relax` parameter (relaxed
/// amalgamation) is reserved for a later milestone and currently only relax == 0
/// is honored.
template <typename Value, typename SizeType>
inline supernode_partition find_supernodes(
    const util::csc_matrix<Value, SizeType>& C,
    const std::vector<std::size_t>& parent,
    std::size_t relax = 0)
{
    (void)relax;  // only fundamental supernodes (relax == 0) supported for now
    const std::size_t n = static_cast<std::size_t>(C.ncols);

    supernode_partition sp;
    sp.super.assign(n, 0);

    if (n == 0) {
        sp.nsuper = 0;
        sp.sn_first.assign(1, 0);
        sp.lnz_ptr.assign(1, 0);
        return sp;
    }

    // Column patterns of L.
    std::vector<std::size_t> Lp, Li;
    symbolic_cholesky_pattern(C, parent, Lp, Li);
    sp.col_counts.resize(n);
    for (std::size_t j = 0; j < n; ++j)
        sp.col_counts[j] = Lp[j + 1] - Lp[j];

    // Number of children of each node in the elimination tree.
    std::vector<std::size_t> nchild(n, 0);
    for (std::size_t j = 0; j < n; ++j)
        if (parent[j] != no_parent)
            ++nchild[parent[j]];

    // Walk columns left to right, opening a new supernode whenever the
    // fundamental-supernode test against the previous column fails.
    sp.sn_first.push_back(0);
    std::size_t cur = 0;
    sp.super[0] = 0;
    for (std::size_t j = 1; j < n; ++j) {
        const bool merge =
            parent[j - 1] == j &&
            nchild[j] == 1 &&
            sp.col_counts[j - 1] == sp.col_counts[j] + 1;
        if (!merge) {
            ++cur;
            sp.sn_first.push_back(j);
        }
        sp.super[j] = cur;
    }
    sp.sn_first.push_back(n);
    sp.nsuper = sp.sn_first.size() - 1;

    // Row structure of each supernode = pattern of its first column (the widest:
    // it holds the dense diagonal block f..l-1 followed by the shared off-diagonal
    // rows). Building it from the first column keeps the diagonal block leading.
    sp.lnz_ptr.assign(sp.nsuper + 1, 0);
    for (std::size_t s = 0; s < sp.nsuper; ++s) {
        const std::size_t f = sp.sn_first[s];
        sp.lnz_ptr[s + 1] = sp.lnz_ptr[s] + (Lp[f + 1] - Lp[f]);
    }
    sp.row_idx.resize(sp.lnz_ptr[sp.nsuper]);
    for (std::size_t s = 0; s < sp.nsuper; ++s) {
        const std::size_t f = sp.sn_first[s];
        const std::size_t l = sp.sn_first[s + 1];
        const std::size_t w = l - f;
        std::size_t dst = sp.lnz_ptr[s];
        for (std::size_t p = Lp[f]; p < Lp[f + 1]; ++p)
            sp.row_idx[dst++] = Li[p];
        // The diagonal block of a fundamental supernode is dense, so the first w
        // rows of the list are exactly f, f+1, ..., l-1.
        (void)w;
        assert([&] {
            for (std::size_t c = 0; c < w; ++c)
                if (sp.row_idx[sp.lnz_ptr[s] + c] != f + c) return false;
            return true;
        }());
    }

    return sp;
}

} // namespace mtl::sparse::analysis
