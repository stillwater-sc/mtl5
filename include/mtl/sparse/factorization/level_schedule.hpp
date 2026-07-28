#pragma once
// MTL5 -- level scheduling for the sparse triangular solve (#297 batch 3).
//
// The dense_lower_solve in triangular_solve.hpp is a CSC column-scatter: column
// j updates rows i>j (x[i] -= L(i,j)*x[j]). Two independent columns writing the
// same x[i] would race, and their subtraction order is not fixed, so the CSC
// form does not parallelize deterministically.
//
// This builds an equivalent ROW-oriented (gather) schedule that IS parallel and
// bit-identical to the serial CSC solve:
//
//   * Transpose the strictly-lower part of L to CSR so each row i lists its
//     entries (k, L(i,k)) with k<i in INCREASING k order -- the same order the
//     CSC scatter accumulates them into x[i]. Then
//         x[i] = (b[i] - sum_{k<i} L(i,k)*x[k]) / L(i,i)
//     performs the identical floating-point operations in the identical order,
//     so the result is bit-for-bit the same as dense_lower_solve.
//   * Assign each row a level: level[i] = 1 + max level of its dependencies
//     (rows k<i with L(i,k)!=0). Rows in the same level are mutually
//     independent, so within a level every row writes its OWN x[i] (no race)
//     while reading only x[k] from earlier levels (already finalized).
//
// The schedule stores only STRUCTURE -- the positions of L's entries, not their
// values -- and reads the numbers from L at solve time. So a same-pattern
// re-factorization that overwrites L's values in place (the transient / mp-spice
// path) can reuse the schedule with fresh values; only a pattern change needs a
// rebuild (caught by comparing L.nnz(), and in normal use a pattern change comes
// with a new symbolic analysis / object). The schedule is O(nnz) to build and is
// meant to be computed once and reused across many solves, so the build cost
// amortizes. Threading is off by default (MTL5_NUM_THREADS=1): the solve then
// runs the levels serially and is byte-identical to dense_lower_solve.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <vector>

#include <mtl/sparse/util/csc.hpp>
#include <mtl/detail/thread_pool.hpp>

namespace mtl::sparse::factorization {

/// Reusable schedule for a level-scheduled lower-triangular forward solve.
/// Built from a lower-triangular L in CSC; equivalent to dense_lower_solve.
/// Stores positions into L's arrays (value-agnostic), so it survives an in-place
/// same-pattern re-factorization of L.
struct lower_solve_schedule {
    std::size_t n = 0;
    std::size_t built_nnz = 0;          // L.nnz() when built (staleness guard)
    // CSR of the strictly-lower entries (k < i), increasing k within each row.
    std::vector<std::size_t> row_ptr;   // length n+1
    std::vector<std::size_t> col_ind;   // length nnz_offdiag: the column k
    std::vector<std::size_t> val_pos;   // length nnz_offdiag: index of L(i,k) in L.values
    std::vector<std::size_t> diag_pos;  // length n: index of L(i,i) in L.values
    // Level partition of the rows: order[level_ptr[l] .. level_ptr[l+1]) are the
    // rows of level l (kept in increasing-row order for determinism).
    std::vector<std::size_t> level_ptr; // length nlevels+1
    std::vector<std::size_t> order;     // length n
    std::size_t grain = 1;              // parallel_for grain (rows per chunk min)
};

/// Build the forward-solve schedule from a lower-triangular L (CSC). The first
/// stored entry of each column must be the diagonal (row j), matching the layout
/// dense_lower_solve expects.
template <typename Value, typename SizeType>
lower_solve_schedule build_lower_solve_schedule(
    const util::csc_matrix<Value, SizeType>& L)
{
    const std::size_t n = static_cast<std::size_t>(L.ncols);
    lower_solve_schedule s;
    s.n = n;
    s.built_nnz = static_cast<std::size_t>(L.nnz());
    s.diag_pos.assign(n, std::size_t{0});
    s.row_ptr.assign(n + 1, std::size_t{0});

    // Pass 1: diagonal position + count strictly-lower entries per row.
    std::size_t nnz_off = 0;
    for (std::size_t j = 0; j < n; ++j) {
        SizeType p = L.col_ptr[j];
        if (p < L.col_ptr[j + 1]) s.diag_pos[j] = static_cast<std::size_t>(p);   // first entry is the diagonal
        for (SizeType q = p + 1; q < L.col_ptr[j + 1]; ++q) {
            const std::size_t i = static_cast<std::size_t>(L.row_ind[q]);   // i > j
            ++s.row_ptr[i + 1];
            ++nnz_off;
        }
    }
    for (std::size_t i = 0; i < n; ++i) s.row_ptr[i + 1] += s.row_ptr[i];
    s.col_ind.resize(nnz_off);
    s.val_pos.resize(nnz_off);

    // Pass 2: scatter into CSR. Iterating columns j in increasing order makes
    // each row's entries land in increasing-column order.
    std::vector<std::size_t> fill(s.row_ptr.begin(), s.row_ptr.end() - 1);
    for (std::size_t j = 0; j < n; ++j) {
        for (SizeType q = L.col_ptr[j] + 1; q < L.col_ptr[j + 1]; ++q) {
            const std::size_t i = static_cast<std::size_t>(L.row_ind[q]);
            const std::size_t dst = fill[i]++;
            s.col_ind[dst] = j;
            s.val_pos[dst] = static_cast<std::size_t>(q);
        }
    }

    // Pass 3: levels. level[i] = 1 + max level of its dependencies; rows i are
    // processed in increasing order so every dependency k<i already has a level.
    std::vector<std::size_t> level(n, 0);
    std::size_t nlevels = 0;
    for (std::size_t i = 0; i < n; ++i) {
        std::size_t lv = 0;
        for (std::size_t p = s.row_ptr[i]; p < s.row_ptr[i + 1]; ++p) {
            const std::size_t k = s.col_ind[p];
            if (level[k] + 1 > lv) lv = level[k] + 1;
        }
        level[i] = lv;
        if (lv + 1 > nlevels) nlevels = lv + 1;
    }

    // Counting-sort the rows into level order.
    s.level_ptr.assign(nlevels + 1, std::size_t{0});
    for (std::size_t i = 0; i < n; ++i) ++s.level_ptr[level[i] + 1];
    for (std::size_t l = 0; l < nlevels; ++l) s.level_ptr[l + 1] += s.level_ptr[l];
    s.order.resize(n);
    std::vector<std::size_t> cursor(s.level_ptr.begin(), s.level_ptr.end() - 1);
    for (std::size_t i = 0; i < n; ++i) s.order[cursor[level[i]]++] = i;

    // Grain: aim for ~64K flops per chunk, using the average row length as the
    // per-row work estimate (2 flops per stored entry).
    const std::size_t avg = n ? std::max<std::size_t>(1, nnz_off / n) : 1;
    s.grain = std::max<std::size_t>(std::size_t{1}, std::size_t{32768} / avg);
    return s;
}

/// Level-scheduled lower-triangular forward solve: solve L x = b with x holding
/// b on entry and the solution on exit. Values are read from L (so a same-pattern
/// re-factorization is picked up), using the positions cached in the schedule.
/// Bit-identical to dense_lower_solve; the rows of each level are solved in
/// parallel on the thread pool (serial when MTL5_NUM_THREADS=1).
template <typename Value, typename SizeType>
void level_scheduled_lower_solve(const util::csc_matrix<Value, SizeType>& L,
                                 const lower_solve_schedule& s,
                                 std::vector<Value>& x)
{
    assert(s.n == static_cast<std::size_t>(L.ncols) &&
           s.built_nnz == static_cast<std::size_t>(L.nnz()) &&
           "schedule does not match L (rebuild after a pattern change)");
    assert(x.size() >= s.n && "solution vector shorter than the system");
    const std::size_t nlevels = s.level_ptr.empty() ? 0 : s.level_ptr.size() - 1;
    for (std::size_t l = 0; l < nlevels; ++l) {
        const std::size_t lb = s.level_ptr[l];
        const std::size_t le = s.level_ptr[l + 1];
        const std::size_t count = le - lb;
        if (count == 0) continue;
        detail::thread_pool::instance().parallel_for(
            count, s.grain,
            [&](std::size_t cb, std::size_t ce) {
                for (std::size_t t = cb; t < ce; ++t) {
                    const std::size_t i = s.order[lb + t];
                    Value acc = x[i];   // b[i]
                    for (std::size_t p = s.row_ptr[i]; p < s.row_ptr[i + 1]; ++p)
                        acc -= L.values[s.val_pos[p]] * x[s.col_ind[p]];
                    x[i] = acc / L.values[s.diag_pos[i]];
                }
            });
    }
}

// ---------------------------------------------------------------------------
// Backward (transpose) solve: L^T x = b, L lower-triangular in CSC.
//
// dense_lower_transpose_solve processes columns in DECREASING order: for each
// col it gathers x[col] -= L(i,col)*x[i] over the below-diagonal entries i>col
// of column col (already stored in increasing-row order), then divides by the
// diagonal. Those below-diagonal entries of column col are exactly row col of
// L^T, so the transpose solve is a row-gather that reads L's CSC columns
// directly -- no transpose of L is needed, unlike the forward solve.
//
// Dependencies run backward: x[col] depends on x[i] for i>col with L(i,col)!=0.
// level[col] = 1 + max level of those i (computed by scanning columns in
// decreasing order). Columns in a level are independent -- each writes its own
// x[col] reading only later columns (earlier levels) -- so a level is solved in
// parallel, and iterating each column's entries in the stored (increasing-row)
// order reproduces the serial gather exactly: bit-identical to
// dense_lower_transpose_solve.

/// Reusable schedule for a level-scheduled lower-triangular TRANSPOSE solve.
/// Value-agnostic: only the column-level partition is stored; L's values (and
/// its column structure) are read at solve time.
struct lower_transpose_solve_schedule {
    std::size_t n = 0;
    std::size_t built_nnz = 0;          // L.nnz() when built (staleness guard)
    std::vector<std::size_t> level_ptr; // length nlevels+1
    std::vector<std::size_t> order;     // length n: columns grouped by level
    std::size_t grain = 1;
};

/// Build the transpose-solve schedule from a lower-triangular L (CSC).
template <typename Value, typename SizeType>
lower_transpose_solve_schedule build_lower_transpose_solve_schedule(
    const util::csc_matrix<Value, SizeType>& L)
{
    const std::size_t n = static_cast<std::size_t>(L.ncols);
    lower_transpose_solve_schedule s;
    s.n = n;
    s.built_nnz = static_cast<std::size_t>(L.nnz());

    // level[col] = 1 + max level of its dependencies (rows i>col in column col).
    // Scan columns in decreasing order so every dependency i>col is done first.
    std::vector<std::size_t> level(n, 0);
    std::size_t nlevels = 0, nnz_off = 0;
    for (std::size_t col = n; col-- > 0; ) {
        const SizeType diag = L.col_ptr[col];
        std::size_t lv = 0;
        for (SizeType p = diag + 1; p < L.col_ptr[col + 1]; ++p) {
            const std::size_t i = static_cast<std::size_t>(L.row_ind[p]);   // i > col
            if (level[i] + 1 > lv) lv = level[i] + 1;
            ++nnz_off;
        }
        level[col] = lv;
        if (lv + 1 > nlevels) nlevels = lv + 1;
    }

    // Counting-sort the columns into level order (increasing col within a level).
    s.level_ptr.assign(nlevels + 1, std::size_t{0});
    for (std::size_t c = 0; c < n; ++c) ++s.level_ptr[level[c] + 1];
    for (std::size_t l = 0; l < nlevels; ++l) s.level_ptr[l + 1] += s.level_ptr[l];
    s.order.resize(n);
    std::vector<std::size_t> cursor(s.level_ptr.begin(), s.level_ptr.end() - 1);
    for (std::size_t c = 0; c < n; ++c) s.order[cursor[level[c]]++] = c;

    const std::size_t avg = n ? std::max<std::size_t>(1, nnz_off / n) : 1;
    s.grain = std::max<std::size_t>(std::size_t{1}, std::size_t{32768} / avg);
    return s;
}

/// Level-scheduled transpose solve: solve L^T x = b with x holding b on entry and
/// the solution on exit. Reads L's values at solve time. Bit-identical to
/// dense_lower_transpose_solve; each level's columns are solved in parallel
/// (serial when MTL5_NUM_THREADS=1).
template <typename Value, typename SizeType>
void level_scheduled_lower_transpose_solve(const util::csc_matrix<Value, SizeType>& L,
                                           const lower_transpose_solve_schedule& s,
                                           std::vector<Value>& x)
{
    assert(s.n == static_cast<std::size_t>(L.ncols) &&
           s.built_nnz == static_cast<std::size_t>(L.nnz()) &&
           "schedule does not match L (rebuild after a pattern change)");
    assert(x.size() >= s.n && "solution vector shorter than the system");
    const std::size_t nlevels = s.level_ptr.empty() ? 0 : s.level_ptr.size() - 1;
    for (std::size_t l = 0; l < nlevels; ++l) {
        const std::size_t lb = s.level_ptr[l];
        const std::size_t le = s.level_ptr[l + 1];
        const std::size_t count = le - lb;
        if (count == 0) continue;
        detail::thread_pool::instance().parallel_for(
            count, s.grain,
            [&](std::size_t cb, std::size_t ce) {
                for (std::size_t t = cb; t < ce; ++t) {
                    const std::size_t col = s.order[lb + t];
                    const SizeType diag = L.col_ptr[col];
                    if (diag >= L.col_ptr[col + 1]) continue;   // empty column: skip (as serial)
                    Value acc = x[col];
                    for (SizeType p = diag + 1; p < L.col_ptr[col + 1]; ++p)
                        acc -= L.values[p] * x[static_cast<std::size_t>(L.row_ind[p])];
                    x[col] = acc / L.values[diag];
                }
            });
    }
}

// ---------------------------------------------------------------------------
// Upper triangular forward-of-the-backsolve: U x = b, U upper in CSC with the
// diagonal stored LAST in each column (the layout dense_upper_solve expects).
//
// dense_upper_solve is a CSC column-scatter in DECREASING column order: for each
// col it divides x[col] by the diagonal, then scatters x[i] -= U(i,col)*x[col]
// over the above-diagonal entries i<col. As with the lower solve this scatter
// does not parallelize deterministically, so we build the equivalent row-gather:
//
//   x[i] = (b[i] - sum_{k>i} U(i,k)*x[k]) / U(i,i)
//
// iterating each row's entries k>i in DECREASING k order -- the same order the
// scatter accumulates them -- so the gather is bit-identical to
// dense_upper_solve. Dependencies run backward (x[i] depends on x[k], k>i), so
// levels are assigned scanning rows in decreasing order; a level's rows are
// independent and solved in parallel.

/// Reusable schedule for a level-scheduled UPPER-triangular solve U x = b.
struct upper_solve_schedule {
    std::size_t n = 0;
    std::size_t built_nnz = 0;
    // CSR of the strictly-upper entries (k > i), DECREASING k within each row.
    std::vector<std::size_t> row_ptr;
    std::vector<std::size_t> col_ind;   // the column k
    std::vector<std::size_t> val_pos;   // index of U(i,k) in U.values
    std::vector<std::size_t> diag_pos;  // index of U(i,i) in U.values (last in column i)
    std::vector<std::size_t> level_ptr;
    std::vector<std::size_t> order;
    std::size_t grain = 1;
};

/// Build the upper-solve schedule from an upper-triangular U (CSC, diagonal last).
template <typename Value, typename SizeType>
upper_solve_schedule build_upper_solve_schedule(
    const util::csc_matrix<Value, SizeType>& U)
{
    const std::size_t n = static_cast<std::size_t>(U.ncols);
    upper_solve_schedule s;
    s.n = n;
    s.built_nnz = static_cast<std::size_t>(U.nnz());
    s.diag_pos.assign(n, std::size_t{0});
    s.row_ptr.assign(n + 1, std::size_t{0});

    // Pass 1: diagonal position (last entry of each column) + count strictly-upper
    // entries per row (rows i < col, i.e. all but the last entry of the column).
    std::size_t nnz_off = 0;
    for (std::size_t j = 0; j < n; ++j) {
        if (U.col_ptr[j] < U.col_ptr[j + 1]) {
            s.diag_pos[j] = static_cast<std::size_t>(U.col_ptr[j + 1] - 1);   // diagonal last
            for (SizeType q = U.col_ptr[j]; q + 1 < U.col_ptr[j + 1]; ++q) {  // above-diagonal
                ++s.row_ptr[static_cast<std::size_t>(U.row_ind[q]) + 1];
                ++nnz_off;
            }
        }
    }
    for (std::size_t i = 0; i < n; ++i) s.row_ptr[i + 1] += s.row_ptr[i];
    s.col_ind.resize(nnz_off);
    s.val_pos.resize(nnz_off);

    // Pass 2: scatter into CSR, iterating columns in DECREASING order so each
    // row's entries land in decreasing-column order (matching the scatter).
    std::vector<std::size_t> fill(s.row_ptr.begin(), s.row_ptr.end() - 1);
    for (std::size_t j = n; j-- > 0; ) {
        if (U.col_ptr[j] >= U.col_ptr[j + 1]) continue;
        for (SizeType q = U.col_ptr[j]; q + 1 < U.col_ptr[j + 1]; ++q) {
            const std::size_t i = static_cast<std::size_t>(U.row_ind[q]);   // i < j
            const std::size_t dst = fill[i]++;
            s.col_ind[dst] = j;
            s.val_pos[dst] = static_cast<std::size_t>(q);
        }
    }

    // Pass 3: levels. level[i] = 1 + max level of its dependencies (k>i in row i);
    // scan rows i in decreasing order so every dependency k>i already has a level.
    std::vector<std::size_t> level(n, 0);
    std::size_t nlevels = 0;
    for (std::size_t ii = 0; ii < n; ++ii) {
        const std::size_t i = n - 1 - ii;
        std::size_t lv = 0;
        for (std::size_t p = s.row_ptr[i]; p < s.row_ptr[i + 1]; ++p) {
            const std::size_t k = s.col_ind[p];
            if (level[k] + 1 > lv) lv = level[k] + 1;
        }
        level[i] = lv;
        if (lv + 1 > nlevels) nlevels = lv + 1;
    }

    s.level_ptr.assign(nlevels + 1, std::size_t{0});
    for (std::size_t i = 0; i < n; ++i) ++s.level_ptr[level[i] + 1];
    for (std::size_t l = 0; l < nlevels; ++l) s.level_ptr[l + 1] += s.level_ptr[l];
    s.order.resize(n);
    std::vector<std::size_t> cursor(s.level_ptr.begin(), s.level_ptr.end() - 1);
    for (std::size_t i = 0; i < n; ++i) s.order[cursor[level[i]]++] = i;

    const std::size_t avg = n ? std::max<std::size_t>(1, nnz_off / n) : 1;
    s.grain = std::max<std::size_t>(std::size_t{1}, std::size_t{32768} / avg);
    return s;
}

/// Level-scheduled upper-triangular solve: solve U x = b with x holding b on
/// entry and the solution on exit. Reads U's values at solve time. Bit-identical
/// to dense_upper_solve; each level's rows are solved in parallel (serial when
/// MTL5_NUM_THREADS=1).
template <typename Value, typename SizeType>
void level_scheduled_upper_solve(const util::csc_matrix<Value, SizeType>& U,
                                 const upper_solve_schedule& s,
                                 std::vector<Value>& x)
{
    assert(s.n == static_cast<std::size_t>(U.ncols) &&
           s.built_nnz == static_cast<std::size_t>(U.nnz()) &&
           "schedule does not match U (rebuild after a pattern change)");
    assert(x.size() >= s.n && "solution vector shorter than the system");
    const std::size_t nlevels = s.level_ptr.empty() ? 0 : s.level_ptr.size() - 1;
    for (std::size_t l = 0; l < nlevels; ++l) {
        const std::size_t lb = s.level_ptr[l];
        const std::size_t le = s.level_ptr[l + 1];
        const std::size_t count = le - lb;
        if (count == 0) continue;
        detail::thread_pool::instance().parallel_for(
            count, s.grain,
            [&](std::size_t cb, std::size_t ce) {
                for (std::size_t t = cb; t < ce; ++t) {
                    const std::size_t i = s.order[lb + t];
                    if (U.col_ptr[i] >= U.col_ptr[i + 1]) continue;   // empty column: skip (as serial)
                    Value acc = x[i];
                    for (std::size_t p = s.row_ptr[i]; p < s.row_ptr[i + 1]; ++p)
                        acc -= U.values[s.val_pos[p]] * x[s.col_ind[p]];
                    x[i] = acc / U.values[s.diag_pos[i]];
                }
            });
    }
}

// ---------------------------------------------------------------------------
// UNIT lower triangular solves (L D L^T factors): L is unit lower with an
// IMPLICIT diagonal -- no diagonal entry is stored and the solves never divide.
// dense_unit_lower_solve scatters x[i] -= L(i,j)*x[j] for i>j (columns
// increasing); dense_unit_lower_transpose_solve gathers x[col] -= L(i,col)*x[i]
// for i>col (columns decreasing). These mirror the non-unit lower / transpose
// schedules but (a) identify the strictly-lower entries by row_ind > col rather
// than by position (there is no stored diagonal to skip) and (b) omit the final
// division. Bit-identical to the dense unit solves.

/// Schedule for a level-scheduled UNIT lower forward solve L x = b.
struct unit_lower_solve_schedule {
    std::size_t n = 0;
    std::size_t built_nnz = 0;
    std::vector<std::size_t> row_ptr;   // CSR of strictly-lower entries, increasing k per row
    std::vector<std::size_t> col_ind;
    std::vector<std::size_t> val_pos;
    std::vector<std::size_t> level_ptr;
    std::vector<std::size_t> order;
    std::size_t grain = 1;
};

template <typename Value, typename SizeType>
unit_lower_solve_schedule build_unit_lower_solve_schedule(
    const util::csc_matrix<Value, SizeType>& L)
{
    const std::size_t n = static_cast<std::size_t>(L.ncols);
    unit_lower_solve_schedule s;
    s.n = n;
    s.built_nnz = static_cast<std::size_t>(L.nnz());
    s.row_ptr.assign(n + 1, std::size_t{0});

    std::size_t nnz_off = 0;
    for (std::size_t j = 0; j < n; ++j)
        for (SizeType q = L.col_ptr[j]; q < L.col_ptr[j + 1]; ++q)
            if (static_cast<std::size_t>(L.row_ind[q]) > j) {   // strictly lower
                ++s.row_ptr[static_cast<std::size_t>(L.row_ind[q]) + 1];
                ++nnz_off;
            }
    for (std::size_t i = 0; i < n; ++i) s.row_ptr[i + 1] += s.row_ptr[i];
    s.col_ind.resize(nnz_off);
    s.val_pos.resize(nnz_off);

    std::vector<std::size_t> fill(s.row_ptr.begin(), s.row_ptr.end() - 1);
    for (std::size_t j = 0; j < n; ++j)   // columns increasing -> rows get increasing-k order
        for (SizeType q = L.col_ptr[j]; q < L.col_ptr[j + 1]; ++q) {
            const std::size_t i = static_cast<std::size_t>(L.row_ind[q]);
            if (i > j) { const std::size_t dst = fill[i]++; s.col_ind[dst] = j; s.val_pos[dst] = static_cast<std::size_t>(q); }
        }

    std::vector<std::size_t> level(n, 0);
    std::size_t nlevels = 0;
    for (std::size_t i = 0; i < n; ++i) {
        std::size_t lv = 0;
        for (std::size_t p = s.row_ptr[i]; p < s.row_ptr[i + 1]; ++p)
            if (level[s.col_ind[p]] + 1 > lv) lv = level[s.col_ind[p]] + 1;
        level[i] = lv;
        if (lv + 1 > nlevels) nlevels = lv + 1;
    }
    s.level_ptr.assign(nlevels + 1, std::size_t{0});
    for (std::size_t i = 0; i < n; ++i) ++s.level_ptr[level[i] + 1];
    for (std::size_t l = 0; l < nlevels; ++l) s.level_ptr[l + 1] += s.level_ptr[l];
    s.order.resize(n);
    std::vector<std::size_t> cursor(s.level_ptr.begin(), s.level_ptr.end() - 1);
    for (std::size_t i = 0; i < n; ++i) s.order[cursor[level[i]]++] = i;

    const std::size_t avg = n ? std::max<std::size_t>(1, nnz_off / n) : 1;
    s.grain = std::max<std::size_t>(std::size_t{1}, std::size_t{32768} / avg);
    return s;
}

/// Level-scheduled UNIT lower forward solve. Bit-identical to
/// dense_unit_lower_solve (no diagonal division).
template <typename Value, typename SizeType>
void level_scheduled_unit_lower_solve(const util::csc_matrix<Value, SizeType>& L,
                                      const unit_lower_solve_schedule& s,
                                      std::vector<Value>& x)
{
    assert(s.n == static_cast<std::size_t>(L.ncols) &&
           s.built_nnz == static_cast<std::size_t>(L.nnz()) && "schedule does not match L");
    assert(x.size() >= s.n && "solution vector shorter than the system");
    const std::size_t nlevels = s.level_ptr.empty() ? 0 : s.level_ptr.size() - 1;
    for (std::size_t l = 0; l < nlevels; ++l) {
        const std::size_t lb = s.level_ptr[l], le = s.level_ptr[l + 1];
        if (le == lb) continue;
        detail::thread_pool::instance().parallel_for(
            le - lb, s.grain,
            [&](std::size_t cb, std::size_t ce) {
                for (std::size_t t = cb; t < ce; ++t) {
                    const std::size_t i = s.order[lb + t];
                    Value acc = x[i];
                    for (std::size_t p = s.row_ptr[i]; p < s.row_ptr[i + 1]; ++p)
                        acc -= L.values[s.val_pos[p]] * x[s.col_ind[p]];
                    x[i] = acc;   // unit diagonal: no division
                }
            });
    }
}

/// Schedule for a level-scheduled UNIT lower TRANSPOSE solve L^T x = b (gathers
/// within CSC columns over the below-diagonal entries; no division).
struct unit_lower_transpose_solve_schedule {
    std::size_t n = 0;
    std::size_t built_nnz = 0;
    std::vector<std::size_t> level_ptr;
    std::vector<std::size_t> order;
    std::size_t grain = 1;
};

template <typename Value, typename SizeType>
unit_lower_transpose_solve_schedule build_unit_lower_transpose_solve_schedule(
    const util::csc_matrix<Value, SizeType>& L)
{
    const std::size_t n = static_cast<std::size_t>(L.ncols);
    unit_lower_transpose_solve_schedule s;
    s.n = n;
    s.built_nnz = static_cast<std::size_t>(L.nnz());

    std::vector<std::size_t> level(n, 0);
    std::size_t nlevels = 0, nnz_off = 0;
    for (std::size_t col = n; col-- > 0; ) {
        std::size_t lv = 0;
        for (SizeType p = L.col_ptr[col]; p < L.col_ptr[col + 1]; ++p) {
            const std::size_t i = static_cast<std::size_t>(L.row_ind[p]);
            if (i > col) { if (level[i] + 1 > lv) lv = level[i] + 1; ++nnz_off; }
        }
        level[col] = lv;
        if (lv + 1 > nlevels) nlevels = lv + 1;
    }
    s.level_ptr.assign(nlevels + 1, std::size_t{0});
    for (std::size_t c = 0; c < n; ++c) ++s.level_ptr[level[c] + 1];
    for (std::size_t l = 0; l < nlevels; ++l) s.level_ptr[l + 1] += s.level_ptr[l];
    s.order.resize(n);
    std::vector<std::size_t> cursor(s.level_ptr.begin(), s.level_ptr.end() - 1);
    for (std::size_t c = 0; c < n; ++c) s.order[cursor[level[c]]++] = c;

    const std::size_t avg = n ? std::max<std::size_t>(1, nnz_off / n) : 1;
    s.grain = std::max<std::size_t>(std::size_t{1}, std::size_t{32768} / avg);
    return s;
}

/// Level-scheduled UNIT lower transpose solve. Bit-identical to
/// dense_unit_lower_transpose_solve (no diagonal division).
template <typename Value, typename SizeType>
void level_scheduled_unit_lower_transpose_solve(const util::csc_matrix<Value, SizeType>& L,
                                                const unit_lower_transpose_solve_schedule& s,
                                                std::vector<Value>& x)
{
    assert(s.n == static_cast<std::size_t>(L.ncols) &&
           s.built_nnz == static_cast<std::size_t>(L.nnz()) && "schedule does not match L");
    assert(x.size() >= s.n && "solution vector shorter than the system");
    const std::size_t nlevels = s.level_ptr.empty() ? 0 : s.level_ptr.size() - 1;
    for (std::size_t l = 0; l < nlevels; ++l) {
        const std::size_t lb = s.level_ptr[l], le = s.level_ptr[l + 1];
        if (le == lb) continue;
        detail::thread_pool::instance().parallel_for(
            le - lb, s.grain,
            [&](std::size_t cb, std::size_t ce) {
                for (std::size_t t = cb; t < ce; ++t) {
                    const std::size_t col = s.order[lb + t];
                    Value acc = x[col];
                    for (SizeType p = L.col_ptr[col]; p < L.col_ptr[col + 1]; ++p)
                        if (static_cast<std::size_t>(L.row_ind[p]) > col)
                            acc -= L.values[p] * x[static_cast<std::size_t>(L.row_ind[p])];
                    x[col] = acc;   // unit diagonal: no division
                }
            });
    }
}

} // namespace mtl::sparse::factorization
