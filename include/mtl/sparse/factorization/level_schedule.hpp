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
// The schedule is O(nnz) to build and is meant to be computed once and reused
// across many solves (the factor-once / solve-many transient path), so the
// build cost amortizes. Threading is off by default (MTL5_NUM_THREADS=1): the
// solve then runs the levels serially and is byte-identical to dense_lower_solve.

#include <algorithm>
#include <cstddef>
#include <vector>

#include <mtl/sparse/util/csc.hpp>
#include <mtl/detail/thread_pool.hpp>

namespace mtl::sparse::factorization {

/// Reusable schedule for a level-scheduled lower-triangular forward solve.
/// Built from a lower-triangular L in CSC; equivalent to dense_lower_solve.
template <typename Value>
struct lower_solve_schedule {
    std::size_t n = 0;
    // CSR of the strictly-lower entries (k < i), increasing k within each row.
    std::vector<std::size_t> row_ptr;   // length n+1
    std::vector<std::size_t> col_ind;   // length nnz_offdiag
    std::vector<Value>       val;       // length nnz_offdiag
    std::vector<Value>       diag;      // length n, L(i,i)
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
lower_solve_schedule<Value> build_lower_solve_schedule(
    const util::csc_matrix<Value, SizeType>& L)
{
    const std::size_t n = static_cast<std::size_t>(L.ncols);
    lower_solve_schedule<Value> s;
    s.n = n;
    s.diag.assign(n, Value{0});
    s.row_ptr.assign(n + 1, std::size_t{0});

    // Pass 1: diagonal + count strictly-lower entries per row (for the transpose).
    std::size_t nnz_off = 0;
    for (std::size_t j = 0; j < n; ++j) {
        SizeType p = L.col_ptr[j];
        if (p < L.col_ptr[j + 1]) s.diag[j] = L.values[p];   // first entry is the diagonal
        for (SizeType q = p + 1; q < L.col_ptr[j + 1]; ++q) {
            const std::size_t i = static_cast<std::size_t>(L.row_ind[q]);   // i > j
            ++s.row_ptr[i + 1];
            ++nnz_off;
        }
    }
    for (std::size_t i = 0; i < n; ++i) s.row_ptr[i + 1] += s.row_ptr[i];
    s.col_ind.resize(nnz_off);
    s.val.resize(nnz_off);

    // Pass 2: scatter into CSR. Iterating columns j in increasing order makes
    // each row's entries land in increasing-column order.
    std::vector<std::size_t> fill(s.row_ptr.begin(), s.row_ptr.end() - 1);
    for (std::size_t j = 0; j < n; ++j) {
        for (SizeType q = L.col_ptr[j] + 1; q < L.col_ptr[j + 1]; ++q) {
            const std::size_t i = static_cast<std::size_t>(L.row_ind[q]);
            const std::size_t dst = fill[i]++;
            s.col_ind[dst] = j;
            s.val[dst] = L.values[q];
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
/// b on entry and the solution on exit. Bit-identical to dense_lower_solve; the
/// rows of each level are solved in parallel on the thread pool (serial when
/// MTL5_NUM_THREADS=1).
template <typename Value>
void level_scheduled_lower_solve(const lower_solve_schedule<Value>& s,
                                 std::vector<Value>& x)
{
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
                        acc -= s.val[p] * x[s.col_ind[p]];
                    x[i] = acc / s.diag[i];
                }
            });
    }
}

} // namespace mtl::sparse::factorization
