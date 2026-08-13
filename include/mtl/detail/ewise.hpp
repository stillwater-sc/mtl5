#pragma once
// MTL5 -- parallel element-wise sweep (#297 batch 10).
//
// An element-wise sweep materializes one output element per index with NO
// cross-element dependency: out[i] = f(inputs at i). The dense vector/matrix
// expression-template assignments (y = a + b, C += A .* B, ...) are exactly this
// shape. Because each output element is produced by exactly one contiguous chunk
// and the per-element computation is identical regardless of which thread runs
// it, contiguous deterministic chunking makes the parallel result BIT-IDENTICAL
// to the serial loop -- no accumulation order to preserve. At MTL5_NUM_THREADS=1
// the whole index space is one chunk on the calling thread (no parallel
// dispatch; body is still invoked per index), so the default build pays zero
// overhead and stays byte-identical to a plain loop.

#include <algorithm>
#include <cstddef>

#include <mtl/detail/thread_pool.hpp>

namespace mtl::detail {

/// Run body(i) for every i in [0, n) across the pool, partitioned into
/// contiguous chunks. `work_per_elem` (>= 1 effectively) estimates the per-index
/// cost so the grain targets ~64K work units per chunk; a deeper expression tree
/// (more flops per element) => smaller grain. Bit-identical across thread counts.
template <typename Body>
inline void parallel_ewise(std::size_t n, std::size_t work_per_elem, Body&& body) {
    const std::size_t w = work_per_elem < 1 ? 1 : work_per_elem;
    const std::size_t grain = std::max<std::size_t>(std::size_t{1}, std::size_t{65536} / w);
    thread_pool::instance().parallel_for(n, grain, [&](std::size_t b, std::size_t e) {
        for (std::size_t i = b; i < e; ++i) body(i);
    });
}

/// Run body(r, c) for every element of a rows x cols index space, in row-major
/// order, across the pool. The 2D space is FLATTENED to a linear element index
/// [0, rows*cols) before chunking, so the work splits on the element count rather
/// than the row count: a wide/short expression (1 x N) parallelizes exactly like a
/// tall one (N x 1), which a row-per-unit decomposition cannot do (#313).
///
/// `work_per_elem` estimates the per-ELEMENT cost (1 for an ordinary element-wise
/// expression), so the grain targets ~64K work units per chunk -- the same chunk
/// size in elements the row-per-unit form produced for tall matrices.
///
/// Each chunk walks (r, c) incrementally from one division at its start, so the
/// per-element cost is an increment and a compare, not a div/mod. The visit order
/// within a chunk is the serial nested loop's, every element is produced by
/// exactly one chunk, and the per-element computation does not depend on which
/// thread runs it -- so the result is BIT-IDENTICAL to the serial loop and to any
/// other thread count. At MTL5_NUM_THREADS=1 the whole space is one chunk on the
/// calling thread -- no parallel dispatch -- with body still invoked per element.
template <typename Body>
inline void parallel_ewise_2d(std::size_t rows, std::size_t cols,
                              std::size_t work_per_elem, Body&& body) {
    if (rows == 0 || cols == 0) return;   // nothing to sweep; keeps the % below safe
    const std::size_t w = work_per_elem < 1 ? 1 : work_per_elem;
    const std::size_t grain = std::max<std::size_t>(std::size_t{1}, std::size_t{65536} / w);
    thread_pool::instance().parallel_for(
        rows * cols, grain, [&](std::size_t b, std::size_t e) {
            std::size_t r = b / cols, c = b % cols;
            for (std::size_t t = b; t < e; ++t) {
                body(r, c);
                if (++c == cols) { c = 0; ++r; }
            }
        });
}

} // namespace mtl::detail
