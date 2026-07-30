#pragma once
// MTL5 -- parallel element-wise sweep (#297 batch 10).
//
// An element-wise sweep materializes one output element per index with NO
// cross-element dependency: out[i] = f(inputs at i). The dense vector/matrix
// expression-template assignments (y = a + b, C += A .* B, ...) are exactly this
// shape. Because each output element is produced by exactly one contiguous chunk
// and the per-element computation is identical regardless of which thread runs
// it, contiguous deterministic chunking makes the parallel result BIT-IDENTICAL
// to the serial loop -- no accumulation order to preserve. Serial (a single body
// call) at MTL5_NUM_THREADS=1, so the default build pays zero overhead and stays
// byte-identical to a plain loop.

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

} // namespace mtl::detail
