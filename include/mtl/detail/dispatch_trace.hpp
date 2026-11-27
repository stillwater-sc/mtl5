#pragma once
// MTL5 -- a test hook that records which kernel mult() dispatched to (#944).
//
// A lost fast path (BLAS, the blocked GEMM, the SIMD GEMV) is the most common
// silent slowdown: the result is still right, only slower. A test that defines
// MTL5_DISPATCH_TRACE before its first MTL5 include can read which path each
// mult() took. In every other build MTL5_DISPATCH_HIT compiles to nothing.
//
// The macro must be defined the same way in every translation unit of one
// program (the functions that call MTL5_DISPATCH_HIT are inline), so it is for
// a test executable of its own, never for a library or an application.

#ifdef MTL5_DISPATCH_TRACE

#include <atomic>
#include <cstddef>

namespace mtl::detail::dispatch_trace {

/// The kernels mult() can choose, for a matrix-matrix and a matrix-vector
/// product.
enum class path : std::size_t {
    gemm_blas,      ///< interface::blas::gemm (MTL5_HAS_BLAS)
    gemm_blocked,   ///< detail::gemm_blocked (MTL5_NATIVE_FAST_GEMM)
    gemm_generic,   ///< detail::mult_generic_par, the portable loop
    gemv_blas,      ///< interface::blas::gemv (MTL5_HAS_BLAS)
    gemv_simd,      ///< detail::gemv_rowmajor / gemv_colmajor (MTL5_NATIVE_FAST_GEMM)
    gemv_generic,   ///< detail::mult_generic_par, the portable loop
    count_
};

inline std::atomic<unsigned long> hits[static_cast<std::size_t>(path::count_)]{};

inline void hit(path p) noexcept { hits[static_cast<std::size_t>(p)].fetch_add(1, std::memory_order_relaxed); }
inline unsigned long count(path p) noexcept { return hits[static_cast<std::size_t>(p)].load(std::memory_order_relaxed); }
inline void reset() noexcept {
    for (auto& h : hits) h.store(0, std::memory_order_relaxed);
}

} // namespace mtl::detail::dispatch_trace

#define MTL5_DISPATCH_HIT(p) ::mtl::detail::dispatch_trace::hit(::mtl::detail::dispatch_trace::path::p)

#else

#define MTL5_DISPATCH_HIT(p) ((void)0)

#endif
