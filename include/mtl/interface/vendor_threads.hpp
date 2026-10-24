#pragma once
// MTL5 -- one vendor thread for a vendor call made inside an MTL5 region (#640).
//
// A vendor BLAS/LAPACK (MKL, BLIS, OpenBLAS) and CHOLMOD/SPQR run thread teams
// of their own. Called from a tid of an MTL5 region, each tid would start a
// team, so a region over the whole pool runs pool-size x vendor-size threads
// on the pool's cores. MTL5's own kernels run nested regions serially; these
// caps give a vendor call the same rule:
//
//   * inside a region (mtl::parallel::detail::in_region()), the vendor runs on
//     one thread. It is set at the first vendor call of the region and kept
//     for the rest of it; the region's calling thread gets its previous
//     setting back when the region ends, and a pool worker, which only ever
//     runs region tids, keeps one;
//   * outside a region nothing changes: the vendor's team is the only
//     parallelism, and capping it would make a direct call serial.
//
// Only a control whose scope is the calling thread can do this. CMake finds
// which vendor the BLAS/LAPACK libraries are and defines one of
//   MTL5_BLAS_VENDOR_MKL       MKL_Set_Num_Threads_Local: this thread
//   MTL5_BLAS_VENDOR_BLIS      bli_thread_set_num_threads: this thread (BLIS >= 0.9)
//   MTL5_BLAS_VENDOR_OPENBLAS  openblas_set_num_threads: the whole process
// OpenBLAS has no per-thread control, so it is never changed here: changing a
// process-wide setting would also make the application's direct calls serial.
// docs/algorithms/on-node-threading.md describes the choice the application
// makes for it. Without a detected vendor every cap is empty.

#include <mtl/parallel/thread_pool.hpp>

#include <cstdint>

#if defined(MTL5_BLAS_VENDOR_MKL)
// MKL's C entry points. The lower-case mkl_set_num_threads_local and
// mkl_get_max_threads exported beside them are the Fortran interfaces, which
// take their arguments by pointer; mkl.h maps the lower-case names to these
// with macros, so the declarations agree with it.
extern "C" {
int MKL_Set_Num_Threads_Local(int nt);
int MKL_Get_Max_Threads(void);
}
#elif defined(MTL5_BLAS_VENDOR_BLIS)
extern "C" {
// dim_t is BLIS's default 64-bit integer.
void bli_thread_set_num_threads(std::int64_t n_threads);
std::int64_t bli_thread_get_num_threads(void);
}
#elif defined(MTL5_BLAS_VENDOR_OPENBLAS)
extern "C" {
int openblas_get_num_threads(void);
}
#endif

namespace mtl::interface {

/// The BLAS/LAPACK vendor CMake detected: "mkl", "blis", "openblas" or "".
constexpr const char* blas_vendor() noexcept {
#if defined(MTL5_BLAS_VENDOR_MKL)
    return "mkl";
#elif defined(MTL5_BLAS_VENDOR_BLIS)
    return "blis";
#elif defined(MTL5_BLAS_VENDOR_OPENBLAS)
    return "openblas";
#else
    return "";
#endif
}

/// Whether a vendor call made inside an MTL5 region runs on one thread.
constexpr bool blas_capped_in_region() noexcept {
#if defined(MTL5_BLAS_VENDOR_MKL) || defined(MTL5_BLAS_VENDOR_BLIS)
    return true;
#else
    return false;
#endif
}

/// The number of threads the detected vendor would use for a call made on
/// this thread now; 0 when no vendor was detected.
inline int blas_vendor_threads() noexcept {
#if defined(MTL5_BLAS_VENDOR_MKL)
    return MKL_Get_Max_Threads();
#elif defined(MTL5_BLAS_VENDOR_BLIS)
    return static_cast<int>(bli_thread_get_num_threads());
#elif defined(MTL5_BLAS_VENDOR_OPENBLAS)
    return openblas_get_num_threads();
#else
    return 0;
#endif
}

namespace detail {

#if defined(MTL5_BLAS_VENDOR_MKL) || defined(MTL5_BLAS_VENDOR_BLIS)
/// The vendor setting a region's calling thread had before its first capped
/// call, restored when that thread leaves the region.
inline std::int64_t& vendor_threads_before_region() noexcept {
    thread_local std::int64_t prev = 0;
    return prev;
}

inline void restore_vendor_threads() noexcept {
#if defined(MTL5_BLAS_VENDOR_MKL)
    MKL_Set_Num_Threads_Local(static_cast<int>(vendor_threads_before_region()));
#else
    bli_thread_set_num_threads(vendor_threads_before_region());
#endif
    parallel::detail::this_region_thread().vendor_capped = false;
}
#endif

/// Before a vendor call: inside an MTL5 region, the vendor runs on this thread
/// only; outside one, nothing happens beyond a thread-local read.
///
/// The vendor is set once per region, not once per call (with BLIS a set costs
/// more than a small ddot). On the region's calling thread the previous
/// setting is restored when the thread leaves the region. On a pool worker,
/// which only ever runs region tids, it stays at one.
class nested_blas_cap {
public:
    nested_blas_cap() noexcept {
#if defined(MTL5_BLAS_VENDOR_MKL) || defined(MTL5_BLAS_VENDOR_BLIS)
        if (!parallel::detail::in_region()) return;
        parallel::detail::region_thread& t = parallel::detail::this_region_thread();
        if (t.vendor_capped) return;
        t.vendor_capped = true;
#  if defined(MTL5_BLAS_VENDOR_MKL)
        const std::int64_t prev = MKL_Set_Num_Threads_Local(1);   // 0: the global setting
#  else
        const std::int64_t prev = bli_thread_get_num_threads();
        if (prev != 1) bli_thread_set_num_threads(1);
#  endif
        if (t.state != parallel::detail::pool_worker) {
            vendor_threads_before_region() = prev;
            t.on_exit = &restore_vendor_threads;
        }
#endif
    }
    nested_blas_cap(const nested_blas_cap&) = delete;
    nested_blas_cap& operator=(const nested_blas_cap&) = delete;
};

/// The same for a CHOLMOD/SPQR call: caps CHOLMOD's own OpenMP team through
/// cholmod_common::nthreads_max, and the BLAS it calls, inside a region.
/// Common is cholmod_common, a template parameter so this header does not
/// need cholmod.h.
template <typename Common>
class nested_suitesparse_cap {
public:
    explicit nested_suitesparse_cap(Common& c) noexcept : c_(c), prev_(c.nthreads_max) {
        if (parallel::detail::in_region()) c_.nthreads_max = 1;
    }
    ~nested_suitesparse_cap() { c_.nthreads_max = prev_; }
    nested_suitesparse_cap(const nested_suitesparse_cap&) = delete;
    nested_suitesparse_cap& operator=(const nested_suitesparse_cap&) = delete;

private:
    nested_blas_cap blas_;
    Common& c_;
    int prev_;
};

} // namespace detail
} // namespace mtl::interface
