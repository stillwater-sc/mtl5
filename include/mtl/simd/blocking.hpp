#pragma once
// MTL5 -- hardware traits + compile-time GEMM blocking-parameter derivation
// (#85, epic #82, Phase 0).
//
// The GEMM micro-kernel (#88) and the cache-blocking loop nest (#90) need a
// register tile (mr x nr) and cache block sizes (kc, mc, nc). Rather than
// search for them empirically, derive them at compile time from a hardware
// description using the BLIS analytical model:
//
//   Low, Igual, Smith, Quintana-Orti, "Analytical Modeling Is Enough for
//   High-Performance BLIS", ACM TOMS. https://www.cs.utexas.edu/~flame/pubs/TOMS-BLIS-Analytical.pdf
//
// Register block (Eqs. 1-3): the mr x nr C microtile must hold enough
// independent FMA accumulators to cover the FMA pipeline:
//     mr * nr  >=  Nvec * Lvfma * Nvfma
// with one dimension (here nr, the vector dimension) a multiple of the SIMD
// width Nvec. Cache blocks (Eq. 4 and analogues): kc sizes the B micro-panel
// to ~half of L1; mc sizes the packed A block to ~half of L2; nc sizes the
// packed B panel to ~L3.

#include <cstddef>

#include <mtl/simd/batch.hpp>   // simd::width<T>

namespace mtl::simd {

// -- small constexpr integer helpers ---------------------------------------
namespace detail {
constexpr std::size_t ceil_div(std::size_t a, std::size_t b) { return b == 0 ? 0 : (a + b - 1) / b; }
constexpr std::size_t round_up(std::size_t n, std::size_t m)   { return m == 0 ? n : ceil_div(n, m) * m; }
constexpr std::size_t round_down(std::size_t n, std::size_t m) { return m == 0 ? n : (n / m) * m; }
constexpr std::size_t isqrt_ceil(std::size_t n) {            // smallest r with r*r >= n
    std::size_t r = 0;
    while (r * r < n) ++r;
    return r;
}
} // namespace detail

/// Hardware description for blocking-parameter derivation. All sizes in bytes.
struct hw_traits {
    std::size_t fma_latency;   // Lvfma: dependent-FMA latency (cycles)
    std::size_t fma_units;     // Nvfma: FMA issue width
    std::size_t l1_bytes;      // per-core L1 data cache
    std::size_t l1_assoc;      // L1 associativity (reserved for refinement)
    std::size_t line_bytes;    // cache line
    std::size_t l2_bytes;      // per-core L2
    std::size_t l3_bytes;      // shared L3 (per core group)
    std::size_t page_bytes;    // page size (TLB reasoning)
};

/// Generic modern-x86 (AVX2-class) default; override per architecture.
/// Matches a Haswell-class core: 32 KB/8-way L1, 256 KB L2, 8 MB L3,
/// FMA latency 4, 2 FMA units.
inline constexpr hw_traits default_hw_traits{
    /*fma_latency*/ 4, /*fma_units*/ 2,
    /*l1_bytes*/ 32u * 1024, /*l1_assoc*/ 8, /*line_bytes*/ 64,
    /*l2_bytes*/ 256u * 1024,
    /*l3_bytes*/ 8u * 1024 * 1024,
    /*page_bytes*/ 4096,
};

/// GEMM blocking parameters: mr x nr register microtile, kc/mc/nc cache blocks.
struct blocking_params {
    std::size_t mr, nr, kc, mc, nc;
};

/// Derive GEMM blocking parameters for element type `T` with `nvec` SIMD lanes
/// (e.g. simd::width<T>) on hardware `hw`. constexpr / pure integer.
template <typename T>
constexpr blocking_params derive_blocking(std::size_t nvec,
                                          const hw_traits& hw = default_hw_traits) {
    using detail::ceil_div; using detail::round_up; using detail::round_down; using detail::isqrt_ceil;
    const std::size_t sdata = sizeof(T);
    if (nvec == 0) nvec = 1;

    // Register block (Eqs. 1-3): area >= Nvec*Lvfma*Nvfma, near-square, nr a
    // multiple of the SIMD width (nr is the vectorized dimension).
    const std::size_t area = nvec * hw.fma_latency * hw.fma_units;
    std::size_t nr = round_up(isqrt_ceil(area), nvec);
    if (nr == 0) nr = nvec;
    std::size_t mr = ceil_div(area, nr);
    if (mr == 0) mr = 1;

    // kc: B micro-panel (kc x nr) occupies ~half of L1.
    std::size_t kc = (hw.l1_bytes / 2) / (nr * sdata);
    if (kc == 0) kc = 1;

    // mc: packed A block (mc x kc) occupies ~half of L2; multiple of mr.
    std::size_t mc = round_down((hw.l2_bytes / 2) / (kc * sdata), mr);
    if (mc == 0) mc = mr;

    // nc: packed B panel (kc x nc) occupies ~L3; multiple of nr.
    std::size_t nc = round_down(hw.l3_bytes / (kc * sdata), nr);
    if (nc == 0) nc = nr;

    return {mr, nr, kc, mc, nc};
}

/// Blocking parameters for `T` using the compiled SIMD width and the default
/// hardware traits. (#90 will let the hw_traits be overridden per build.)
template <typename T>
inline constexpr blocking_params default_blocking = derive_blocking<T>(width<T>);

} // namespace mtl::simd
