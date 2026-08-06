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
    // APPENDED, not inserted (#351). hw_traits is a public aggregate and is
    // initialized positionally in tests and downstream code; inserting a field
    // mid-struct silently shifts every later one -- during this change that
    // turned an l1_bytes of 32 KB into 8 bytes with no diagnostic. Appending
    // means an initializer that predates this field value-initializes it to 0,
    // which derive_blocking treats as "unknown" and falls back to the
    // latency-floor tile, i.e. the pre-#351 behaviour.
    std::size_t vec_registers; // Nreg: architectural vector registers (0 = unknown)
};

/// Generic modern-x86 (AVX2-class) default; override per architecture.
/// Matches a Haswell-class core: 32 KB/8-way L1, 256 KB L2, 8 MB L3,
/// FMA latency 4, 2 FMA units, 16 ymm registers.
inline constexpr hw_traits default_hw_traits{
    /*fma_latency*/ 4, /*fma_units*/ 2,
    /*l1_bytes*/ 32u * 1024, /*l1_assoc*/ 8, /*line_bytes*/ 64,
    /*l2_bytes*/ 256u * 1024,
    /*l3_bytes*/ 8u * 1024 * 1024,
    /*page_bytes*/ 4096,
    /*vec_registers*/ 16,
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

    // Register block. Two constraints, and which one BINDS is the whole point:
    //
    //   floor    mr*nr >= Nvec*Lvfma*Nvfma -- enough accumulators to cover
    //            dependent-FMA latency at the issue width. Eq. 1 of the BLIS
    //            analytical model. It is a LOWER bound.
    //   ceiling  the accumulators, plus the operands they are multiplied by,
    //            must FIT the architectural vector register file. Exceed it and
    //            the microkernel spills every k step.
    //
    // This used to size the tile at exactly the floor -- `area` was used as the
    // target rather than the minimum -- and there was no register-file term at
    // all. On an AVX2 core that gave 4x8 for double: 8 accumulator vectors out
    // of 16 ymm, half the file idle. Measured cost (#351, i7-12700K, one pinned
    // P-core, fp64, median of 3):
    //
    //     tile   acc vectors   N=1024   N=2048
    //     4x8         8         57.90    56.90     <- floor-sized, was default
    //     5x8        10         63.41    64.16
    //     6x8        12         67.57    66.96     <- +17%
    //     8x8        16         57.68    57.03     <- fills the file, spills
    //     8x12       24         55.31    55.72
    //
    // Both directions cost: too few accumulators cannot hide latency, too many
    // leave nothing for operands. The peak sits at ~3/4 of the register file,
    // which is also where BLIS's hand-written Haswell dgemm kernel sits (6x8),
    // and where its AVX-512 kernel sits (8x24 = 24 of 32).
    //
    // So: budget 3/4 of the file for accumulators, hold the B micro-panel in
    // NRVEC vector registers, and let the floor raise mr if a narrow file would
    // otherwise put us under it.
    constexpr std::size_t NRVEC = 2;                 // B panel width, in vectors
    const std::size_t area_floor = nvec * hw.fma_latency * hw.fma_units;

    std::size_t nr, mr;
    if (hw.vec_registers == 0) {
        // Register file unknown (an initializer predating this field): fall back
        // to the pre-#351 near-square tile sized at the latency floor.
        nr = round_up(isqrt_ceil(area_floor), nvec);
        if (nr == 0) nr = nvec;
        mr = ceil_div(area_floor, nr);
        if (mr == 0) mr = 1;
    } else {
        nr = nvec * NRVEC;
        if (nr == 0) nr = nvec;
        // Accumulator budget, leaving room for the B panel and the A broadcast.
        std::size_t acc_budget = (hw.vec_registers * 3) / 4;
        if (acc_budget < NRVEC + 1) acc_budget = NRVEC + 1;
        mr = acc_budget / NRVEC;
        if (mr == 0) mr = 1;
        // The floor is a floor: if a narrow file put us under it, fall back to
        // the near-square shape rather than growing mr alone into a degenerate
        // aspect ratio.
        if (mr * nr < area_floor) {
            nr = round_up(isqrt_ceil(area_floor), nvec);
            if (nr == 0) nr = nvec;
            mr = ceil_div(area_floor, nr);
            if (mr == 0) mr = 1;
        }
    }

    // kc: B micro-panel (kc x nr) occupies ~half of L1.
    std::size_t kc = (hw.l1_bytes / 2) / (nr * sdata);
    if (kc == 0) kc = 1;

    // mc: packed A block (mc x kc) occupies ~half of L2.
    //
    // NOT rounded to a multiple of mr (#408). That coupling made a CACHE
    // quantity depend on the REGISTER TILE, and the dependency is destructive:
    // the L2 budget here yields exactly 64, which survives mr = 4 but drops to
    // 60 under #382's mr = 6. That moved the ic-block count at m = 1024 from 16
    // (exactly 2.00 blocks per thread on 8 threads) to 18 (2.25), a 1.41x
    // critical path that turned #382's +21.5% single-thread win into a -7.4%
    // eight-thread regression.
    //
    // The rounding was never a correctness requirement -- pack_A already handles
    // a ragged final panel, since m is not a multiple of mr in general. Dropping
    // it costs one partial panel per block instead of one per matrix, which
    // measured within noise single-threaded, and it lets detail::balanced_mc
    // choose the block size against the thread count at runtime, where m and
    // ic_nt are actually known.
    std::size_t mc = (hw.l2_bytes / 2) / (kc * sdata);
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
