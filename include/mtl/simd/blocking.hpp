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

#include <mtl/simd/batch.hpp>        // simd::width<T>
#include <mtl/util/cache_info.hpp>   // util::cached_cache_info (#222)

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
/// hardware traits. Compile-time, and the source of the mr x nr register tile
/// (a micro-kernel template argument) everywhere.
template <typename T>
inline constexpr blocking_params default_blocking = derive_blocking<T>(width<T>);

/// `default_hw_traits` with the CACHE fields replaced by what the running machine
/// reports (#222). Detected once per process; a field the platform could not
/// report keeps its compile-time default, so an undetectable machine behaves
/// exactly as it did before this existed.
///
/// The FMA latency/issue width and the vector register file are NOT overridden.
/// They determine mr x nr, which is a template argument of the compiled
/// micro-kernel: a runtime value could not be applied to it, and pretending
/// otherwise would produce a tile the kernel does not implement. Keeping them
/// compile-time is also what guarantees `runtime_blocking<T>().mr == default_blocking<T>.mr`
/// (likewise nr), so the two agree on the register tile by construction and only
/// the cache blocks can differ.
/// Overlay detected cache figures on a base description, field by field. Pure,
/// so the fallback is testable with hierarchies the host does not have.
///
/// A 0 field means NOT DETECTED and keeps the base value.
///
/// L3 IS DELIBERATELY NOT APPLIED, though cache_info detects it. l3_bytes feeds
/// only nc, and nc sets the jc BLOCK COUNT, njb = ceil(n/nc) -- which the
/// threaded nest hands to jc-teams round-robin, so the critical path is
/// ceil(njb/jc_nt) and an njb that is not a multiple of jc_nt leaves teams
/// unequal. Measured (Xeon E5-2420 v2, 6 threads, m=96 n=16384 k=1024, fp64):
/// applying the detected 15 MB L3 moved nc 2048 -> 3840 and njb 8 -> 5, so two
/// teams took 3 and 2 blocks instead of 4 and 4 -- a 1.2x critical path and a
/// measured 10-25% REGRESSION. kc and mc carry no such hazard: they size blocks
/// within a partition rather than the number of partitioned units.
///
/// This is #408's failure mode one loop over -- there a register-tile change
/// moved mc 64 -> 60 and nib 16 -> 18, unbalancing the ic partition. The ic loop
/// has detail::balanced_mc to absorb it; the jc loop has no equivalent yet. Once
/// a balanced_nc exists, L3 can be applied here. Tracked separately so that
/// partition change gets its own measurement rather than riding along with cache
/// detection.
///
/// A SHARED cache contributes only its per-core share. The BLIS model wants the
/// L1/L2 one core gets to itself, and `cache_info` reports the instance size
/// alongside how many physical cores share it, so the division happens here where
/// the model is. It matters: an Alder Lake E-cluster publishes one 2 MiB L2 for
/// four cores, and taking that at face value sized `mc` for roughly 4x the L2 a
/// core actually has (#432). SMT siblings are deliberately NOT counted as sharers
/// -- see cache_info -- so a hyperthreaded machine is unaffected.
constexpr hw_traits with_detected_caches(hw_traits base, const util::cache_info& c) {
    const std::size_t l1_share = c.l1d_sharing_cores ? c.l1d_sharing_cores : 1;
    const std::size_t l2_share = c.l2_sharing_cores  ? c.l2_sharing_cores  : 1;
    if (c.l1d_bytes  != 0) base.l1_bytes   = c.l1d_bytes / l1_share;   // -> kc
    if (c.l1d_assoc  != 0) base.l1_assoc   = c.l1d_assoc;
    if (c.line_bytes != 0) base.line_bytes = c.line_bytes;
    if (c.l2_bytes   != 0) base.l2_bytes   = c.l2_bytes / l2_share;    // -> mc
    return base;
}

/// Runtime cache detection is OPT-IN: define MTL5_ENABLE_CACHE_DETECTION to let
/// the detected hierarchy feed kc/mc. Without it this returns the compile-time
/// defaults, which is what MTL5 ships.
///
/// That is not caution, it is a measurement. Detection was wired in by #426 on
/// the assumption that real cache sizes must beat figures hardcoded to a
/// Haswell-class core. Measured on an i7-12700K (Golden Cove, P-cores pinned,
/// AVX2, fp64, 5 interleaved rounds, min of 5) it LOST on 9 of 10 points:
///
///     detected  kc=384 mc=213      default  kc=256 mc=64
///
///     shape            T=1     T=8
///     1024^3          0.965   0.551      <- 3 of 8 threads left idle
///     2048^3          0.920   0.929
///     4096^3          0.901   0.985
///     852 x 8192      0.914   0.806      <- two 12.6 MB B panels in a 25 MB L3
///     852 x 12288     0.906   0.675
///     (ratio = detected / default throughput; < 1 is a loss)
///
/// Three distinct causes, and only the first is about cache sizing:
///
///   1. At T=1 the larger kc/mc are slower at every size -- 3.5% at 1024^3 and
///      8-10% at the rest. The analytical "half of L1, half of L2" sizing is
///      beaten by the hand-tuned constants on this microarchitecture. No
///      threading is involved.
///   2. A larger mc means FEWER ic blocks, and gemm_blocked fixes
///      ic_nt = min(budget, nib) from nib BEFORE balanced_mc runs. At 1024^3,
///      mc 64 -> 213 takes nib 16 -> 5, so five threads of eight do the work.
///   3. A larger kc inflates the packed B panel (kc x nc), and once the jc loop
///      splits into teams each team holds one: 2 x 12.6 MB against a 25 MB L3.
///
/// (2) and (3) are the #408 / #429 family -- a cache-derived parameter moving a
/// BLOCK COUNT that the thread partition is sensitive to. They are real defects
/// independent of detection, and detection is what exposed them.
///
/// So the machinery stays, tested and available, and the shipped blocking is
/// byte-identical to pre-#426 until the model is shown to win on hardware.
/// benchmarks/run_blocking_ab.sh is how that is decided; #430 is the experiment.
inline const hw_traits& detected_hw_traits() {
#if defined(MTL5_ENABLE_CACHE_DETECTION)
    static const hw_traits hw =
        with_detected_caches(default_hw_traits, util::cached_cache_info());
    return hw;
#else
    return default_hw_traits;   // constexpr object: static storage, safe to bind
#endif
}

/// Blocking parameters for `T` derived against the DETECTED hardware. Use kc and
/// mc from this; take mr/nr from `default_blocking<T>` (the tile the micro-kernel
/// was instantiated with) and nc from there too, since L3 is not overridden --
/// see with_detected_caches. Computed once per element type.
template <typename T>
inline const blocking_params& runtime_blocking() {
    static const blocking_params bp = [] {
        blocking_params p = derive_blocking<T>(width<T>, detected_hw_traits());
        // Pin nc to the compile-time derivation. Withholding l3_bytes from
        // with_detected_caches is NOT sufficient to hold nc still, because
        // nc = round_down(l3/(kc*sdata), nr) and kc IS detected: a machine with a
        // larger L1 gets a larger kc and therefore a smaller nc, from an
        // unchanged L3. (Invisible on a 32 KB-L1 x86 host, where kc matches the
        // default; an Apple M-series 128 KB L1d moves it four-fold.) Since the jc
        // partition is what must not move (#429), pin the result rather than an
        // input, and leave this struct saying exactly what the nest should use.
        p.nc = default_blocking<T>.nc;
        return p;
    }();
    return bp;
}

} // namespace mtl::simd
