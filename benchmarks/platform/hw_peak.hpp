#pragma once
// mtl5_bench -- the HARDWARE FMA peak, independent of the build (#721 M2).
//
// probes.hpp measures the FMA peak of simd::batch, i.e. of the SIMD target the
// build compiled. That is the right ceiling for "how close do MTL5's kernels get
// in this build", and the wrong one for characterizing the MACHINE: a build
// without Highway has a scalar batch, so its "peak" is a scalar FMA, and its
// BLAS-backed or auto-vectorized GEMM then reads as thousands of percent of it
// (2300% native, 39000% BLAS, on the first run of platform.sh). A card has to
// say how much of the machine a build reaches, so the roofline needs the
// machine's ceiling.
//
// So this probe picks the widest vector ISA the CPU AND the operating system
// support, at run time, and runs a register-resident kernel written in that
// ISA's intrinsics -- compiled for it with a target attribute, whatever the
// build's flags:
//   x86     AVX-512F, else AVX2 + FMA, else SSE2 (mul + add, no FMA). "Supported"
//           means CPUID says so AND XCR0 shows the OS saves the registers.
//   AArch64 NEON (mandatory). SVE wider than 128 bits (Neoverse V1, A64FX) is
//           NOT covered: on those the build's probe, if built for SVE, is the
//           higher one, and the runner takes the max of the two.
//   other   none; the build's probe is the only ceiling.

#include <cstddef>
#include <string>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#  define MTL5_HW_PEAK_X86 1
#  include <immintrin.h>
#  if defined(_MSC_VER) && !defined(__clang__)
#    include <intrin.h>
#    define MTL5_HW_TARGET(isa)   // MSVC compiles any intrinsic in any function
#  else
#    include <cpuid.h>
#    define MTL5_HW_TARGET(isa) __attribute__((target(isa)))
#  endif
#elif defined(__aarch64__) || defined(_M_ARM64)
#  define MTL5_HW_PEAK_ARM64 1
#  include <arm_neon.h>
#endif

namespace mtl::bench::platform::hw {

/// One thread's kernel: `iters` rounds over independent accumulator chains.
/// Returns a value depending on every chain, so none is dead code.
using kernel_fn = double (*)(std::size_t iters, double seed);

struct isa_kernels {
    std::string isa = "none";
    kernel_fn f64 = nullptr;
    kernel_fn f32 = nullptr;
    double f64_flops_per_iter = 0.0;   // per thread, per outer iteration
    double f32_flops_per_iter = 0.0;
};

// How every kernel here is written, and why (each point was a measured failure):
//
// * NAMED ACCUMULATORS, never an array. With `V acc[16]`, GCC for aarch64 kept
//   the array in memory and stored every accumulator on every iteration (8 stp
//   per 16 fmla): the loop was store-bound and the NEON "peak" on Neoverse N2 read
//   18.1 GFLOP/s f64 where the core sustains at least 27. Named locals are always
//   register-allocated. The chain lists below generate them.
// * ACCUMULATE INTO THE ADDEND: acc = m*c + acc. AArch64's FMLA adds only into
//   its destination; with the accumulator as a multiplicand (acc*m + c) GCC
//   emitted two vector moves per FMA, and the peak read 6.8. x86 FMA has a form
//   for every operand order, so neither problem showed there.
//   Each chain grows by 1e-6 per iteration: bounded and normal for any run length
//   the timing driver chooses.
// * SSE2 has no FMA, so its kernel keeps acc*m + c (two instructions, two flops):
//   with m*c + acc, the product would be loop-invariant and the loop would time
//   adds only.
//
// Chains: 12 on x86 (FMA latency 4 x 2 ports with headroom, inside the 16
// registers of SSE/AVX2 with the two constants), 16 on AArch64 (the V-class
// Neoverse cores issue four 128-bit FMAs per cycle at latency 4; 32 registers).
inline constexpr int kChains = 12;
inline constexpr int kChainsNeon = 16;

#define MTL5_HW_C12(X) X(0) X(1) X(2) X(3) X(4) X(5) X(6) X(7) X(8) X(9) X(10) X(11)
#define MTL5_HW_C16(X) MTL5_HW_C12(X) X(12) X(13) X(14) X(15)
// The three steps of a kernel, in terms of per-kernel MTL5_V, MTL5_SET, MTL5_STEP
// and MTL5_ADD, which each kernel defines and undefines around its body.
#define MTL5_HW_DECL(i) MTL5_V a##i = MTL5_SET(seed + (i));
#define MTL5_HW_UPDATE(i) a##i = MTL5_STEP(a##i);
#define MTL5_HW_SUM(i) if ((i) != 0) a0 = MTL5_ADD(a0, a##i);
#define MTL5_HW_BODY(CHAINS)                                                   \
    CHAINS(MTL5_HW_DECL)                                                       \
    for (std::size_t it = 0; it < iters; ++it) { CHAINS(MTL5_HW_UPDATE) }      \
    CHAINS(MTL5_HW_SUM)

#if defined(MTL5_HW_PEAK_X86)

namespace detail {
inline void cpuid(unsigned leaf, unsigned sub, unsigned r[4]) {
#  if defined(_MSC_VER) && !defined(__clang__)
    int v[4];
    __cpuidex(v, static_cast<int>(leaf), static_cast<int>(sub));
    for (int i = 0; i < 4; ++i) r[i] = static_cast<unsigned>(v[i]);
#  else
    __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
#  endif
}
inline unsigned long long xcr0() {
#  if defined(_MSC_VER) && !defined(__clang__)
    return _xgetbv(0);
#  else
    unsigned eax = 0, edx = 0;
    __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    return (static_cast<unsigned long long>(edx) << 32) | eax;
#  endif
}
} // namespace detail

MTL5_HW_TARGET("avx512f")
inline double avx512_f64(std::size_t iters, double seed) {
#define MTL5_V __m512d
#define MTL5_SET(x) _mm512_set1_pd(x)
#define MTL5_STEP(a) _mm512_fmadd_pd(m, c, a)
#define MTL5_ADD(a, b) _mm512_add_pd(a, b)
    const __m512d m = _mm512_set1_pd(1e-3), c = _mm512_set1_pd(1e-3);
    MTL5_HW_BODY(MTL5_HW_C12)
#undef MTL5_V
#undef MTL5_SET
#undef MTL5_STEP
#undef MTL5_ADD
    double o[8];
    _mm512_storeu_pd(o, a0);
    return o[0] + o[1] + o[2] + o[3] + o[4] + o[5] + o[6] + o[7];
}
MTL5_HW_TARGET("avx512f")
inline double avx512_f32(std::size_t iters, double seed) {
#define MTL5_V __m512
#define MTL5_SET(x) _mm512_set1_ps(static_cast<float>(x))
#define MTL5_STEP(a) _mm512_fmadd_ps(m, c, a)
#define MTL5_ADD(a, b) _mm512_add_ps(a, b)
    const __m512 m = _mm512_set1_ps(1e-3f), c = _mm512_set1_ps(1e-3f);
    MTL5_HW_BODY(MTL5_HW_C12)
#undef MTL5_V
#undef MTL5_SET
#undef MTL5_STEP
#undef MTL5_ADD
    float o[16];
    _mm512_storeu_ps(o, a0);
    double r = 0;
    for (float v : o) r += v;
    return r;
}
MTL5_HW_TARGET("avx2,fma")
inline double avx2_f64(std::size_t iters, double seed) {
#define MTL5_V __m256d
#define MTL5_SET(x) _mm256_set1_pd(x)
#define MTL5_STEP(a) _mm256_fmadd_pd(m, c, a)
#define MTL5_ADD(a, b) _mm256_add_pd(a, b)
    const __m256d m = _mm256_set1_pd(1e-3), c = _mm256_set1_pd(1e-3);
    MTL5_HW_BODY(MTL5_HW_C12)
#undef MTL5_V
#undef MTL5_SET
#undef MTL5_STEP
#undef MTL5_ADD
    double o[4];
    _mm256_storeu_pd(o, a0);
    return o[0] + o[1] + o[2] + o[3];
}
MTL5_HW_TARGET("avx2,fma")
inline double avx2_f32(std::size_t iters, double seed) {
#define MTL5_V __m256
#define MTL5_SET(x) _mm256_set1_ps(static_cast<float>(x))
#define MTL5_STEP(a) _mm256_fmadd_ps(m, c, a)
#define MTL5_ADD(a, b) _mm256_add_ps(a, b)
    const __m256 m = _mm256_set1_ps(1e-3f), c = _mm256_set1_ps(1e-3f);
    MTL5_HW_BODY(MTL5_HW_C12)
#undef MTL5_V
#undef MTL5_SET
#undef MTL5_STEP
#undef MTL5_ADD
    float o[8];
    _mm256_storeu_ps(o, a0);
    double r = 0;
    for (float v : o) r += v;
    return r;
}
// SSE2 has no FMA: a multiply and an add per lane, two flops, two instructions.
inline double sse2_f64(std::size_t iters, double seed) {
#define MTL5_V __m128d
#define MTL5_SET(x) _mm_set1_pd(x)
#define MTL5_STEP(a) _mm_add_pd(_mm_mul_pd(a, m), c)
#define MTL5_ADD(a, b) _mm_add_pd(a, b)
    const __m128d m = _mm_set1_pd(0.999999), c = _mm_set1_pd(1e-6);
    MTL5_HW_BODY(MTL5_HW_C12)
#undef MTL5_V
#undef MTL5_SET
#undef MTL5_STEP
#undef MTL5_ADD
    double o[2];
    _mm_storeu_pd(o, a0);
    return o[0] + o[1];
}
inline double sse2_f32(std::size_t iters, double seed) {
#define MTL5_V __m128
#define MTL5_SET(x) _mm_set1_ps(static_cast<float>(x))
#define MTL5_STEP(a) _mm_add_ps(_mm_mul_ps(a, m), c)
#define MTL5_ADD(a, b) _mm_add_ps(a, b)
    const __m128 m = _mm_set1_ps(0.999999f), c = _mm_set1_ps(1e-6f);
    MTL5_HW_BODY(MTL5_HW_C12)
#undef MTL5_V
#undef MTL5_SET
#undef MTL5_STEP
#undef MTL5_ADD
    float o[4];
    _mm_storeu_ps(o, a0);
    return double(o[0]) + o[1] + o[2] + o[3];
}

inline isa_kernels select() {
    unsigned r[4];
    detail::cpuid(0, 0, r);
    const unsigned max_leaf = r[0];
    detail::cpuid(1, 0, r);
    const bool sse2 = (r[3] >> 26) & 1u, fma = (r[2] >> 12) & 1u;
    const bool osxsave = (r[2] >> 27) & 1u, avx = (r[2] >> 28) & 1u;
    bool avx2 = false, avx512f = false;
    if (max_leaf >= 7) {
        detail::cpuid(7, 0, r);
        avx2 = (r[1] >> 5) & 1u;
        avx512f = (r[1] >> 16) & 1u;
    }
    const unsigned long long x = osxsave ? detail::xcr0() : 0;
    const bool os_ymm = (x & 0x6) == 0x6;     // XMM + YMM state
    const bool os_zmm = (x & 0xE6) == 0xE6;   // + opmask, ZMM_Hi256, Hi16_ZMM
    if (avx512f && os_zmm)
        return {"AVX-512F", avx512_f64, avx512_f32, 2.0 * 8 * kChains, 2.0 * 16 * kChains};
    if (avx && avx2 && fma && os_ymm)
        return {"AVX2+FMA", avx2_f64, avx2_f32, 2.0 * 4 * kChains, 2.0 * 8 * kChains};
    if (sse2)
        return {"SSE2", sse2_f64, sse2_f32, 2.0 * 2 * kChains, 2.0 * 4 * kChains};
    return {};
}

#elif defined(MTL5_HW_PEAK_ARM64)

inline double neon_f64(std::size_t iters, double seed) {
#define MTL5_V float64x2_t
#define MTL5_SET(x) vdupq_n_f64(x)
#define MTL5_STEP(a) vfmaq_f64(a, m, c)   // a + m*c
#define MTL5_ADD(a, b) vaddq_f64(a, b)
    const float64x2_t m = vdupq_n_f64(1e-3), c = vdupq_n_f64(1e-3);
    MTL5_HW_BODY(MTL5_HW_C16)
#undef MTL5_V
#undef MTL5_SET
#undef MTL5_STEP
#undef MTL5_ADD
    return vgetq_lane_f64(a0, 0) + vgetq_lane_f64(a0, 1);
}
inline double neon_f32(std::size_t iters, double seed) {
#define MTL5_V float32x4_t
#define MTL5_SET(x) vdupq_n_f32(static_cast<float>(x))
#define MTL5_STEP(a) vfmaq_f32(a, m, c)
#define MTL5_ADD(a, b) vaddq_f32(a, b)
    const float32x4_t m = vdupq_n_f32(1e-3f), c = vdupq_n_f32(1e-3f);
    MTL5_HW_BODY(MTL5_HW_C16)
#undef MTL5_V
#undef MTL5_SET
#undef MTL5_STEP
#undef MTL5_ADD
    return double(vgetq_lane_f32(a0, 0)) + vgetq_lane_f32(a0, 1) + vgetq_lane_f32(a0, 2) +
           vgetq_lane_f32(a0, 3);
}

inline isa_kernels select() {
    return {"NEON", neon_f64, neon_f32, 2.0 * 2 * kChainsNeon, 2.0 * 4 * kChainsNeon};
}

#else

inline isa_kernels select() { return {}; }

#endif

} // namespace mtl::bench::platform::hw
