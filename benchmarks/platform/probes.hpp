#pragma once
// mtl5_bench -- machine probes (#721).
//
// Every rate the catalogue reports is placed against what THIS machine,
// running THIS build, can sustain, so "% of roofline" means the same thing on a
// Jetson and on a Ryzen. Two ceilings, measured, never looked up:
//
//   FMA peak   register-resident kernels with independent accumulators, no
//              loads, no stores, measured twice:
//                build     over simd::batch<T>, the SIMD target this build
//                          compiled (the same batch the GEMM micro-kernel uses)
//                hardware  in the widest ISA the CPU and OS support, picked at
//                          run time whatever the build's flags (hw_peak.hpp)
//              The roofline uses the HIGHER of the two, so every build on a
//              machine is judged against the same, machine-level ceiling: a
//              build without Highway has a scalar batch, and judging it by its
//              own scalar peak put its GEMM at thousands of percent.
//   bandwidth  two streams over arrays of at least four times the last-level
//              cache:
//                read   sum of a[i] + b[i] + c[i] (SIMD): 24 B/elem, no stores
//                triad  a[i] = b[i] + s*c[i], counted as 24 B/elem, the STREAM
//                       convention
//                update a[i] = a[i] + s*b[i] (the axpy pattern): 24 B/elem
//              The roofline uses the HIGHEST of the three, because the memory
//              system's best rate depends on the access pattern. Triad alone understates
//              what the memory system delivers: storing to a[] costs a hidden
//              read-for-ownership of each line, so its 24 counted bytes move 32,
//              and a read-dominated kernel (dot, gemv) then measures well above
//              "100%" of it -- 130-150% on the first machine this ran on.
//
// Both run on `threads` threads: the size of MTL5's pool, which is what the
// operations themselves use (MTL5_NUM_THREADS). Comparing a 1-thread GEMM with
// an all-core peak would understate it by the core count.

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <mtl/simd/batch.hpp>
#include <mtl/simd/blocking.hpp>
#include <mtl/util/cache_info.hpp>

#include <benchmarks/platform/hw_peak.hpp>

namespace mtl::bench::platform {

struct probe_results {
    unsigned threads = 1;
    // The ceilings the roofline uses: max(build, hardware).
    double fma_peak_f64_gflops = 0.0;
    double fma_peak_f32_gflops = 0.0;
    // The two measurements behind them.
    double build_fma_peak_f64_gflops = 0.0;
    double build_fma_peak_f32_gflops = 0.0;
    double hw_fma_peak_f64_gflops = 0.0;
    double hw_fma_peak_f32_gflops = 0.0;
    std::string hw_isa = "none";
    double triad_gbytes_per_s = 0.0;
    double read_gbytes_per_s = 0.0;
    double update_gbytes_per_s = 0.0;
    std::size_t triad_array_bytes = 0;
    /// The bandwidth ceiling the roofline uses: the best of the three streams.
    double bandwidth_gbytes_per_s() const {
        return std::max({read_gbytes_per_s, triad_gbytes_per_s, update_gbytes_per_s});
    }
};

namespace detail {

using clock = std::chrono::steady_clock;

/// Seconds for `threads` copies of fn() started together.
template <typename Fn>
double run_on_threads(unsigned threads, Fn fn) {
    std::vector<std::thread> pool;
    pool.reserve(threads);
    const auto t0 = clock::now();
    for (unsigned t = 0; t < threads; ++t) pool.emplace_back([&fn, t] { fn(t); });
    for (auto& th : pool) th.join();
    return std::chrono::duration<double>(clock::now() - t0).count();
}

/// One thread's FMA kernel: ACC independent accumulator chains, `iters` deep.
/// Returns a value derived from every accumulator so none is dead code.
template <typename T, std::size_t ACC>
T fma_kernel(std::size_t iters, T seed) {
    using B = simd::batch<T>;
    B acc[ACC];
    for (std::size_t a = 0; a < ACC; ++a) acc[a] = B(seed + static_cast<T>(a));
    // x*m + c with m slightly below 1 keeps every chain bounded and normal.
    const B m(static_cast<T>(0.999999));
    const B c(static_cast<T>(1e-6));
    for (std::size_t i = 0; i < iters; ++i)
        for (std::size_t a = 0; a < ACC; ++a) acc[a] = fma(acc[a], m, c);
    B s = acc[0];
    for (std::size_t a = 1; a < ACC; ++a) s = s + acc[a];
    return reduce_add(s);
}

/// Time `kernel(iters, seed)` on `threads` threads: grow `iters` until a run
/// takes ~0.2 s, then keep the best of three. `flops_per_iter` is per thread.
template <typename Kernel>
double peak_gflops(unsigned threads, double flops_per_iter, Kernel kernel) {
    // One result slot per worker: a single shared sink written by every
    // thread is a data race, which volatile does not prevent.
    std::vector<double> sink(threads, 0.0);
    volatile double seed = 1.0;
    std::size_t iters = std::size_t(1) << 16;
    double best = 0.0;
    for (int round = 0; round < 32; ++round) {
        const double s = run_on_threads(threads, [&](unsigned t) { sink[t] = kernel(iters, seed); });
        if (s < 0.2) { iters *= 2; continue; }
        for (int rep = 0; rep < 3; ++rep) {
            const double sec = run_on_threads(threads, [&](unsigned t) { sink[t] = kernel(iters, seed); });
            best = std::max(best, flops_per_iter * double(iters) * threads / sec * 1e-9);
        }
        break;
    }
    volatile double keep = sink[0];   // the results are observable, so no chain is dead code
    (void)keep;
    return best;
}

/// The build's peak: fma_kernel over simd::batch<T>.
template <typename T>
double fma_peak_gflops(unsigned threads) {
    // Enough chains to cover latency x issue width on every core measured so
    // far (4 x 4 on the V-class Neoverse needs 16): 16 with a 32-register file,
    // 12 with 16 registers -- 3/4 of the file, leaving room for the constants.
    constexpr std::size_t ACC = simd::target_vec_registers() >= 32 ? 16 : 12;
    constexpr std::size_t W = simd::batch<T>::size;
    return peak_gflops(threads, 2.0 * double(W) * double(ACC), [](std::size_t iters, double seed) {
        return static_cast<double>(fma_kernel<T, ACC>(iters, static_cast<T>(seed)));
    });
}

/// Sum of a[i] + b[i] + c[i] over [lo, hi): THREE read streams, with
/// independent SIMD accumulators so the loop is bound by the loads, not by add
/// latency. Three streams, not one: a single core's bandwidth is limited by how
/// many misses it keeps in flight, and the hardware prefetchers track streams,
/// so one stream measured 25 GB/s on a core where a two-stream dot reached 37.
inline double stream_read(const double* a, const double* b, const double* c,
                          std::size_t lo, std::size_t hi) {
    using B = simd::batch<double>;
    constexpr std::size_t W = B::size, U = 4;
    B acc[U];
    std::size_t i = lo;
    for (; i + U * W <= hi; i += U * W)
        for (std::size_t u = 0; u < U; ++u) {
            const std::size_t k = i + u * W;
            acc[u] = acc[u] + B::load_unaligned(a + k) + B::load_unaligned(b + k) + B::load_unaligned(c + k);
        }
    B s = acc[0];
    for (std::size_t u = 1; u < U; ++u) s = s + acc[u];
    double r = reduce_add(s);
    for (; i < hi; ++i) r += a[i] + b[i] + c[i];
    return r;
}

inline void bandwidth(unsigned threads, probe_results& out) {
    double& triad_best = out.triad_gbytes_per_s;
    double& read_best = out.read_gbytes_per_s;
    std::size_t& array_bytes = out.triad_array_bytes;
    const auto& ci = util::cached_cache_info();
    const std::size_t llc = std::max({ci.l3_bytes, ci.l2_bytes, std::size_t(4) << 20});
    array_bytes = std::max<std::size_t>(4 * llc, std::size_t(32) << 20);
    const std::size_t n = array_bytes / sizeof(double);
    // Uninitialized: a std::vector would zero the arrays on THIS thread, which
    // places every page on one NUMA node before the first-touch pass below.
    std::unique_ptr<double[]> a(new double[n]), b(new double[n]), c(new double[n]);
    const std::size_t chunk = (n + threads - 1) / threads;
    auto slice = [&](unsigned t, auto body) {
        const std::size_t lo = std::min(n, std::size_t(t) * chunk);
        const std::size_t hi = std::min(n, lo + chunk);
        body(lo, hi);
    };
    // First touch on the thread that will stream the slice (NUMA placement).
    run_on_threads(threads, [&](unsigned t) {
        slice(t, [&](std::size_t lo, std::size_t hi) {
            for (std::size_t i = lo; i < hi; ++i) { a[i] = 0.0; b[i] = 1.0; c[i] = 2.0; }
        });
    });
    const double s = 0.5;
    triad_best = read_best = 0.0;
    std::vector<double> partial(threads, 0.0);
    for (int rep = 0; rep < 6; ++rep) {
        const double rsec = run_on_threads(threads, [&](unsigned t) {
            slice(t, [&](std::size_t lo, std::size_t hi) {
                partial[t] = stream_read(a.get(), b.get(), c.get(), lo, hi);
            });
        });
        if (rep > 0) read_best = std::max(read_best, 3.0 * sizeof(double) * double(n) / rsec * 1e-9);
        const double sec = run_on_threads(threads, [&](unsigned t) {
            slice(t, [&](std::size_t lo, std::size_t hi) {
                double* __restrict pa = a.get();
                const double* __restrict pb = b.get();
                const double* __restrict pc = c.get();
                for (std::size_t i = lo; i < hi; ++i) pa[i] = pb[i] + s * pc[i];
            });
        });
        if (rep > 0) triad_best = std::max(triad_best, 3.0 * sizeof(double) * double(n) / sec * 1e-9);
        // c <- 0.5 c + b converges (to 2), so c stays bounded and normal.
        const double usec = run_on_threads(threads, [&](unsigned t) {
            slice(t, [&](std::size_t lo, std::size_t hi) {
                double* __restrict pc = c.get();
                const double* __restrict pb = b.get();
                for (std::size_t i = lo; i < hi; ++i) pc[i] = pc[i] * s + pb[i];
            });
        });
        if (rep > 0) out.update_gbytes_per_s = std::max(out.update_gbytes_per_s,
                                                        3.0 * sizeof(double) * double(n) / usec * 1e-9);
    }
    volatile double sink = a[n / 2] + partial[0];
    (void)sink;
}

} // namespace detail

inline probe_results run_probes(unsigned threads) {
    probe_results p;
    p.threads = threads == 0 ? 1 : threads;
    p.build_fma_peak_f64_gflops = detail::fma_peak_gflops<double>(p.threads);
    p.build_fma_peak_f32_gflops = detail::fma_peak_gflops<float>(p.threads);
    const hw::isa_kernels k = hw::select();
    p.hw_isa = k.isa;
    if (k.f64) p.hw_fma_peak_f64_gflops = detail::peak_gflops(p.threads, k.f64_flops_per_iter, k.f64);
    if (k.f32) p.hw_fma_peak_f32_gflops = detail::peak_gflops(p.threads, k.f32_flops_per_iter, k.f32);
    p.fma_peak_f64_gflops = std::max(p.build_fma_peak_f64_gflops, p.hw_fma_peak_f64_gflops);
    p.fma_peak_f32_gflops = std::max(p.build_fma_peak_f32_gflops, p.hw_fma_peak_f32_gflops);
    detail::bandwidth(p.threads, p);
    return p;
}

/// Attainable GFLOP/s for an operation of the given intensity (flops/byte):
/// min(compute peak, bandwidth x intensity) -- the roofline.
inline double roofline_gflops(double peak_gflops, double bw_gbytes, double flops, double bytes) {
    if (bytes <= 0.0) return peak_gflops;
    return std::min(peak_gflops, bw_gbytes * (flops / bytes));
}

} // namespace mtl::bench::platform
