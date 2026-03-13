#pragma once
// MTL5 Benchmark Harness -- High-resolution timing and statistics
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <string>
#include <string_view>
#include <vector>

namespace mtl::bench {

/// Timing result for a single benchmark configuration
struct timing {
    std::string operation;
    std::string backend;
    std::size_t size       = 0;
    double      median_ns  = 0.0;
    double      min_ns     = 0.0;
    double      max_ns     = 0.0;
    double      mean_ns    = 0.0;
    double      stddev_ns  = 0.0;
    double      gflops     = 0.0;   // if flop count is known
    std::size_t iterations = 0;
};

/// Measure the execution time of a callable, returning nanoseconds.
/// Runs warmup iterations, then timed iterations, returns sorted samples.
template <typename Fn>
timing measure(Fn&& fn, std::string_view op_name, std::string_view backend_name,
               std::size_t size, double flop_count = 0.0,
               std::size_t warmup = 3, std::size_t iterations = 10) {
    using clock = std::chrono::high_resolution_clock;

    // Warmup
    for (std::size_t i = 0; i < warmup; ++i) {
        fn();
    }

    // Timed runs
    std::vector<double> samples(iterations);
    for (std::size_t i = 0; i < iterations; ++i) {
        auto t0 = clock::now();
        fn();
        auto t1 = clock::now();
        samples[i] = std::chrono::duration<double, std::nano>(t1 - t0).count();
    }

    std::sort(samples.begin(), samples.end());

    double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
    double mean = sum / static_cast<double>(iterations);
    double sq_sum = 0.0;
    for (auto s : samples) {
        double d = s - mean;
        sq_sum += d * d;
    }

    timing result;
    result.operation  = std::string(op_name);
    result.backend    = std::string(backend_name);
    result.size       = size;
    result.min_ns     = samples.front();
    result.max_ns     = samples.back();
    result.median_ns  = samples[iterations / 2];
    result.mean_ns    = mean;
    result.stddev_ns  = std::sqrt(sq_sum / static_cast<double>(iterations));
    result.iterations = iterations;

    if (flop_count > 0.0) {
        result.gflops = flop_count / (result.median_ns); // GFLOP/s = flops / ns
    }

    return result;
}

/// Auto-calibrate iteration count: run until total time exceeds min_total_ns.
/// Returns the calibrated iteration count (minimum 3).
template <typename Fn>
std::size_t calibrate(Fn&& fn, double min_total_ns = 1e9 /* 1 second */) {
    using clock = std::chrono::high_resolution_clock;

    // Single probe run
    auto t0 = clock::now();
    fn();
    auto t1 = clock::now();
    double single_ns = std::chrono::duration<double, std::nano>(t1 - t0).count();

    if (single_ns <= 0.0) single_ns = 1.0;
    auto iters = static_cast<std::size_t>(min_total_ns / single_ns);
    return std::max<std::size_t>(iters, 3);
}

} // namespace mtl::bench
