#pragma once
// mtl5_bench -- the one place a problem is timed (#721).
//
// Sampling: each sample is setup() (untimed) then run() (timed). Samples are
// taken until the class's time budget is spent, with at least kMinSamples and
// at most kMaxSamples, and the MEDIAN is reported: robust to the occasional
// interrupt or frequency step, which a mean is not. Two untimed warm-up runs
// come first (page faults, lazy pool start, instruction cache).
//
// The budget is per entry, so the run time of the whole catalogue is bounded by
// (entries x budget) plus setup, whatever the machine's speed. A slow machine
// takes fewer samples of the same problem; it is never given a smaller one.

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <exception>
#include <limits>
#include <string>
#include <vector>

#include <benchmarks/platform/probes.hpp>
#include <benchmarks/platform/registry.hpp>

namespace mtl::bench::platform {

struct record {
    std::string area, op, dtype, shape;
    double median_s = 0.0;
    double min_s = 0.0;
    std::size_t samples = 0;
    double flops = 0.0, bytes = 0.0;
    double gflops = 0.0;          // flops / median
    double gbytes_per_s = 0.0;    // bytes / median
    double roofline_gflops = 0.0; // attainable at this intensity
    double roofline_pct = 0.0;    // gflops / roofline_gflops * 100
    bool   cache_resident = false; // bytes <= last-level cache: the DRAM roofline does not apply
    check_result check;
    std::string error;            // non-empty if the entry threw
    /// No work model (an ordering, say): timed and checked, but with no
    /// meaningful rate or roofline, so reports show "n/a" rather than 0%.
    bool timed_only() const { return flops <= 0.0 && bytes <= 0.0; }
    bool roofline = true;          // false: a rate, but no ceiling to compare it with
    /// No meaningful %roof: timed only, or no ceiling for this kind of work.
    bool no_roofline() const { return timed_only() || !roofline; }
    // Thread scaling against a --baseline run of the same entry (NaN if none):
    // speedup = baseline median / this median; efficiency = speedup / (T / T0).
    double speedup = std::numeric_limits<double>::quiet_NaN();
    double efficiency = std::numeric_limits<double>::quiet_NaN();
};

inline double budget_seconds(size_class c) {
    switch (c) {
    case size_class::small:  return 0.05;
    case size_class::medium: return 0.5;
    case size_class::large:  return 2.0;
    }
    return 0.5;
}

inline constexpr std::size_t kMinSamples = 3;
inline constexpr std::size_t kMaxSamples = 1000;

/// The largest cache level the machine reports (L3, else L2), in bytes.
inline std::size_t last_level_cache_bytes() {
    const auto& ci = util::cached_cache_info();
    return ci.l3_bytes ? ci.l3_bytes : ci.l2_bytes;
}

inline double peak_for(const probe_results& p, const std::string& dtype) {
    return dtype == "f32" ? p.fma_peak_f32_gflops : p.fma_peak_f64_gflops;
}

inline record run_entry(const entry& e, size_class cls, const probe_results& probes) {
    using clock = std::chrono::steady_clock;
    record r;
    r.area = e.area; r.op = e.op; r.dtype = e.dtype;
    try {
        problem p = e.make(cls);
        r.shape = p.shape; r.flops = p.flops; r.bytes = p.bytes; r.roofline = p.roofline;
        for (int w = 0; w < 2; ++w) { p.setup(); p.run(); }

        std::vector<double> samples;
        const double budget = budget_seconds(cls);
        double spent = 0.0;
        while (samples.size() < kMaxSamples &&
               (samples.size() < kMinSamples || spent < budget)) {
            p.setup();
            const auto t0 = clock::now();
            p.run();
            const double s = std::chrono::duration<double>(clock::now() - t0).count();
            samples.push_back(s);
            spent += s;
        }
        std::sort(samples.begin(), samples.end());
        r.samples  = samples.size();
        r.median_s = samples[samples.size() / 2];
        r.min_s    = samples.front();
        // check() reads what the last run() left; setup() came before it.
        r.check = p.check();
    } catch (const std::exception& ex) {
        r.error = ex.what();
        r.check.ok = false;
        r.check.what = "threw";
    }
    if (r.median_s > 0.0) {
        r.gflops       = r.flops / r.median_s * 1e-9;
        r.gbytes_per_s = r.bytes / r.median_s * 1e-9;
        r.roofline_gflops = roofline_gflops(peak_for(probes, r.dtype),
                                            probes.bandwidth_gbytes_per_s(), r.flops, r.bytes);
        if (r.roofline_gflops > 0.0) r.roofline_pct = r.gflops / r.roofline_gflops * 100.0;
    }
    // Only a BANDWIDTH-bound result is misjudged by fitting in cache; a
    // compute-bound one has the FMA peak as its ceiling wherever its data is.
    const bool bandwidth_bound = r.bytes > 0.0 &&
        probes.bandwidth_gbytes_per_s() * (r.flops / r.bytes) < peak_for(probes, r.dtype);
    r.cache_resident = r.roofline && bandwidth_bound && r.bytes <= double(last_level_cache_bytes());
    return r;
}

} // namespace mtl::bench::platform
