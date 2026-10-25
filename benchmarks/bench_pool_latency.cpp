// MTL5 Benchmark -- dispatch and join latency of the two pool types (#636)
//
// Runs many near-empty regions on a pool of T threads and times each one:
//
//   dispatch  from the caller's last timestamp before run() to the latest
//             worker's first instruction in the task (the slowest wake-up)
//   join      from the latest tid's last instruction in the task to the
//             caller's first timestamp after run() returns
//   region    the whole run(), caller to caller
//
// reported as percentiles (p50, p90, p99, p99.9, max) over the regions, in
// nanoseconds. The arms are the pool types:
//
//   thread    mtl::parallel::thread_pool (mutex + condition variables)
//   spin      mtl::parallel::spin_pool with its default spin
//   sleep     mtl::parallel::spin_pool with spin 0: no spinning, every wait
//             sleeps on the futex at once
//   hold      mtl::parallel::spin_pool with --spin iterations (default 40000,
//             about 1.4 ms on a 12700K P-core): a spin longer than the gap, so
//             the workers are still spinning when the next region arrives --
//             the configuration for a fixed-rate loop on cores of its own
//
// --gap-us puts that much idle time between regions (a sleep), as a control
// loop at a fixed rate does; a gap longer than the spin means the spin_pool's
// workers have gone to sleep when the next region is published.
//
// Timestamps are std::chrono::steady_clock, read on different cores; on x86
// Linux that is the invariant TSC, synchronized across cores.
//
// Usage:
//   bench_pool_latency --arm thread|spin|sleep|hold [--threads 2,4,8] [--gap-us 0]
//       [--regions 20000] [--spin 40000] [--round N] [--csv f]
// The CSV is appended to, with a header when it is new. Run each arm in a
// process of its own; benchmarks/run_pool_latency.sh does.

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <mtl/parallel/spin_pool.hpp>
#include <mtl/parallel/thread_pool.hpp>
#include <mtl/util/system_info.hpp>
#include <mtl/build_info.hpp>

namespace {

using clk = std::chrono::steady_clock;

std::int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        clk::now().time_since_epoch()).count();
}

struct alignas(64) stamp {   // one per tid, on a cache line of its own
    std::int64_t start;
    std::int64_t end;
};

struct series {
    std::vector<std::int64_t> dispatch, join, region;
};

template <typename Pool>
series measure(Pool& pool, int regions, int warmup, int gap_us) {
    const unsigned T = pool.size();
    std::vector<stamp> st(T);
    series s;
    s.dispatch.reserve(regions);
    s.join.reserve(regions);
    s.region.reserve(regions);
    auto task = [&](unsigned tid) {
        st[tid].start = now_ns();
        st[tid].end = now_ns();
    };
    for (int r = -warmup; r < regions; ++r) {
        if (gap_us > 0) std::this_thread::sleep_for(std::chrono::microseconds(gap_us));
        const std::int64_t t0 = now_ns();
        pool.run(T, task);
        const std::int64_t t1 = now_ns();
        if (r < 0) continue;
        std::int64_t last_start = st[0].start, last_end = st[0].end;
        for (unsigned t = 1; t < T; ++t) {
            last_start = std::max(last_start, st[t].start);
            last_end = std::max(last_end, st[t].end);
        }
        s.dispatch.push_back(last_start - t0);
        s.join.push_back(t1 - last_end);
        s.region.push_back(t1 - t0);
    }
    return s;
}

struct pct { std::int64_t p50, p90, p99, p999, max; };

pct percentiles(std::vector<std::int64_t> v) {
    std::sort(v.begin(), v.end());
    auto at = [&](double q) {
        const std::size_t i = static_cast<std::size_t>(q * static_cast<double>(v.size() - 1) + 0.5);
        return v[std::min(i, v.size() - 1)];
    };
    return {at(0.50), at(0.90), at(0.99), at(0.999), v.back()};
}

std::vector<int> parse_list(const char* s) {
    std::vector<int> out;
    for (const char* p = s; *p;) {
        out.push_back(std::atoi(p));
        while (*p && *p != ',') ++p;
        if (*p == ',') ++p;
    }
    return out;
}

} // namespace

int main(int argc, char** argv) {
    std::string arm = "thread", csv;
    std::vector<int> threads{2, 4, 8};
    int gap_us = 0, regions = 20000, round = 1;
    unsigned hold_spin = 40000;
    for (int i = 1; i < argc; ++i) {
        auto next = [&] { return i + 1 < argc ? argv[++i] : ""; };
        if (!std::strcmp(argv[i], "--arm")) arm = next();
        else if (!std::strcmp(argv[i], "--threads")) threads = parse_list(next());
        else if (!std::strcmp(argv[i], "--gap-us")) gap_us = std::atoi(next());
        else if (!std::strcmp(argv[i], "--regions")) regions = std::atoi(next());
        else if (!std::strcmp(argv[i], "--round")) round = std::atoi(next());
        else if (!std::strcmp(argv[i], "--spin")) hold_spin = static_cast<unsigned>(std::atol(next()));
        else if (!std::strcmp(argv[i], "--csv")) csv = next();
        else { std::fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
    }
    if (arm != "thread" && arm != "spin" && arm != "sleep" && arm != "hold") {
        std::fprintf(stderr, "--arm must be thread, spin, sleep or hold\n");
        return 2;
    }
    const int warmup = std::max(50, regions / 20);

    bool new_csv = false;
    if (!csv.empty()) { std::ifstream probe(csv); new_csv = !probe.good(); }
    std::FILE* out = csv.empty() ? nullptr : std::fopen(csv.c_str(), "a");
    if (out && new_csv)
        std::fputs("arm,threads,gap_us,round,spin,metric,regions,p50_ns,p90_ns,p99_ns,p999_ns,max_ns\n", out);

    for (int T : threads) {
        series s;
        unsigned spin = 0;
        if (arm == "thread") {
            mtl::parallel::thread_pool pool(static_cast<unsigned>(T));
            s = measure(pool, regions, warmup, gap_us);
        } else {
            spin = arm == "spin" ? mtl::parallel::spin_pool::default_spin
                 : arm == "hold" ? hold_spin : 0u;
            mtl::parallel::spin_pool pool(static_cast<unsigned>(T), spin);
            s = measure(pool, regions, warmup, gap_us);
        }
        const std::pair<const char*, std::vector<std::int64_t>*> metrics[] = {
            {"dispatch", &s.dispatch}, {"join", &s.join}, {"region", &s.region}};
        for (const auto& [name, v] : metrics) {
            const pct p = percentiles(*v);
            std::printf("  %-6s T=%d gap=%5dus %-8s p50 %7lld  p90 %7lld  p99 %8lld  p99.9 %8lld  max %9lld ns\n",
                        arm.c_str(), T, gap_us, name, (long long)p.p50, (long long)p.p90,
                        (long long)p.p99, (long long)p.p999, (long long)p.max);
            if (out)
                std::fprintf(out, "%s,%d,%d,%d,%u,%s,%d,%lld,%lld,%lld,%lld,%lld\n", arm.c_str(), T,
                             gap_us, round, spin, name, regions, (long long)p.p50, (long long)p.p90,
                             (long long)p.p99, (long long)p.p999, (long long)p.max);
        }
    }
    if (out) std::fclose(out);

    // Provenance (#442).
    if (!csv.empty()) {
        if (std::ofstream si{csv + ".sysinfo"}) {
            si << "label=pool-latency\n"
               << mtl::util::to_keyvals(mtl::util::identify())
               << "git_commit="  << mtl::build_git_commit  << "\n"
               << "git_dirty="   << mtl::build_git_dirty   << "\n"
               << "cxx_flags="   << mtl::build_cxx_flags   << "\n"
               << "cmake_build_type=" << mtl::build_cmake_type << "\n"
               << "default_spin=" << mtl::parallel::spin_pool::default_spin << "\n"
               << "regions=" << regions << "\n";
            for (const char* name : {"MTL5_NUM_THREADS", "MTL5_POOL_SPIN", "BENCH_PCPUS"})
                if (const char* v = std::getenv(name)) si << "env_" << name << '=' << v << "\n";
        }
    }
    return 0;
}
