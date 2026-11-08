#pragma once
// mtl5_bench -- the platform benchmark's registry (#721).
//
// One ENTRY per (operation, element type). An entry does not time anything
// itself; it builds a PROBLEM for a size class, and the runner times it. That
// split is what keeps every entry measured the same way on every machine: the
// sampling, the clock, the time budget and the reporting live in one place
// (run.hpp), and an entry only says what to compute and how to check it.
//
// A problem has four parts:
//   setup()  restores the inputs. Runs before EVERY sample and is not timed, so
//            an in-place operation (axpy, trsm, ...) never drifts toward
//            overflow or denormals across repetitions -- either would change
//            what is being timed, and the denormal case by orders of magnitude.
//   run()    the operation, through the public mtl:: API only. The build
//            decides the backend (generic, native-fast, BLAS), exactly as it
//            does for an application; see benchmarks/README.md.
//   check()  runs once, after the last sample, on the state run() left. A
//            fast wrong answer is a FAILURE, never a number.
//   flops / bytes   the work model, so the runner can place the result on the
//            machine's roofline (probes.hpp). bytes is the compulsory traffic
//            (each operand read once, each output written once), the STREAM
//            convention, so a memory-bound op is judged against bandwidth.

#include <cstddef>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace mtl::bench::platform {

/// Problem sizes. `small` is a smoke run (about a minute for the whole
/// catalogue, and what CI runs); `medium` is the standard characterization;
/// `large` is for machines with the memory and time to spare.
enum class size_class { small, medium, large };

inline const char* to_string(size_class c) {
    switch (c) {
    case size_class::small:  return "small";
    case size_class::medium: return "medium";
    case size_class::large:  return "large";
    }
    return "?";
}

/// Result of a correctness check: `ok`, and the measured error against the
/// tolerance it had to meet, both reported so a near-miss is visible.
struct check_result {
    bool        ok = false;
    double      error = 0.0;
    double      tolerance = 0.0;
    std::string what;   // the metric, e.g. "max rel err vs reference"
};

struct problem {
    std::string shape;   // human-readable size, e.g. "n=1024" or "512x512x512"
    double      flops = 0.0;
    double      bytes = 0.0;
    std::function<void()>         setup;
    std::function<void()>         run;
    std::function<check_result()> check;
    /// False for work the probes have no ceiling for (integer arithmetic: the
    /// FMA probes are floating point). The rate is still reported -- as GOP/s --
    /// but not placed on a roofline.
    bool roofline = true;
};

struct entry {
    std::string area;    // "blas1", "blas2", "blas3", ... (the card's sections)
    std::string op;      // "dot", "gemm", ...
    std::string dtype;   // "f64", "f32", ...
    std::function<problem(size_class)> make;
};

class registry {
public:
    void add(std::string area, std::string op, std::string dtype,
             std::function<problem(size_class)> make) {
        entries_.push_back({std::move(area), std::move(op), std::move(dtype), std::move(make)});
    }
    const std::vector<entry>& entries() const { return entries_; }

private:
    std::vector<entry> entries_;
};

} // namespace mtl::bench::platform
