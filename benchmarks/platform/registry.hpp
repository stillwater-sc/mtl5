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

/// An entry's definition version (#928), written into each result beside
/// area/op/dtype/shape. A trend compares results of one version only, so a
/// changed benchmark starts a new series instead of showing as a change in
/// MTL5.
///
/// BUMP IT whenever a run of the new code would not measure the same thing:
///   - the problem: its size, generator, matrix, sparsity pattern or right-hand
///     side, also when the shape string stays the same;
///   - the work model: flops or bytes;
///   - what the check compares (its metric or reference).
/// Do NOT bump it for a faster implementation of the same thing: that change is
/// what a trend is for. An entry starts at 1. The entries that changed since
/// #721 start at 2 (benchmarks/README.md lists them).
struct entry {
    std::string area;    // "blas1", "blas2", "blas3", ... (the card's sections)
    std::string op;      // "dot", "gemm", ...
    std::string dtype;   // "f64", "f32", ...
    std::function<problem(size_class)> make;
    int version = 1;
};

/// An entry this build cannot run (its library was not found, say), with the
/// reason. Reported on the card and in the JSON, so a missing reference row
/// reads as "not measured here" rather than as an oversight (#810).
struct skipped_entry {
    std::string area, op, dtype, reason;
};

class registry {
public:
    void add(std::string area, std::string op, std::string dtype,
             std::function<problem(size_class)> make, int version = 1) {
        entries_.push_back({std::move(area), std::move(op), std::move(dtype), std::move(make), version});
    }
    void skip(std::string area, std::string op, std::string dtype, std::string reason) {
        skipped_.push_back({std::move(area), std::move(op), std::move(dtype), std::move(reason)});
    }
    const std::vector<entry>& entries() const { return entries_; }
    const std::vector<skipped_entry>& skipped() const { return skipped_; }

private:
    std::vector<entry> entries_;
    std::vector<skipped_entry> skipped_;
};

} // namespace mtl::bench::platform
