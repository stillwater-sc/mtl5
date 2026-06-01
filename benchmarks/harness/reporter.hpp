#pragma once
// MTL5 Benchmark Harness -- Output formatting (console table + CSV)
#include <benchmarks/harness/timer.hpp>
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace mtl::bench {

/// Collects timing results and prints comparison tables
class reporter {
public:
    void add(const timing& t) { results_.push_back(t); }

    /// Print a formatted results table to stdout. One binary == one backend,
    /// so there is no in-run baseline; cross-backend speedups are computed at
    /// analysis time across the per-backend CSVs (see plot_results.py).
    void print_table() const {
        if (results_.empty()) return;

        std::printf("\n%-20s %-10s %8s %14s %14s %10s\n",
                    "Operation", "Backend", "Size", "Median(us)", "Min(us)", "GFLOP/s");
        std::printf("%s\n", std::string(80, '-').c_str());

        std::string prev_op;
        for (const auto& r : results_) {
            if (!prev_op.empty() && r.operation != prev_op) std::printf("\n");
            prev_op = r.operation;

            double median_us = r.median_ns / 1000.0;
            double min_us    = r.min_ns / 1000.0;
            char gflops_str[16] = "      --";
            if (r.gflops > 0.0)
                std::snprintf(gflops_str, sizeof(gflops_str), "%8.2f", r.gflops);

            std::printf("%-20s %-10s %8zu %14.2f %14.2f %10s\n",
                        r.operation.c_str(), r.backend.c_str(),
                        r.size, median_us, min_us, gflops_str);
        }
        std::printf("\n");
    }

    /// Write results to CSV file
    void write_csv(const std::filesystem::path& path) const {
        std::ofstream out(path);
        out << "operation,backend,size,median_ns,min_ns,max_ns,mean_ns,stddev_ns,gflops,iterations\n";
        for (const auto& r : results_) {
            out << r.operation << ','
                << r.backend << ','
                << r.size << ','
                << std::fixed << std::setprecision(1)
                << r.median_ns << ','
                << r.min_ns << ','
                << r.max_ns << ','
                << r.mean_ns << ','
                << r.stddev_ns << ','
                << std::setprecision(4) << r.gflops << ','
                << r.iterations << '\n';
        }
        std::cout << "Results written to: " << path << '\n';
    }

    const std::vector<timing>& results() const { return results_; }
    void clear() { results_.clear(); }

private:
    std::vector<timing> results_;
};

} // namespace mtl::bench
