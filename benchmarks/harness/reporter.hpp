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

    /// Print a formatted comparison table to stdout
    void print_table() const {
        if (results_.empty()) return;

        // Header
        std::printf("\n%-20s %-10s %8s %14s %14s %14s %10s\n",
                    "Operation", "Backend", "Size", "Median(us)", "Min(us)", "Speedup", "GFLOP/s");
        std::printf("%s\n", std::string(92, '-').c_str());

        // Group by (operation, size) to compute speedups relative to native
        std::size_t i = 0;
        while (i < results_.size()) {
            // Find all results for this (operation, size) group
            auto op   = results_[i].operation;
            auto size = results_[i].size;
            std::size_t group_start = i;
            while (i < results_.size() &&
                   results_[i].operation == op && results_[i].size == size) {
                ++i;
            }

            // Find native baseline in this group
            double baseline_ns = 0.0;
            for (std::size_t j = group_start; j < i; ++j) {
                if (results_[j].backend == "native") {
                    baseline_ns = results_[j].median_ns;
                    break;
                }
            }

            // Print each row in the group
            for (std::size_t j = group_start; j < i; ++j) {
                const auto& r = results_[j];
                double speedup = (baseline_ns > 0.0 && r.median_ns > 0.0)
                    ? baseline_ns / r.median_ns : 0.0;
                double median_us = r.median_ns / 1000.0;
                double min_us    = r.min_ns / 1000.0;

                char gflops_str[16] = "  --";
                if (r.gflops > 0.0)
                    std::snprintf(gflops_str, sizeof(gflops_str), "%8.2f", r.gflops);

                char speedup_str[16];
                if (r.backend == "native")
                    std::snprintf(speedup_str, sizeof(speedup_str), "  (base)");
                else
                    std::snprintf(speedup_str, sizeof(speedup_str), "%8.2fx", speedup);

                std::printf("%-20s %-10s %8zu %14.2f %14.2f %14s %10s\n",
                            r.operation.c_str(), r.backend.c_str(),
                            r.size, median_us, min_us,
                            speedup_str, gflops_str);
            }
            // Separator between groups
            if (i < results_.size())
                std::printf("\n");
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
