#pragma once
// MTL5 -- Base case tests for recursive algorithms
// Port from MTL4: boost/numeric/mtl/recursion/base_case_test.hpp
// Key changes: constexpr functors, no Boost dependencies

#include <algorithm>
#include <cstddef>

namespace mtl::recursion {

/// Stops recursion when min(rows, cols) <= threshold
struct min_dim_test {
    std::size_t threshold;

    constexpr explicit min_dim_test(std::size_t t) : threshold(t) {}

    template <typename Recursator>
    constexpr bool operator()(const Recursator& rec) const {
        return std::min(rec.num_rows(), rec.num_cols()) <= threshold;
    }
};

/// Stops recursion when max(rows, cols) <= threshold
struct max_dim_test {
    std::size_t threshold;

    constexpr explicit max_dim_test(std::size_t t) : threshold(t) {}

    template <typename Recursator>
    constexpr bool operator()(const Recursator& rec) const {
        return std::max(rec.num_rows(), rec.num_cols()) <= threshold;
    }
};

/// Compile-time threshold version of max_dim_test
template <std::size_t N>
struct max_dim_test_static {
    static constexpr std::size_t threshold = N;

    template <typename Recursator>
    constexpr bool operator()(const Recursator& rec) const {
        return std::max(rec.num_rows(), rec.num_cols()) <= N;
    }
};

} // namespace mtl::recursion
