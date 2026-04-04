#pragma once
// MTL5 -- Reduction operations for ndarray
//
// Provides sum, prod, min, max, mean as free functions.
// Full reductions return a scalar; axis reductions return an ndarray
// of reduced rank.

#include <algorithm>
#include <cstddef>
#include <limits>
#include <type_traits>

#include <mtl/array/ndarray.hpp>
#include <mtl/math/identity.hpp>

namespace mtl::array {

// ── Full reductions (scalar result) ────────────────────────────────

/// Sum all elements.
template <typename V, std::size_t N, typename O>
V sum(const ndarray<V, N, O>& a) {
    V result = V{0};
    a.for_each_element([&](const V& v) { result += v; });
    return result;
}

/// Product of all elements.
template <typename V, std::size_t N, typename O>
V prod(const ndarray<V, N, O>& a) {
    V result = V{1};
    a.for_each_element([&](const V& v) { result *= v; });
    return result;
}

/// Minimum element.
template <typename V, std::size_t N, typename O>
    requires std::totally_ordered<V>
V min(const ndarray<V, N, O>& a) {
    V result = std::numeric_limits<V>::max();
    a.for_each_element([&](const V& v) { if (v < result) result = v; });
    return result;
}

/// Maximum element.
template <typename V, std::size_t N, typename O>
    requires std::totally_ordered<V>
V max(const ndarray<V, N, O>& a) {
    V result = std::numeric_limits<V>::lowest();
    a.for_each_element([&](const V& v) { if (v > result) result = v; });
    return result;
}

/// Mean of all elements.
template <typename V, std::size_t N, typename O>
V mean(const ndarray<V, N, O>& a) {
    return sum(a) / static_cast<V>(a.size());
}

// ── Axis reductions (ndarray result) ───────────────────────────────

/// Sum along a specified axis, producing an ndarray of rank N-1.
template <typename V, std::size_t N, typename O>
    requires (N > 1)
ndarray<V, N - 1, O> sum_axis(const ndarray<V, N, O>& a, std::size_t axis) {
    assert(axis < N);

    // Build output shape: remove the axis dimension
    shape<N - 1> out_shape;
    std::size_t j = 0;
    for (std::size_t i = 0; i < N; ++i) {
        if (i != axis) out_shape[j++] = a.extent(i);
    }

    ndarray<V, N - 1, O> result(out_shape, V{0});

    // Iterate over all indices of the input
    a.iterate_indices([&](const std::array<std::size_t, N>& idx) {
        // Build the output index by skipping the axis
        std::array<std::size_t, N - 1> out_idx;
        std::size_t k = 0;
        for (std::size_t i = 0; i < N; ++i) {
            if (i != axis) out_idx[k++] = idx[i];
        }
        result[out_idx] += a[idx];
    });

    return result;
}

/// Mean along a specified axis.
template <typename V, std::size_t N, typename O>
    requires (N > 1)
ndarray<V, N - 1, O> mean_axis(const ndarray<V, N, O>& a, std::size_t axis) {
    auto result = sum_axis(a, axis);
    V divisor = static_cast<V>(a.extent(axis));
    result.for_each_element([&](V& v) { v /= divisor; });
    return result;
}

} // namespace mtl::array
