#pragma once
// MTL5 -- Reduction operations for ndarray
//
// Provides sum, prod, min, max, mean as free functions.
// Full reductions return a scalar; axis reductions return an ndarray
// of reduced rank.
//
// They take any ndarray, a read-only one (ndarray<const V>, as as_ndarray of a
// const vector gives) included, and return the unqualified element type. The
// identities are math::zero<V>() and math::one<V>(), so custom number types
// work. min, max and mean have no identity: on an empty array they throw
// std::invalid_argument (#856; min and max returned numeric_limits' max and
// lowest, which is 0 for a type without them, and mean divided by zero).

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <mtl/array/ndarray.hpp>
#include <mtl/detail/exceptions.hpp>
#include <mtl/math/identity.hpp>

namespace mtl::array {

namespace detail_ops {

[[noreturn]] inline void empty_reduction(const char* what) {
    ::mtl::detail::throw_or_abort(
        std::invalid_argument(std::string(what) + ": an empty array has no " + what));
}

} // namespace detail_ops

// -- Full reductions (scalar result) --------------------------------

/// Sum all elements; zero for an empty array.
template <typename V, std::size_t N, typename O>
std::remove_cv_t<V> sum(const ndarray<V, N, O>& a) {
    using R = std::remove_cv_t<V>;
    R result = math::zero<R>();
    a.for_each_element([&](const V& v) { result += v; });
    return result;
}

/// Product of all elements; one for an empty array.
template <typename V, std::size_t N, typename O>
std::remove_cv_t<V> prod(const ndarray<V, N, O>& a) {
    using R = std::remove_cv_t<V>;
    R result = math::one<R>();
    a.for_each_element([&](const V& v) { result *= v; });
    return result;
}

/// Minimum element, seeded from the first one. Throws on an empty array.
template <typename V, std::size_t N, typename O>
    requires std::totally_ordered<std::remove_cv_t<V>>
std::remove_cv_t<V> min(const ndarray<V, N, O>& a) {
    using R = std::remove_cv_t<V>;
    if (a.empty()) detail_ops::empty_reduction("min");
    const R* best = nullptr;
    a.for_each_element([&](const V& v) { if (!best || v < *best) best = &v; });
    return *best;
}

/// Maximum element, seeded from the first one. Throws on an empty array.
template <typename V, std::size_t N, typename O>
    requires std::totally_ordered<std::remove_cv_t<V>>
std::remove_cv_t<V> max(const ndarray<V, N, O>& a) {
    using R = std::remove_cv_t<V>;
    if (a.empty()) detail_ops::empty_reduction("max");
    const R* best = nullptr;
    a.for_each_element([&](const V& v) { if (!best || v > *best) best = &v; });
    return *best;
}

/// Mean of all elements. Throws on an empty array.
template <typename V, std::size_t N, typename O>
std::remove_cv_t<V> mean(const ndarray<V, N, O>& a) {
    using R = std::remove_cv_t<V>;
    if (a.empty()) detail_ops::empty_reduction("mean");
    return sum(a) / static_cast<R>(a.size());
}

// -- Axis reductions (ndarray result) -------------------------------

/// Sum along a specified axis, producing an ndarray of rank N-1. An axis of
/// extent 0 gives zeros. Throws std::out_of_range for an axis >= N.
template <typename V, std::size_t N, typename O>
    requires (N > 1)
ndarray<std::remove_cv_t<V>, N - 1, O> sum_axis(const ndarray<V, N, O>& a, std::size_t axis) {
    using R = std::remove_cv_t<V>;
    if (axis >= N)
        ::mtl::detail::throw_or_abort(std::out_of_range(
            "sum_axis: axis " + std::to_string(axis) + " of a rank-" + std::to_string(N) + " array"));

    // Build output shape: remove the axis dimension
    shape<N - 1> out_shape;
    std::size_t j = 0;
    for (std::size_t i = 0; i < N; ++i) {
        if (i != axis) out_shape[j++] = a.extent(i);
    }

    ndarray<R, N - 1, O> result(out_shape, math::zero<R>());

    // One walk over the input, with the output's strides lifted to rank N and
    // stride 0 along the axis: every input element lands in its output
    // element, with no index array rebuilt per element (#856). Each output
    // element still receives its terms in increasing axis index.
    std::array<std::size_t, N> out_strides{};
    j = 0;
    for (std::size_t i = 0; i < N; ++i)
        out_strides[i] = (i == axis) ? 0 : result.get_strides()[j++];
    const V* src = a.data();
    R* dst = result.data();
    detail_nd::walk_offsets<O>(a.get_shape(),
                               std::array<std::array<std::size_t, N>, 2>{a.get_strides(), out_strides},
                               [&](const auto& o) { dst[o[1]] += src[o[0]]; });
    return result;
}

/// Mean along a specified axis. Throws std::invalid_argument when the axis
/// has extent 0 (there is nothing to average).
template <typename V, std::size_t N, typename O>
    requires (N > 1)
ndarray<std::remove_cv_t<V>, N - 1, O> mean_axis(const ndarray<V, N, O>& a, std::size_t axis) {
    using R = std::remove_cv_t<V>;
    auto result = sum_axis(a, axis);
    if (a.extent(axis) == 0) detail_ops::empty_reduction("mean_axis");
    const R divisor = static_cast<R>(a.extent(axis));
    result.for_each_element([&](R& v) { v /= divisor; });
    return result;
}

} // namespace mtl::array
