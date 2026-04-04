#pragma once
// MTL5 -- Slicing helpers for ndarray
//
// Provides slice descriptors that can be used with ndarray::slice():
//   all       — select entire dimension
//   range     — select a contiguous range with optional step
//   integer   — select a single index (reduces rank by 1)

#include <array>
#include <cstddef>
#include <tuple>
#include <type_traits>

#include <mtl/array/ndarray.hpp>

namespace mtl::array {

// ── Slice descriptors ──────────────────────────────────────────────

/// Select all elements along a dimension.
struct all_t {};
inline constexpr all_t all{};

/// Select a contiguous range [start, stop) with optional step.
struct range {
    std::size_t start;
    std::size_t stop;
    std::size_t step;

    constexpr range(std::size_t s, std::size_t e, std::size_t st = 1)
        : start(s), stop(e), step(st) {}

    constexpr std::size_t extent() const {
        return (stop > start) ? (stop - start + step - 1) / step : 0;
    }
};

// ── Slice implementation ───────────────────────────────────────────

namespace detail_slice {

// Count how many integer (non-slice) arguments there are → rank reduction
template <typename T>
struct is_index_arg : std::false_type {};

template <>
struct is_index_arg<all_t> : std::false_type {};

template <>
struct is_index_arg<range> : std::false_type {};

// Anything convertible to size_t that isn't all_t or range is an integer index
template <typename T>
concept SliceIndex = std::convertible_to<T, std::size_t>
    && !std::is_same_v<std::remove_cvref_t<T>, all_t>
    && !std::is_same_v<std::remove_cvref_t<T>, range>;

// Count non-integer slice args (these become dimensions in the result)
template <typename... Args>
constexpr std::size_t count_kept_dims() {
    return (... + (!SliceIndex<Args> ? std::size_t{1} : std::size_t{0}));
}

} // namespace detail_slice

/// Slice an ndarray, returning a view with potentially reduced rank.
///
/// Each argument is one of:
///   - all       → keep the full dimension
///   - range     → select a sub-range (adjusts shape + offset)
///   - integer   → fix an index (reduces rank by 1)
///
/// Returns an ndarray view with rank = N - (number of integer args).
template <typename Value, std::size_t N, typename Order, typename... Args>
    requires (sizeof...(Args) == N)
auto slice(ndarray<Value, N, Order>& arr, Args... args) {
    constexpr std::size_t M = detail_slice::count_kept_dims<Args...>();

    // Compute the new shape, strides, and base offset
    shape<M>                   new_shape;
    std::array<std::size_t, M> new_strides;
    std::size_t base_offset = 0;

    auto process = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        std::size_t out_dim = 0;
        auto process_one = [&](auto arg, std::size_t dim) {
            using Arg = decltype(arg);
            if constexpr (std::is_same_v<std::remove_cvref_t<Arg>, all_t>) {
                new_shape[out_dim]   = arr.extent(dim);
                new_strides[out_dim] = arr.get_strides()[dim];
                ++out_dim;
            } else if constexpr (std::is_same_v<std::remove_cvref_t<Arg>, range>) {
                new_shape[out_dim]   = arg.extent();
                new_strides[out_dim] = arr.get_strides()[dim] * arg.step;
                base_offset += arg.start * arr.get_strides()[dim];
                ++out_dim;
            } else {
                // Integer index: fix this dimension
                base_offset += static_cast<std::size_t>(arg) * arr.get_strides()[dim];
            }
        };
        (process_one(args, Is), ...);
    };
    process(std::make_index_sequence<N>{});

    return ndarray<Value, M, Order>(
        arr.data() + base_offset, new_shape, new_strides);
}

/// Const overload: returns a view over const data.
template <typename Value, std::size_t N, typename Order, typename... Args>
    requires (sizeof...(Args) == N)
auto slice(const ndarray<Value, N, Order>& arr, Args... args) {
    constexpr std::size_t M = detail_slice::count_kept_dims<Args...>();

    shape<M>                   new_shape;
    std::array<std::size_t, M> new_strides;
    std::size_t base_offset = 0;

    auto process = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        std::size_t out_dim = 0;
        auto process_one = [&](auto arg, std::size_t dim) {
            using Arg = decltype(arg);
            if constexpr (std::is_same_v<std::remove_cvref_t<Arg>, all_t>) {
                new_shape[out_dim]   = arr.extent(dim);
                new_strides[out_dim] = arr.get_strides()[dim];
                ++out_dim;
            } else if constexpr (std::is_same_v<std::remove_cvref_t<Arg>, range>) {
                new_shape[out_dim]   = arg.extent();
                new_strides[out_dim] = arr.get_strides()[dim] * arg.step;
                base_offset += arg.start * arr.get_strides()[dim];
                ++out_dim;
            } else {
                base_offset += static_cast<std::size_t>(arg) * arr.get_strides()[dim];
            }
        };
        (process_one(args, Is), ...);
    };
    process(std::make_index_sequence<N>{});

    return ndarray<const Value, M, Order>(
        arr.data() + base_offset, new_shape, new_strides);
}

} // namespace mtl::array
