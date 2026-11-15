#pragma once
// MTL5 -- Slicing helpers for ndarray
//
// Provides slice descriptors that can be used with ndarray::slice():
//   all       - select entire dimension
//   range     - select a contiguous range with optional step
//   integer   - select a single index (reduces rank by 1)

#include <algorithm>
#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>

#include <mtl/array/ndarray.hpp>
#include <mtl/detail/exceptions.hpp>

namespace mtl::array {

// -- Slice descriptors ----------------------------------------------

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

// -- Slice implementation -------------------------------------------

namespace detail_slice {

// Count how many integer (non-slice) arguments there are -> rank reduction
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
    return (std::size_t{0} + ... + (!SliceIndex<Args> ? std::size_t{1} : std::size_t{0}));
}

/// The shape, strides and base offset of a slice of an array with the given
/// extents and strides. Every argument is checked, in every build (#856):
///   - a range is clamped to the extent, as NumPy clamps it: range(2, 10) of
///     3 rows is the one row 2, and a range that starts past the end is empty;
///   - step 0 throws std::invalid_argument (it divided by zero);
///   - an integer index outside [0, extent) throws std::out_of_range (it
///     addressed memory past the array).
template <std::size_t M, std::size_t N, typename... Args>
void slice_layout(const shape<N>& ext, const std::array<std::size_t, N>& strides,
                  shape<M>& new_shape, std::array<std::size_t, M>& new_strides,
                  std::size_t& base_offset, Args... args) {
    base_offset = 0;
    std::size_t out_dim = 0;
    std::size_t dim = 0;
    auto process_one = [&](auto arg) {
        using Arg = std::remove_cvref_t<decltype(arg)>;
        const std::size_t n = ext[dim];
        if constexpr (std::is_same_v<Arg, all_t>) {
            new_shape[out_dim]   = n;
            new_strides[out_dim] = strides[dim];
            ++out_dim;
        } else if constexpr (std::is_same_v<Arg, range>) {
            if (arg.step == 0)
                ::mtl::detail::throw_or_abort(std::invalid_argument("slice: range step 0"));
            const range r(std::min(arg.start, n), std::min(arg.stop, n), arg.step);
            new_shape[out_dim]   = r.extent();
            new_strides[out_dim] = strides[dim] * r.step;
            if (r.extent() != 0) base_offset += r.start * strides[dim];
            ++out_dim;
        } else {
            // Integer index: fix this dimension. A negative int converts to a
            // huge size_t and is caught here too.
            const std::size_t i = static_cast<std::size_t>(arg);
            if (i >= n)
                ::mtl::detail::throw_or_abort(std::out_of_range(
                    "slice: index " + std::to_string(i) + " out of range for an extent of "
                    + std::to_string(n)));
            base_offset += i * strides[dim];
        }
        ++dim;
    };
    (process_one(args), ...);
    static_cast<void>(process_one);   // rank 0: no arguments
}

} // namespace detail_slice

/// Slice an ndarray, returning a view with potentially reduced rank.
///
/// Each argument is one of:
///   - all       -> keep the full dimension
///   - range     -> select a sub-range (adjusts shape + offset), clamped to
///                  the extent; step 0 throws
///   - integer   -> fix an index (reduces rank by 1); out of range throws
///
/// Returns an ndarray view with rank = N - (number of integer args). A rank-0
/// array takes no arguments and gives a view of itself.
template <typename Value, std::size_t N, typename Order, typename... Args>
    requires (sizeof...(Args) == N)
auto slice(ndarray<Value, N, Order>& arr, Args... args) {
    constexpr std::size_t M = detail_slice::count_kept_dims<Args...>();
    shape<M>                   new_shape;
    std::array<std::size_t, M> new_strides{};
    std::size_t base_offset = 0;
    detail_slice::slice_layout(arr.get_shape(), arr.get_strides(), new_shape, new_strides,
                               base_offset, args...);
    return ndarray<Value, M, Order>(arr.data() + base_offset, new_shape, new_strides);
}

/// Const overload: returns a view over const data.
template <typename Value, std::size_t N, typename Order, typename... Args>
    requires (sizeof...(Args) == N)
auto slice(const ndarray<Value, N, Order>& arr, Args... args) {
    constexpr std::size_t M = detail_slice::count_kept_dims<Args...>();
    shape<M>                   new_shape;
    std::array<std::size_t, M> new_strides{};
    std::size_t base_offset = 0;
    detail_slice::slice_layout(arr.get_shape(), arr.get_strides(), new_shape, new_strides,
                               base_offset, args...);
    return ndarray<const Value, M, Order>(arr.data() + base_offset, new_shape, new_strides);
}

} // namespace mtl::array
