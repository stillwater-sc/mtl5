#pragma once
// MTL5 -- N-dimensional shape and stride utilities for ndarray
#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <numeric>
#include <stdexcept>

#include <mtl/tag/orientation.hpp>

namespace mtl::array {

/// Layout order for multi-dimensional arrays
using c_order = tag::row_major;     // last index varies fastest (NumPy default)
using f_order = tag::col_major;     // first index varies fastest (Fortran/MATLAB)

// ── Shape ──────────────────────────────────────────────────────────

/// Static-rank shape: an array of N extents.
template <std::size_t N>
class shape {
    std::array<std::size_t, N> extents_{};

public:
    static constexpr std::size_t rank = N;

    constexpr shape() = default;

    constexpr shape(std::array<std::size_t, N> ext) : extents_(ext) {}

    constexpr shape(std::initializer_list<std::size_t> il) {
        assert(il.size() == N);
        std::copy_n(il.begin(), N, extents_.begin());
    }

    constexpr std::size_t  operator[](std::size_t i) const { return extents_[i]; }
    constexpr std::size_t& operator[](std::size_t i)       { return extents_[i]; }

    constexpr const std::array<std::size_t, N>& extents() const { return extents_; }

    constexpr std::size_t total_size() const {
        std::size_t s = 1;
        for (std::size_t i = 0; i < N; ++i) s *= extents_[i];
        return s;
    }

    constexpr bool operator==(const shape&) const = default;
};

// ── Strides ────────────────────────────────────────────────────────

/// Compute C-order (row-major) strides: last dimension has stride 1.
template <std::size_t N>
constexpr std::array<std::size_t, N> c_order_strides(const shape<N>& sh) {
    std::array<std::size_t, N> strides{};
    if constexpr (N > 0) {
        strides[N - 1] = 1;
        for (std::size_t i = N - 1; i > 0; --i) {
            strides[i - 1] = strides[i] * sh[i];
        }
    }
    return strides;
}

/// Compute F-order (column-major) strides: first dimension has stride 1.
template <std::size_t N>
constexpr std::array<std::size_t, N> f_order_strides(const shape<N>& sh) {
    std::array<std::size_t, N> strides{};
    if constexpr (N > 0) {
        strides[0] = 1;
        for (std::size_t i = 1; i < N; ++i) {
            strides[i] = strides[i - 1] * sh[i - 1];
        }
    }
    return strides;
}

/// Compute strides for a given layout order.
template <typename Order, std::size_t N>
constexpr std::array<std::size_t, N> compute_strides(const shape<N>& sh) {
    if constexpr (std::is_same_v<Order, f_order>)
        return f_order_strides(sh);
    else
        return c_order_strides(sh);
}

// ── Offset computation ─────────────────────────────────────────────

/// Compute flat offset from multi-index and strides.
template <std::size_t N>
constexpr std::size_t compute_offset(const std::array<std::size_t, N>& indices,
                                     const std::array<std::size_t, N>& strides) {
    std::size_t offset = 0;
    for (std::size_t i = 0; i < N; ++i) {
        offset += indices[i] * strides[i];
    }
    return offset;
}

/// Variadic offset computation — converts (i, j, k, ...) to flat index.
template <std::size_t N, typename... Indices>
constexpr std::size_t offset_from(const std::array<std::size_t, N>& strides,
                                  Indices... indices) {
    static_assert(sizeof...(Indices) == N, "Number of indices must match rank");
    const std::array<std::size_t, N> idx{static_cast<std::size_t>(indices)...};
    return compute_offset(idx, strides);
}

// ── Broadcasting ───────────────────────────────────────────────────

/// Compute the broadcast shape of two shapes.
/// Follows NumPy rules: trailing dimensions aligned, size-1 axes stretch.
/// Returns false if shapes are incompatible.
template <std::size_t N>
constexpr bool broadcast_shape(const shape<N>& a, const shape<N>& b, shape<N>& result) {
    for (std::size_t i = 0; i < N; ++i) {
        if (a[i] == b[i]) {
            result[i] = a[i];
        } else if (a[i] == 1) {
            result[i] = b[i];
        } else if (b[i] == 1) {
            result[i] = a[i];
        } else {
            return false;  // incompatible
        }
    }
    return true;
}

/// Compute broadcast strides: dimension with extent 1 gets stride 0 (repeat).
template <std::size_t N>
constexpr std::array<std::size_t, N> broadcast_strides(const shape<N>& original,
                                                       const std::array<std::size_t, N>& strides,
                                                       const shape<N>& target) {
    std::array<std::size_t, N> result{};
    for (std::size_t i = 0; i < N; ++i) {
        result[i] = (original[i] == target[i]) ? strides[i] : 0;
    }
    return result;
}

// ── Contiguity check ───────────────────────────────────────────────

/// Check if strides correspond to a contiguous C-order layout.
template <std::size_t N>
constexpr bool is_contiguous_c(const shape<N>& sh, const std::array<std::size_t, N>& strides) {
    return strides == c_order_strides(sh);
}

/// Check if strides correspond to a contiguous F-order layout.
template <std::size_t N>
constexpr bool is_contiguous_f(const shape<N>& sh, const std::array<std::size_t, N>& strides) {
    return strides == f_order_strides(sh);
}

/// Check if the array is contiguous in any order.
template <std::size_t N>
constexpr bool is_contiguous(const shape<N>& sh, const std::array<std::size_t, N>& strides) {
    return is_contiguous_c(sh, strides) || is_contiguous_f(sh, strides);
}

} // namespace mtl::array
