#pragma once
// MTL5 -- Interop between ndarray and existing vector/matrix types
//
// Provides:
//   - as_ndarray(vec)  → ndarray<T, 1> view (zero-copy)
//   - as_ndarray(mat)  → ndarray<T, 2> view (zero-copy, row-major only)
//   - as_vector(ndarray<T,1>) → dense_vector view (zero-copy)
//   - as_matrix(ndarray<T,2>) → dense2D view (zero-copy, contiguous only)
//   - Generic algorithms: transform, reduce, flatten

#include <array>
#include <cassert>
#include <cstddef>
#include <functional>
#include <stdexcept>

#include <mtl/array/ndarray.hpp>
#include <mtl/array/shape.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/mat/dense2D.hpp>

namespace mtl::array {

// ── Vector → ndarray view ──────────────────────────────────────────

/// Create a 1D ndarray view over a dense_vector (zero-copy).
template <typename Value, typename Params>
ndarray<Value, 1, c_order> as_ndarray(vec::dense_vector<Value, Params>& v) {
    return ndarray<Value, 1, c_order>(v.data(), shape<1>{v.size()});
}

template <typename Value, typename Params>
ndarray<const Value, 1, c_order> as_ndarray(const vec::dense_vector<Value, Params>& v) {
    return ndarray<const Value, 1, c_order>(
        const_cast<const Value*>(v.data()), shape<1>{v.size()});
}

// ── Matrix → ndarray view ──────────────────────────────────────────

/// Create a 2D ndarray view over a dense2D matrix (zero-copy).
/// Works for row-major matrices; column-major uses F-order strides.
template <typename Value, typename Params>
auto as_ndarray(mat::dense2D<Value, Params>& m) {
    using orientation = typename Params::orientation;
    shape<2> sh{m.num_rows(), m.num_cols()};

    if constexpr (std::is_same_v<orientation, tag::row_major>) {
        return ndarray<Value, 2, c_order>(m.data(), sh);
    } else {
        // Column-major: strides are {1, num_rows}
        std::array<std::size_t, 2> strides{1, m.num_rows()};
        return ndarray<Value, 2, c_order>(m.data(), sh, strides);
    }
}

template <typename Value, typename Params>
auto as_ndarray(const mat::dense2D<Value, Params>& m) {
    using orientation = typename Params::orientation;
    shape<2> sh{m.num_rows(), m.num_cols()};

    if constexpr (std::is_same_v<orientation, tag::row_major>) {
        return ndarray<const Value, 2, c_order>(
            const_cast<const Value*>(m.data()), sh);
    } else {
        std::array<std::size_t, 2> strides{1, m.num_rows()};
        return ndarray<const Value, 2, c_order>(
            const_cast<const Value*>(m.data()), sh, strides);
    }
}

// ── ndarray → vector view ──────────────────────────────────────────

/// Create a dense_vector view over a 1D ndarray (zero-copy).
/// Requires contiguous storage.
template <typename Value, typename Order>
vec::dense_vector<Value> as_vector(ndarray<Value, 1, Order>& a) {
    assert(a.is_contiguous() && "as_vector requires contiguous ndarray");
    return vec::dense_vector<Value>(a.size(), a.data());
}

template <typename Value, typename Order>
vec::dense_vector<const Value> as_vector(const ndarray<Value, 1, Order>& a) {
    assert(a.is_contiguous() && "as_vector requires contiguous ndarray");
    return vec::dense_vector<const Value>(a.size(), const_cast<const Value*>(a.data()));
}

// ── ndarray → matrix view ──────────────────────────────────────────

/// Create a dense2D view over a 2D ndarray (zero-copy).
/// Requires contiguous C-order (row-major) storage.
template <typename Value, typename Order>
    requires std::is_same_v<Order, c_order>
mat::dense2D<Value> as_matrix(ndarray<Value, 2, Order>& a) {
    assert(a.is_contiguous() && "as_matrix requires contiguous ndarray");
    return mat::dense2D<Value>(a.extent(0), a.extent(1), a.data());
}

template <typename Value, typename Order>
    requires std::is_same_v<Order, c_order>
mat::dense2D<const Value> as_matrix(const ndarray<Value, 2, Order>& a) {
    assert(a.is_contiguous() && "as_matrix requires contiguous ndarray");
    return mat::dense2D<const Value>(
        a.extent(0), a.extent(1), const_cast<const Value*>(a.data()));
}

// ── Generic algorithms ─────────────────────────────────────────────

/// Element-wise transform: apply f to every element of a, writing to out.
/// Works on any NdArray (ndarray, or adapted vector/matrix views).
template <typename V, std::size_t N, typename O, typename F>
ndarray<V, N, O> transform(const ndarray<V, N, O>& a, F&& f) {
    ndarray<V, N, O> result(a.get_shape());
    a.iterate_indices([&](const std::array<std::size_t, N>& idx) {
        result[idx] = f(a[idx]);
    });
    return result;
}

/// In-place transform: apply f to every element.
template <typename V, std::size_t N, typename O, typename F>
void transform_inplace(ndarray<V, N, O>& a, F&& f) {
    a.for_each_element([&](V& v) { v = f(v); });
}

/// Reduce all elements with a binary operation and initial value.
template <typename V, std::size_t N, typename O, typename T, typename BinOp>
T reduce(const ndarray<V, N, O>& a, T init, BinOp op) {
    a.for_each_element([&](const V& v) { init = op(init, v); });
    return init;
}

/// Flatten any ndarray to a contiguous 1D ndarray (always copies for safety).
template <typename V, std::size_t N, typename O>
ndarray<V, 1, c_order> flatten(const ndarray<V, N, O>& a) {
    ndarray<V, 1, c_order> result(shape<1>{a.size()});
    std::size_t idx = 0;
    a.for_each_element([&](const V& v) {
        result(idx++) = v;
    });
    return result;
}

} // namespace mtl::array
