#pragma once
// MTL5 -- Metric tensor operations: raising and lowering indices
//
// A metric tensor g_ij enables converting between covariant and
// contravariant components:
//   raise:  T^i = g^ij * T_j     (covariant → contravariant)
//   lower:  T_i = g_ij * T^j     (contravariant → covariant)
//
// Provides common metrics: Euclidean (identity) and Minkowski.

#include <cstddef>
#include <mtl/tensor/tensor.hpp>

namespace mtl::tensor {

// ── Standard metrics ───────────────────────────────────────────────

/// Euclidean metric: g_ij = delta_ij (identity matrix).
template <typename V, std::size_t D>
tensor<V, 2, D> euclidean_metric() {
    return identity<V, D>();
}

/// Minkowski metric: diag(-1, +1, +1, +1) for 4D spacetime.
/// Convention: (-,+,+,+) "mostly plus" / particle physics convention.
template <typename V>
tensor<V, 2, 4> minkowski_metric() {
    tensor<V, 2, 4> g;
    g(0, 0) = V{-1};
    g(1, 1) = V{1};
    g(2, 2) = V{1};
    g(3, 3) = V{1};
    return g;
}

// ── Index raising/lowering for rank-1 tensors ──────────────────────

/// Lower a contravariant vector: v_i = g_ij * v^j
template <typename V, std::size_t D>
tensor<V, 1, D> lower(const tensor<V, 1, D>& v, const tensor<V, 2, D>& g) {
    tensor<V, 1, D> result;
    for (std::size_t i = 0; i < D; ++i) {
        V sum = V{0};
        for (std::size_t j = 0; j < D; ++j) sum += g(i, j) * v(j);
        result(i) = sum;
    }
    return result;
}

/// Raise a covariant vector: v^i = g^ij * v_j
/// Here g_inv is the inverse metric g^ij.
template <typename V, std::size_t D>
tensor<V, 1, D> raise(const tensor<V, 1, D>& v, const tensor<V, 2, D>& g_inv) {
    return lower(v, g_inv); // same operation, just the metric meaning differs
}

// ── Index raising/lowering for rank-2 tensors ──────────────────────

/// Lower the first index of a rank-2 tensor: T_ij = g_ik * T^k_j
template <typename V, std::size_t D>
tensor<V, 2, D> lower_first(const tensor<V, 2, D>& t, const tensor<V, 2, D>& g) {
    tensor<V, 2, D> result;
    for (std::size_t i = 0; i < D; ++i)
        for (std::size_t j = 0; j < D; ++j) {
            V sum = V{0};
            for (std::size_t k = 0; k < D; ++k) sum += g(i, k) * t(k, j);
            result(i, j) = sum;
        }
    return result;
}

/// Lower the second index of a rank-2 tensor: T_ij = T^i_k * g_kj
template <typename V, std::size_t D>
tensor<V, 2, D> lower_second(const tensor<V, 2, D>& t, const tensor<V, 2, D>& g) {
    tensor<V, 2, D> result;
    for (std::size_t i = 0; i < D; ++i)
        for (std::size_t j = 0; j < D; ++j) {
            V sum = V{0};
            for (std::size_t k = 0; k < D; ++k) sum += t(i, k) * g(k, j);
            result(i, j) = sum;
        }
    return result;
}

/// Raise the first index of a rank-2 tensor: T^i_j = g^ik * T_kj
template <typename V, std::size_t D>
tensor<V, 2, D> raise_first(const tensor<V, 2, D>& t, const tensor<V, 2, D>& g_inv) {
    return lower_first(t, g_inv);
}

/// Raise the second index of a rank-2 tensor: T^i_j = T_ik * g^kj
template <typename V, std::size_t D>
tensor<V, 2, D> raise_second(const tensor<V, 2, D>& t, const tensor<V, 2, D>& g_inv) {
    return lower_second(t, g_inv);
}

} // namespace mtl::tensor
