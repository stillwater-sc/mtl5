#pragma once
// MTL5 -- Tensor index algebra and Einstein summation
//
// Provides:
//   - Index variance tags: up (contravariant) and down (covariant)
//   - Named index objects: Index<'i'>, Index<'j'>, ...
//   - contract(A(i,j), B(j,k)) — Einstein summation over repeated indices
//   - Compile-time detection of repeated indices and validation

#include <array>
#include <cstddef>
#include <type_traits>

#include <mtl/tensor/tensor.hpp>

namespace mtl::tensor {

// ── Index variance tags ────────────────────────────────────────────

/// Contravariant (upper) index — transforms inversely to basis vectors.
struct up {};

/// Covariant (lower) index — transforms with basis vectors.
struct down {};

// ── Named index objects ────────────────────────────────────────────

/// Compile-time named index for Einstein notation.
/// Usage: Index<'i'> i; Index<'j'> j;
template <char Name>
struct Index {
    static constexpr char name = Name;
};

// ── Index expression (tensor bound to named indices) ───────────────

/// A tensor expression with named indices attached.
/// Created by: A(i, j) where i, j are Index objects.
template <typename TensorType, char... Names>
struct indexed_tensor {
    const TensorType& ref;
    static constexpr std::size_t num_indices = sizeof...(Names);
    static constexpr std::array<char, sizeof...(Names)> names{Names...};
};

// ── Binding tensors to indices ─────────────────────────────────────

/// Bind a rank-2 tensor to two named indices: A(i, j)
template <typename V, std::size_t D, char I, char J>
auto bind(const tensor<V, 2, D>& t, Index<I>, Index<J>) {
    return indexed_tensor<tensor<V, 2, D>, I, J>{t};
}

/// Bind a rank-1 tensor to one named index: v(i)
template <typename V, std::size_t D, char I>
auto bind(const tensor<V, 1, D>& t, Index<I>) {
    return indexed_tensor<tensor<V, 1, D>, I>{t};
}

// ── Contraction (Einstein summation) ───────────────────────────────

namespace detail_index {

/// Find the repeated index between two name arrays.
/// Returns the character of the repeated index, or '\0' if none.
template <std::size_t NA, std::size_t NB>
constexpr char find_repeated(const std::array<char, NA>& a,
                             const std::array<char, NB>& b) {
    for (std::size_t i = 0; i < NA; ++i)
        for (std::size_t j = 0; j < NB; ++j)
            if (a[i] == b[j]) return a[i];
    return '\0';
}

/// Find the position of a character in an array.
template <std::size_t N>
constexpr std::size_t find_pos(const std::array<char, N>& arr, char c) {
    for (std::size_t i = 0; i < N; ++i)
        if (arr[i] == c) return i;
    return N; // not found
}

} // namespace detail_index

/// Contract two rank-2 tensors over a repeated index.
/// C^i_k = sum_j A^i_j * B^j_k
///
/// Usage:
///   Index<'i'> i; Index<'j'> j; Index<'k'> k;
///   auto C = contract(bind(A, i, j), bind(B, j, k));
template <typename V, std::size_t D, char AI, char AJ, char BI, char BJ>
auto contract(const indexed_tensor<tensor<V, 2, D>, AI, AJ>& a,
              const indexed_tensor<tensor<V, 2, D>, BI, BJ>& b) {
    // Find the contracted (repeated) index
    constexpr std::array<char, 2> a_names{AI, AJ};
    constexpr std::array<char, 2> b_names{BI, BJ};
    constexpr char summed = detail_index::find_repeated(a_names, b_names);
    static_assert(summed != '\0', "contract: no repeated index found between the two tensors");

    // Identify which position in each tensor is summed
    constexpr std::size_t a_sum_pos = detail_index::find_pos(a_names, summed);
    constexpr std::size_t b_sum_pos = detail_index::find_pos(b_names, summed);

    // The result has the non-summed indices
    tensor<V, 2, D> result;
    for (std::size_t i = 0; i < D; ++i) {
        for (std::size_t k = 0; k < D; ++k) {
            V sum = V{0};
            for (std::size_t j = 0; j < D; ++j) {
                // Build indices for A and B based on which position is summed
                std::array<std::size_t, 2> a_idx, b_idx;
                if constexpr (a_sum_pos == 0) { a_idx = {j, i}; }
                else                          { a_idx = {i, j}; }
                if constexpr (b_sum_pos == 0) { b_idx = {j, k}; }
                else                          { b_idx = {k, j}; }
                sum += a.ref[a_idx] * b.ref[b_idx];
            }
            result(i, k) = sum;
        }
    }
    return result;
}

/// Contract a rank-2 tensor with a rank-1 tensor (matrix-vector product).
/// c^i = sum_j A^i_j * v^j
template <typename V, std::size_t D, char AI, char AJ, char VI>
auto contract(const indexed_tensor<tensor<V, 2, D>, AI, AJ>& a,
              const indexed_tensor<tensor<V, 1, D>, VI>& v) {
    constexpr std::array<char, 2> a_names{AI, AJ};
    constexpr std::array<char, 1> v_names{VI};
    constexpr char summed = detail_index::find_repeated(a_names, v_names);
    static_assert(summed != '\0', "contract: no repeated index found");

    constexpr std::size_t a_sum_pos = detail_index::find_pos(a_names, summed);

    tensor<V, 1, D> result;
    for (std::size_t i = 0; i < D; ++i) {
        V sum = V{0};
        for (std::size_t j = 0; j < D; ++j) {
            std::array<std::size_t, 2> a_idx;
            if constexpr (a_sum_pos == 0) { a_idx = {j, i}; }
            else                          { a_idx = {i, j}; }
            sum += a.ref[a_idx] * v.ref[{j}];
        }
        result(i) = sum;
    }
    return result;
}

// ── Outer product ──────────────────────────────────────────────────

/// Outer product of two rank-1 tensors: C_ij = a_i * b_j
template <typename V, std::size_t D>
tensor<V, 2, D> outer(const tensor<V, 1, D>& a, const tensor<V, 1, D>& b) {
    tensor<V, 2, D> result;
    for (std::size_t i = 0; i < D; ++i)
        for (std::size_t j = 0; j < D; ++j)
            result(i, j) = a(i) * b(j);
    return result;
}

/// Outer product of two rank-2 tensors: C_ijkl = A_ij * B_kl
template <typename V, std::size_t D>
tensor<V, 4, D> outer(const tensor<V, 2, D>& a, const tensor<V, 2, D>& b) {
    tensor<V, 4, D> result;
    for (std::size_t i = 0; i < D; ++i)
        for (std::size_t j = 0; j < D; ++j)
            for (std::size_t k = 0; k < D; ++k)
                for (std::size_t l = 0; l < D; ++l)
                    result(i, j, k, l) = a(i, j) * b(k, l);
    return result;
}

} // namespace mtl::tensor
