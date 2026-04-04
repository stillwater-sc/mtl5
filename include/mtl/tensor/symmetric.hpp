#pragma once
// MTL5 -- Symmetric and antisymmetric tensors with reduced storage
//
// symmetric_tensor<V, 2, D>:     D*(D+1)/2 independent components
// antisymmetric_tensor<V, 2, D>: D*(D-1)/2 independent components
//
// Voigt notation support for the rank-4 elasticity tensor (21 components in 3D).

#include <array>
#include <cassert>
#include <cstddef>
#include <type_traits>

#include <mtl/config.hpp>
#include <mtl/tensor/tensor.hpp>

namespace mtl::tensor {

// ── Symmetric rank-2 tensor ────────────────────────────────────────

namespace detail_sym {

/// Number of independent components for a symmetric rank-2 tensor in D dimensions.
constexpr std::size_t sym2_count(std::size_t D) { return D * (D + 1) / 2; }

/// Map (i, j) with i <= j to a flat index in upper-triangular storage.
/// Storage order: (0,0), (0,1), (0,2), ..., (1,1), (1,2), ..., (D-1,D-1)
constexpr std::size_t sym2_index(std::size_t i, std::size_t j, std::size_t D) {
    if (i > j) { auto tmp = i; i = j; j = tmp; }
    return i * D - i * (i + 1) / 2 + j;
}

/// Number of independent components for an antisymmetric rank-2 tensor.
constexpr std::size_t asym2_count(std::size_t D) { return D * (D - 1) / 2; }

/// Map (i, j) with i < j to a flat index. Returns D*(D-1)/2 for diagonal.
constexpr std::size_t asym2_index(std::size_t i, std::size_t j, std::size_t D) {
    if (i == j) return D * (D - 1) / 2; // sentinel for zero diagonal
    bool neg = (i > j);
    if (neg) { auto tmp = i; i = j; j = tmp; }
    return i * D - i * (i + 1) / 2 + j - i - 1;
}

} // namespace detail_sym

/// Symmetric rank-2 tensor: stores only D*(D+1)/2 independent components.
/// T(i,j) == T(j,i) is enforced structurally.
template <typename Value, std::size_t Dim>
class symmetric_tensor {
public:
    using value_type      = Value;
    using reference       = Value&;
    using const_reference = const Value&;
    using size_type       = std::size_t;

    static constexpr size_type rank      = 2;
    static constexpr size_type dimension = Dim;
    static constexpr size_type num_stored = detail_sym::sym2_count(Dim);
    static constexpr size_type num_components = Dim * Dim;  // logical size

private:
    std::array<Value, num_stored> data_{};

public:
    symmetric_tensor() = default;

    explicit symmetric_tensor(const Value& val) { data_.fill(val); }

    /// Access: automatically symmetrizes indices.
    reference operator()(size_type i, size_type j) {
        return data_[detail_sym::sym2_index(i, j, Dim)];
    }

    const_reference operator()(size_type i, size_type j) const {
        return data_[detail_sym::sym2_index(i, j, Dim)];
    }

    /// Number of stored (independent) components.
    static constexpr size_type stored_size() { return num_stored; }

    /// Convert to a full (non-symmetric) tensor.
    tensor<Value, 2, Dim> to_full() const {
        tensor<Value, 2, Dim> result;
        for (size_type i = 0; i < Dim; ++i)
            for (size_type j = 0; j < Dim; ++j)
                result(i, j) = (*this)(i, j);
        return result;
    }

    /// Construct from a full tensor (symmetrizes by averaging).
    static symmetric_tensor from_full(const tensor<Value, 2, Dim>& t) {
        symmetric_tensor result;
        for (size_type i = 0; i < Dim; ++i)
            for (size_type j = i; j < Dim; ++j)
                result(i, j) = (t(i, j) + t(j, i)) / Value{2};
        return result;
    }

    // -- Arithmetic ----------------------------------------------------------

    symmetric_tensor& operator+=(const symmetric_tensor& other) {
        for (size_type i = 0; i < num_stored; ++i) data_[i] += other.data_[i];
        return *this;
    }

    symmetric_tensor& operator-=(const symmetric_tensor& other) {
        for (size_type i = 0; i < num_stored; ++i) data_[i] -= other.data_[i];
        return *this;
    }

    symmetric_tensor& operator*=(const Value& s) {
        for (size_type i = 0; i < num_stored; ++i) data_[i] *= s;
        return *this;
    }

    friend symmetric_tensor operator+(symmetric_tensor a, const symmetric_tensor& b) { return a += b; }
    friend symmetric_tensor operator-(symmetric_tensor a, const symmetric_tensor& b) { return a -= b; }
    friend symmetric_tensor operator*(symmetric_tensor a, const Value& s) { return a *= s; }
    friend symmetric_tensor operator*(const Value& s, symmetric_tensor a) { return a *= s; }

    bool operator==(const symmetric_tensor& other) const { return data_ == other.data_; }
    bool operator!=(const symmetric_tensor& other) const { return data_ != other.data_; }
};

/// Trace of a symmetric tensor.
template <typename V, std::size_t D>
V trace(const symmetric_tensor<V, D>& t) {
    V result = V{0};
    for (std::size_t i = 0; i < D; ++i) result += t(i, i);
    return result;
}

// ── Antisymmetric rank-2 tensor ────────────────────────────────────

/// Antisymmetric (skew-symmetric) rank-2 tensor.
/// T(i,j) == -T(j,i), T(i,i) == 0.
/// Stores only D*(D-1)/2 independent components (strictly upper triangle).
template <typename Value, std::size_t Dim>
class antisymmetric_tensor {
public:
    using value_type = Value;
    using size_type  = std::size_t;

    static constexpr size_type rank      = 2;
    static constexpr size_type dimension = Dim;
    static constexpr size_type num_stored = detail_sym::asym2_count(Dim);

private:
    std::array<Value, num_stored> data_{};
    static inline const Value zero_val_ = Value{0};

public:
    antisymmetric_tensor() = default;

    /// Set the (i,j) component where i < j. Setting (j,i) sets -(i,j).
    void set(size_type i, size_type j, const Value& val) {
        assert(i != j && "Diagonal of antisymmetric tensor is always zero");
        if (i < j) {
            data_[detail_sym::asym2_index(i, j, Dim)] = val;
        } else {
            data_[detail_sym::asym2_index(j, i, Dim)] = -val;
        }
    }

    /// Access: returns -component for (j,i), zero for diagonal.
    Value operator()(size_type i, size_type j) const {
        if (i == j) return Value{0};
        if (i < j) return data_[detail_sym::asym2_index(i, j, Dim)];
        return -data_[detail_sym::asym2_index(j, i, Dim)];
    }

    static constexpr size_type stored_size() { return num_stored; }

    /// Convert to a full tensor.
    tensor<Value, 2, Dim> to_full() const {
        tensor<Value, 2, Dim> result;
        for (size_type i = 0; i < Dim; ++i)
            for (size_type j = 0; j < Dim; ++j)
                result(i, j) = (*this)(i, j);
        return result;
    }

    bool operator==(const antisymmetric_tensor& other) const { return data_ == other.data_; }
};

} // namespace mtl::tensor
