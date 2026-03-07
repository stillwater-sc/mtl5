#pragma once
// MTL5 -- Compile-time traits for BLAS/LAPACK dispatch decisions
// Used by operation files to select hardware-accelerated paths when available.

#include <type_traits>
#include <mtl/tag/orientation.hpp>

namespace mtl::interface {

/// True for scalar types supported by standard BLAS/LAPACK (float, double).
template <typename T>
inline constexpr bool is_blas_scalar_v =
    std::is_same_v<T, float> || std::is_same_v<T, double>;

/// Concept satisfied by dense matrix types eligible for BLAS/LAPACK dispatch.
/// Requires: float/double value_type, contiguous data() pointer, num_rows/num_cols.
template <typename M>
concept BlasDenseMatrix =
    is_blas_scalar_v<typename M::value_type> &&
    requires(const M& m) {
        { m.data() } -> std::convertible_to<const typename M::value_type*>;
        { m.num_rows() };
        { m.num_cols() };
    };

/// Concept satisfied by dense vector types eligible for BLAS dispatch.
/// Requires: float/double value_type, contiguous data() pointer, size().
template <typename V>
concept BlasDenseVector =
    is_blas_scalar_v<typename V::value_type> &&
    requires(const V& v) {
        { v.data() } -> std::convertible_to<const typename V::value_type*>;
        { v.size() };
    };

/// Check if a matrix type uses row-major orientation.
template <typename M>
inline constexpr bool is_row_major_v =
    std::is_same_v<typename M::orientation, tag::row_major>;

} // namespace mtl::interface
