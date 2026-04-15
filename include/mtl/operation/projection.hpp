#pragma once
// MTL5 -- Type projection and embedding for mixed-precision workflows
//
// project_onto<Target>(source): wider -> narrower (lossy)
//   Converts each element from a wider type to a narrower type.
//   Mathematical analogy: projecting from a higher-dimensional
//   representational space onto a lower-dimensional subspace.
//
// embed_into<Target>(source): narrower -> wider (lossless)
//   Converts each element from a narrower type to a wider type.
//   The value is exactly representable in the target type.
//   Mathematical analogy: embedding into a higher-dimensional space.
//
// Compile-time directional enforcement via concepts:
//   project_onto requires Target has fewer or equal digits than Source
//   embed_into requires Target has more or equal digits than Source
//
// Supported containers:
//   - dense_vector<T> -> dense_vector<U>
//   - dense2D<T> -> dense2D<U>

#include <cstddef>
#include <limits>
#include <mtl/concepts/scalar.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/mat/dense2D.hpp>

namespace mtl {

// ============================================================================
// Compile-time directional constraints
// ============================================================================

/// ProjectableOnto: Target has fewer (or equal) significant digits than Source.
/// This validates that project_onto is a narrowing (lossy) conversion.
template <typename Target, typename Source>
concept ProjectableOnto =
    std::numeric_limits<Target>::digits <= std::numeric_limits<Source>::digits;

/// EmbeddableInto: Target has more (or equal) significant digits than Source.
/// This validates that embed_into is a widening (lossless) conversion.
template <typename Target, typename Source>
concept EmbeddableInto =
    std::numeric_limits<Target>::digits >= std::numeric_limits<Source>::digits;

// ============================================================================
// project_onto<Target>: wider -> narrower (lossy)
// ============================================================================

/// Project a dense vector onto a narrower element type.
template <Scalar Target, Scalar Source>
    requires ProjectableOnto<Target, Source>
vec::dense_vector<Target> project_onto(const vec::dense_vector<Source>& src) {
    vec::dense_vector<Target> dst(src.size());
    for (std::size_t i = 0; i < src.size(); ++i)
        dst(i) = static_cast<Target>(src(i));
    return dst;
}

/// Scaled projection of a dense vector: dst(i) = static_cast<Target>(scale * src(i)).
/// The scale factor maps the source dynamic range onto the target dynamic range,
/// essential for quantization workflows (e.g., float -> int8 with scale = 127).
template <Scalar Target, Scalar Source, typename ScaleType>
    requires ProjectableOnto<Target, Source>
vec::dense_vector<Target> project_onto(const vec::dense_vector<Source>& src, ScaleType scale) {
    vec::dense_vector<Target> dst(src.size());
    for (std::size_t i = 0; i < src.size(); ++i)
        dst(i) = static_cast<Target>(scale * src(i));
    return dst;
}

/// Project a dense 2D matrix onto a narrower element type.
template <Scalar Target, Scalar Source>
    requires ProjectableOnto<Target, Source>
mat::dense2D<Target> project_onto(const mat::dense2D<Source>& src) {
    mat::dense2D<Target> dst(src.num_rows(), src.num_cols());
    for (std::size_t r = 0; r < src.num_rows(); ++r)
        for (std::size_t c = 0; c < src.num_cols(); ++c)
            dst(r, c) = static_cast<Target>(src(r, c));
    return dst;
}

/// Scaled projection of a dense 2D matrix: dst(r,c) = static_cast<Target>(scale * src(r,c)).
template <Scalar Target, Scalar Source, typename ScaleType>
    requires ProjectableOnto<Target, Source>
mat::dense2D<Target> project_onto(const mat::dense2D<Source>& src, ScaleType scale) {
    mat::dense2D<Target> dst(src.num_rows(), src.num_cols());
    for (std::size_t r = 0; r < src.num_rows(); ++r)
        for (std::size_t c = 0; c < src.num_cols(); ++c)
            dst(r, c) = static_cast<Target>(scale * src(r, c));
    return dst;
}

// ============================================================================
// embed_into<Target>: narrower -> wider (lossless)
// ============================================================================

/// Embed a dense vector into a wider element type.
template <Scalar Target, Scalar Source>
    requires EmbeddableInto<Target, Source>
vec::dense_vector<Target> embed_into(const vec::dense_vector<Source>& src) {
    vec::dense_vector<Target> dst(src.size());
    for (std::size_t i = 0; i < src.size(); ++i)
        dst(i) = static_cast<Target>(src(i));
    return dst;
}

/// Scaled embedding of a dense vector: dst(i) = static_cast<Target>(scale * src(i)).
/// Typically used with the inverse of the projection scale for dequantization
/// (e.g., int8 -> float with scale = 1.0/127).
template <Scalar Target, Scalar Source, typename ScaleType>
    requires EmbeddableInto<Target, Source>
vec::dense_vector<Target> embed_into(const vec::dense_vector<Source>& src, ScaleType scale) {
    vec::dense_vector<Target> dst(src.size());
    for (std::size_t i = 0; i < src.size(); ++i)
        dst(i) = static_cast<Target>(scale * src(i));
    return dst;
}

/// Embed a dense 2D matrix into a wider element type.
template <Scalar Target, Scalar Source>
    requires EmbeddableInto<Target, Source>
mat::dense2D<Target> embed_into(const mat::dense2D<Source>& src) {
    mat::dense2D<Target> dst(src.num_rows(), src.num_cols());
    for (std::size_t r = 0; r < src.num_rows(); ++r)
        for (std::size_t c = 0; c < src.num_cols(); ++c)
            dst(r, c) = static_cast<Target>(src(r, c));
    return dst;
}

/// Scaled embedding of a dense 2D matrix: dst(r,c) = static_cast<Target>(scale * src(r,c)).
template <Scalar Target, Scalar Source, typename ScaleType>
    requires EmbeddableInto<Target, Source>
mat::dense2D<Target> embed_into(const mat::dense2D<Source>& src, ScaleType scale) {
    mat::dense2D<Target> dst(src.num_rows(), src.num_cols());
    for (std::size_t r = 0; r < src.num_rows(); ++r)
        for (std::size_t c = 0; c < src.num_cols(); ++c)
            dst(r, c) = static_cast<Target>(scale * src(r, c));
    return dst;
}

} // namespace mtl
