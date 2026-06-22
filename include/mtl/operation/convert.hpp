#pragma once
// MTL5 -- element-wise tensor convert/cast between number types (issue #164)
//
// A standalone re-quantization operator: copy a dense vector or matrix into a
// different element type (e.g. fp32 -> bfloat16 to stage a tensor at lower
// precision between kernels, or a low-precision tensor up to fp64 for analysis).
//
// IMPORTANT: this is for NON-FUSED re-typing. It is *not* the accumulate->output
// path -- a mixed-precision dot/gemv/gemm fuses the accumulator->result
// conversion into its final store (see mtl::math::accumulator_traits::value and
// the Accumulator/Result parameters on dot/mult). Reach for `convert` only when
// you genuinely need a separate pass over a stored tensor.

#include <cassert>
#include <cstddef>
#include <mtl/concepts/vector.hpp>
#include <mtl/concepts/matrix.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/mat/dense2D.hpp>

namespace mtl {

/// Convert a vector into a pre-allocated destination of (possibly) another type.
template <Vector VSrc, Vector VDst>
void convert(const VSrc& src, VDst& dst) {
    assert(static_cast<std::size_t>(src.size()) == static_cast<std::size_t>(dst.size()));
    using D = typename VDst::value_type;
    for (typename VSrc::size_type i = 0; i < src.size(); ++i)
        dst(static_cast<int>(i)) = static_cast<D>(src(static_cast<int>(i)));
}

/// Convert a matrix into a pre-allocated destination of (possibly) another type.
template <Matrix MSrc, Matrix MDst>
void convert(const MSrc& src, MDst& dst) {
    assert(src.num_rows() == dst.num_rows());
    assert(src.num_cols() == dst.num_cols());
    using D = typename MDst::value_type;
    for (typename MSrc::size_type r = 0; r < src.num_rows(); ++r)
        for (typename MSrc::size_type c = 0; c < src.num_cols(); ++c)
            dst(r, c) = static_cast<D>(src(r, c));
}

/// Convert a vector to a new dense_vector<DstValue> (out-of-place convenience).
template <typename DstValue, Vector VSrc>
vec::dense_vector<DstValue> convert(const VSrc& src) {
    vec::dense_vector<DstValue> dst(static_cast<typename vec::dense_vector<DstValue>::size_type>(src.size()));
    convert(src, dst);
    return dst;
}

/// Convert a matrix to a new dense2D<DstValue> (out-of-place convenience).
template <typename DstValue, Matrix MSrc>
mat::dense2D<DstValue> convert(const MSrc& src) {
    mat::dense2D<DstValue> dst(src.num_rows(), src.num_cols());
    convert(src, dst);
    return dst;
}

} // namespace mtl
