#pragma once
// MTL5 -- the column sweep: the generic row kernels on a column-major A (#777)
//
// The generic gemv, trmv and forward substitution sum W rows per pass, one
// accumulator per row, reading A(r0..r0+W-1, c) for each column c (#749,
// #772). On a column-major A those W values are contiguous, but the next
// column's are m elements away, so the pass touches a fresh cache line and
// often a fresh page for every W elements. At n = 4096 that ran at 2.5-6 GB/s.
//
// The column sweep keeps a STRIP of S rows' accumulators in a buffer on the
// stack (16 KB, so it stays in L1) and walks down each column in turn:
//
//     for each column c:  acc[0..S) = acc[0..S) + A(r0..r0+S-1, c) * x(c)
//
// That reads A column by column, contiguously, and the inner loop is a plain
// lane-wise multiply and add the compiler vectorizes on its own: vertical
// SIMD, one row per lane, with no Highway needed. Each row still adds the same
// terms in ascending column order with the same expression as the one-row
// loop, so the result is BIT-IDENTICAL to it, as the row kernels are.
//
// On an i7-12700K (n = 4096, one core, generic build), f32 gemv on a
// column-major A went from 26.5 to 2.2 ms (about 78% of the bandwidth
// roofline; 1.9 ms under -march=native), and f64 from 26.3 to 4.7 ms.
//
// It applies to a contiguous column-major A (dense2D) with an arithmetic
// accumulator: float, double, long double and the integers. Other types keep
// the row kernels, which a user's number type may well prefer; a row-major A
// is better served by them too (vertical lanes there need a transpose, which
// measured 1.4-2x slower than the scalar rows, #777).

#include <algorithm>
#include <cstddef>
#include <type_traits>

#include <mtl/interface/dispatch_traits.hpp>
#include <mtl/tag/orientation.hpp>

namespace mtl::detail {

/// Whether M is a dense matrix whose data() is column-major with leading
/// dimension num_rows(): element (i, j) at data()[j * num_rows() + i].
template <typename M>
inline constexpr bool col_major_contiguous_v = [] {
    if constexpr (interface::ContiguousMatrixData<M> && requires { typename M::orientation; })
        return std::is_same_v<typename M::orientation, tag::col_major>;
    else
        return false;
}();

/// Whether the generic row kernels take the column sweep for an accumulator
/// of type Acc over a matrix of type M.
template <typename Acc, typename M>
inline constexpr bool column_sweep_v =
    col_major_contiguous_v<M> && std::is_arithmetic_v<std::remove_cvref_t<Acc>> &&
    std::is_arithmetic_v<typename M::value_type>;

/// Rows per strip: 16 KB of accumulators, at least one.
template <typename T>
inline constexpr std::size_t column_strip_rows = std::max<std::size_t>(1, 16384 / sizeof(T));

} // namespace mtl::detail
