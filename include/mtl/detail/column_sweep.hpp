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
#include <utility>

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

/// Columns per pass over a strip (#884). One column per pass loads and stores
/// every accumulator of the strip for each column, so on a short strip the
/// loop is bound by that traffic, not by A: a 4 x 1000 gemv ran at half the
/// row kernel's speed. Eight columns per pass keep each accumulator in a
/// register across eight terms, measured 1.3-2x faster at every shape on an
/// i7-12700K, generic and -march=native.
inline constexpr std::size_t column_sweep_columns = 8;

/// Columns [cb, ce) of the column-major a (leading dimension ld) into the
/// strip's accumulators: acc[l] = op(acc[l], a[c * ld + r0 + l], x(c)) for
/// l < rr and c ascending. C columns per pass, then 4, then 1 for the rest.
/// Every accumulator takes the same terms in the same order with the same
/// expression as one column per pass, so the result is bit-identical to it
/// (and so to the one-row loop the sweep matches).
template <std::size_t C, typename Acc, typename T, typename XAt, typename Op>
void sweep_columns(Acc* acc, const T* a, std::size_t ld, std::size_t r0, std::size_t rr,
                   std::size_t cb, std::size_t ce, const XAt& x_at, const Op& op);

/// Strips of at most this many rows take sweep_fixed_rows (#884).
inline constexpr std::size_t column_sweep_fixed_rows = 16;

/// sweep_columns for a strip of exactly R rows, R known at compile time: the R
/// accumulators are indexed only by a pack, so they stay in registers across
/// all the columns, where the run-time strip loads and stores them every pass.
/// On a 4-row strip that measured 1.5-4.8x faster than the run-time strip
/// (i7-12700K, GCC and Clang, generic and -march=native). The same terms in
/// the same order, so the same bits.
template <std::size_t R, typename Acc, typename T, typename XAt, typename Op>
void sweep_fixed_rows(Acc* acc, const T* a, std::size_t ld, std::size_t r0,
                      std::size_t cb, std::size_t ce, const XAt& x_at, const Op& op) {
    [&]<std::size_t... K>(std::index_sequence<K...>) {
        Acc t[R] = {acc[K]...};
        for (std::size_t c = cb; c < ce; ++c) {
            const T* col = a + c * ld + r0;
            const auto& xc = x_at(c);
            ((t[K] = op(t[K], col[K], xc)), ...);
        }
        ((acc[K] = t[K]), ...);
    }(std::make_index_sequence<R>{});
}

/// Columns [cb, ce) into a strip of rr rows: sweep_fixed_rows for a short
/// strip, sweep_columns<C> otherwise. Bit-identical either way.
template <std::size_t C, typename Acc, typename T, typename XAt, typename Op>
void strip_columns(Acc* acc, const T* a, std::size_t ld, std::size_t r0, std::size_t rr,
                   std::size_t cb, std::size_t ce, const XAt& x_at, const Op& op) {
    if (rr == 0 || cb >= ce) return;
    if (rr <= column_sweep_fixed_rows) {
        [&]<std::size_t... R>(std::index_sequence<R...>) {
            (void)((rr == R + 1 ? (sweep_fixed_rows<R + 1>(acc, a, ld, r0, cb, ce, x_at, op), true) : false) || ...);
        }(std::make_index_sequence<column_sweep_fixed_rows>{});
        return;
    }
    sweep_columns<C>(acc, a, ld, r0, rr, cb, ce, x_at, op);
}

template <std::size_t C, typename Acc, typename T, typename XAt, typename Op>
void sweep_columns(Acc* acc, const T* a, std::size_t ld, std::size_t r0, std::size_t rr,
                   std::size_t cb, std::size_t ce, const XAt& x_at, const Op& op) {
    std::size_t c = cb;
    if constexpr (C > 1) {
        using XV = std::remove_cvref_t<decltype(x_at(cb))>;
        for (; ce - c >= C; c += C) {
            const T* col[C];
            XV xs[C];
            for (std::size_t k = 0; k < C; ++k) {
                col[k] = a + (c + k) * ld + r0;
                xs[k] = x_at(c + k);
            }
            for (std::size_t l = 0; l < rr; ++l) {
                Acc t = acc[l];
                [&]<std::size_t... K>(std::index_sequence<K...>) {
                    ((t = op(t, col[K][l], xs[K])), ...);
                }(std::make_index_sequence<C>{});
                acc[l] = t;
            }
        }
        if constexpr (C > 4)
            sweep_columns<4>(acc, a, ld, r0, rr, c, ce, x_at, op);
        else
            sweep_columns<1>(acc, a, ld, r0, rr, c, ce, x_at, op);
    } else {
        for (; c < ce; ++c) {
            const auto& xc = x_at(c);
            const T* col = a + c * ld + r0;
            for (std::size_t l = 0; l < rr; ++l) acc[l] = op(acc[l], col[l], xc);
        }
    }
}

} // namespace mtl::detail
