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
// #884 made it faster on small and short, wide matrices, where it had lost to
// the row kernels: eight columns per pass, a strip of at most 16 rows in
// registers, the triangular diagonal blocks eight columns at a time, and the
// row kernels for the smallest shapes. The constants below say where each
// applies and what it was measured against; docs/benchmarks/column-sweep.md
// has the five machines' results.
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

/// GCC's loop vectorizer off for one function, the SLP vectorizer on (#884,
/// #898). In the fixed-row strip (and so in the row kernels on a column-major
/// A under GCC on x86-64, row_kernel_fixed_rows_v), the R accumulators are
/// R independent chains over the columns. GCC's SLP vectorizer packs them into
/// registers, which is what these loops want. With AVX its loop vectorizer can
/// instead vectorize the column loop itself, as an in-order reduction: R x 4
/// columns loaded, then each chain rebuilt with shuffles and scalar merges.
/// Which one it picks depends on the calling context, so the same
/// instantiation ran at 0.35 us in one binary and 2.2 us in another (64 x 64
/// f64 gemv, i7-12700K, -O3 -march=native). Clang and MSVC don't do it.
#if defined(__GNUC__) && !defined(__clang__)
#define MTL5_NO_LOOP_VECTORIZE __attribute__((optimize("no-tree-loop-vectorize")))
#else
#define MTL5_NO_LOOP_VECTORIZE
#endif

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
/// register across eight terms.
///
/// Measured with bench_column_sweep on five machines (i7-12700K, Xeon
/// E5-2420 v2, Ryzen 7 8845HS, Jetson Orin, Arduino Ventuno Q), GCC and Clang,
/// generic and native: gemv 1.15-2.2x faster than one column per pass in
/// geometric mean over the shapes. MSVC is the exception: its code for the
/// eight-column pass ran at 0.23-0.89 of the one-column loop from 48 rows up,
/// so under MSVC the sweep stays as #777 wrote it.
#if defined(_MSC_VER) && !defined(__clang__)
inline constexpr std::size_t column_sweep_columns = 1;
#else
inline constexpr std::size_t column_sweep_columns = 8;
#endif

/// A run-time strip goes one column per pass when A's columns are this many
/// bytes apart or more (#884): eight columns per pass then read eight streams
/// that far apart, and on the Jetson Orin and the Ventuno that ran at 0.46-0.75
/// of one column per pass at n = 2048-4096 (8 KB and up), as on the Xeon at
/// 4096. Below it the eight-column pass won on every machine.
inline constexpr std::size_t column_sweep_unrolled_stride = 8192;

/// Whether a sweep with C columns per pass over columns ld elements of T apart
/// takes the unrolled path (strip_columns<C>) rather than sweep_columns<1>.
template <std::size_t C, typename T>
constexpr bool column_sweep_unrolled(std::size_t ld) {
    return C > 1 && ld * sizeof(T) < column_sweep_unrolled_stride;
}

/// The shapes the row kernels take although A is column-major (#884): where
/// the sweep's per-strip fixed cost isn't amortized. Measured across the five
/// machines above, a size up to these minimized the geometric mean of the time
/// over the best of the sweep and the row kernel. #884 set gemv m <= 2 and
/// trmv n <= 32, held there by GCC's code for the row kernels on a
/// column-major A on x86 (1.3-3.3x slower native than generic); with that
/// fixed (#898, row_kernel_fixed_rows_v) the same simulation gives m <= 4 and
/// n <= 64: gemv 1.110 -> 1.098 of the best, trmv 1.046 -> 1.013, and the
/// worst case 7.9 -> 5.6 and 2.0 -> 1.5.
inline constexpr std::size_t column_sweep_gemv_row_kernel_rows = 4;   // gemv: m <= 4
inline constexpr std::size_t column_sweep_trmv_row_kernel_n = 64;     // trmv: n <= 64
inline constexpr std::size_t column_sweep_trsv_row_kernel_n = 64;     // forward substitution: n <= 64

/// Columns [cb, ce) of the column-major a (leading dimension ld) into the
/// strip's accumulators: acc[l] = op(acc[l], a[c * ld + r0 + l], x(c)) for
/// l < rr and c ascending. C columns per pass, then 4, then 1 for the rest.
/// Every accumulator takes the same terms in the same order with the same
/// expression as one column per pass, so the result is bit-identical to it
/// (and so to the one-row loop the sweep matches).
template <std::size_t C, typename Acc, typename T, typename XAt, typename Op>
void sweep_columns(Acc* acc, const T* a, std::size_t ld, std::size_t r0, std::size_t rr,
                   std::size_t cb, std::size_t ce, const XAt& x_at, const Op& op);

/// Whether the row kernels (mult_rows_interleaved, trmv_upper_rows,
/// trmv_lower_rows) run a block's column loop as sweep_fixed_rows<W> on a
/// contiguous column-major A (#898): GCC on x86-64 only. There GCC's loop
/// vectorizer could turn the loop into an in-order reduction across the
/// columns, which on the i7-12700K and the Xeon E5-2420 v2 ran gemv 2.0-4.0x
/// and trmv 1.6-2.4x slower than sweep_fixed_rows (geometric mean over the
/// shapes up to 256, generic and -march=native). Elsewhere the row kernel's
/// own loop measured as fast or faster: GCC on the Jetson and the Ventuno,
/// Clang and MSVC lost up to 2x on trmv through sweep_fixed_rows. A row-major
/// A always keeps its loop.
template <typename Acc, typename M>
inline constexpr bool row_kernel_fixed_rows_v =
#if defined(__GNUC__) && !defined(__clang__) && defined(__x86_64__)
    column_sweep_v<Acc, M>;
#else
    false;
#endif

/// ... and only over at least this many columns: below it the call costs more
/// than the loop (2 x 2 and 4 x 4 ran at 0.46-0.83 through it).
inline constexpr std::size_t row_kernel_fixed_rows_min_columns = 16;

/// Strips of at most this many rows take sweep_fixed_rows (#884).
inline constexpr std::size_t column_sweep_fixed_rows = 16;

/// sweep_columns for a strip of exactly R rows, R known at compile time: the R
/// accumulators are indexed only by a pack, so they stay in registers across
/// all the columns, where the run-time strip loads and stores them every pass.
/// On a 4-row strip that measured 1.5-4.8x faster than the run-time strip
/// (i7-12700K, GCC and Clang, generic and -march=native). The same terms in
/// the same order, so the same bits.
///
/// GCC's loop vectorizer is off for it: with AVX it vectorized the column
/// loop, eight columns per iteration put back together with shuffles, where
/// the SLP form (the R accumulators in one register, x broadcast per column)
/// is what this loop wants. On a Xeon E5-2420 v2 the f32 4-row strip ran at
/// half the speed of one column per pass (#884). The SLP vectorizer stays on.
template <std::size_t R, typename Acc, typename T, typename XAt, typename Op>
MTL5_NO_LOOP_VECTORIZE
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
/// strip, sweep_columns<C> otherwise. Bit-identical either way. The callers
/// take sweep_columns<1> themselves at a wide column stride
/// (column_sweep_unrolled<T>): behind this extra call, the same one-column loop
/// measured 25-35% slower than called directly (#884).
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
