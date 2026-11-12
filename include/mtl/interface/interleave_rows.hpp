#pragma once
// MTL5 -- How many rows the generic row kernels interleave (#772)
//
// The generic mat*vec (detail::mult_generic_rows), the generic trmv and the
// generic forward substitution (detail::lower_trisolve_rows) each sum W rows
// in one pass, one accumulator per row (#749). Each row still sums its own
// terms in its own order, so the result is bit-identical for every W: the
// width is a pure speed knob, and changing it never changes a result.
//
// What W should be depends on the hardware, the element type, the storage
// order and the compiler:
//   - The floor is set by latency against bandwidth. A serial reduction adds
//     one term per add latency; W chains add W. That is what took f64 from
//     44% to about 100% of the bandwidth roofline at W = 4 (#749).
//   - The ceiling is set by the register file. Each chain holds one
//     accumulator, and spilling one costs more than the chain gains.
//   - In between, the compiler decides. GCC's SLP vectorizer packs the W
//     accumulators of a row-major A into vector registers, assembled from W
//     scalar loads a row apart, which costs more than it saves. For float,
//     the kernels stop it with detail::keep_scalar (#776); the measurements
//     below predate that, and with it f32 gemv at W = 6 takes 2.71 ms.
//
// So the default for float and double is measured, not derived, per ISA
// class, on the six #737 machines (generic build: no -march, n = 4096, one
// pinned core, GCC and Clang or MSVC and clang-cl; #780,
// docs/benchmarks/interleave-width.md). Each figure is the time at that
// width over the time at each kernel's fastest width:
//   - x86-64: 8. Over every machine, compiler, kernel and type it averages
//     1.015, worst 1.100 (Clang on the i7-12700K, f32 gemv); the old 6
//     averaged 1.041, worst 1.315 (clang-cl's f64 trmv on Zen 4). On both
//     Zen 4 machines, 8 is the fastest width or within 0.2% of it. The Ivy
//     Bridge Xeon prefers 4 and loses at most 6% at 8.
//   - AArch64: 4. Under Clang (Arduino Ventuno Q) 4 is the fastest width of
//     every kernel and the old 6 was 18-24% slower. Under GCC, gemv agrees on
//     both machines; on the Jetson Orin, f64 trmv and the forward substitution
//     are fastest at 2 and lose 12% at 4 (57-59% at 6).
//   - Except GCC on AArch64 for float trmv and the forward substitution: 2.
//     At W >= 4 those take the keep_scalar barrier (#776), and there they run
//     5.6-7.2x slower at 4 than at 2 on both machines, and 1.4-4.5x slower at
//     6 to 12 (#795). 2 is their fastest width (Arduino: trmv 4.76 ms against
//     6.79 at 6 and 31.7 at 4). gemv takes
//     the same barrier but is fastest at 4 on both machines, so keeps 4.
//   - Other targets: 6, the i7-12700K figure that was the default for every
//     target before (#772).
// Each is capped at half the vector registers the target has: 32-bit x86 and
// unrecognized targets get 4. The default is 4 for every other type: long
// double (x87 has eight stack slots), integers, and software number types,
// whose cost only the user knows.
//
// Run-time selection (#778) would recover the rest: the Xeon's 4 and GCC's
// 12 on the i7, which a portable x86-64 binary can't know at compile time.
//
// A contiguous column-major A with an arithmetic accumulator doesn't use W at
// all: it takes the column sweep (detail/column_sweep.hpp, #777), which reads
// A down its columns instead of W rows at a time.
//
// Another machine may well want another width; that is what the overrides
// and benchmarks/bench_interleave_rows are for. Every width gives the same
// bits, so picking one per machine never changes a result.
//
// Overriding it, most specific first:
//   1. Specialize interleave_rows<T, Kernel> for a type, a kernel, or both.
//   2. -DMTL5_INTERLEAVE_ROWS_F32=W or -DMTL5_INTERLEAVE_ROWS_F64=W for
//      float or double.
//   3. -DMTL5_INTERLEAVE_ROWS=W for every type and kernel. CMake passes it
//      from the MTL5_INTERLEAVE_ROWS cache variable.
//
// benchmarks/bench_interleave_rows times each width on the machine it runs
// on and prints the defines for the fastest.

#include <cstddef>
#include <type_traits>

namespace mtl::interface {

/// The kernels that take their width from interleave_rows.
namespace kernel {
struct gemv {};              ///< detail::mult_generic_rows, default accumulator
struct trmv {};              ///< the generic trmv, both triangles
struct lower_trisolve {};    ///< detail::lower_trisolve_rows (trsv with a lower A)
} // namespace kernel

/// The vector registers on the target, from the compiler's ISA macros: one
/// floating-point accumulator occupies one, scalar or not. Unknown targets get
/// the 32-bit x86 count, which keeps the default at its pre-#772 width of 4.
inline constexpr std::size_t vector_registers =
#if defined(__AVX512F__)
    32;
#elif defined(__aarch64__) || defined(_M_ARM64)
    32;
#elif defined(__x86_64__) || defined(_M_X64)
    16;
#else
    8;
#endif

/// The measured float/double width for Kernel on this target, before the
/// register cap (see the top of this file).
template <typename T, typename Kernel>
consteval std::size_t measured_interleave_rows() {
#if defined(__x86_64__) || defined(_M_X64)
    return 8;
#elif defined(__aarch64__) || defined(_M_ARM64)
#  if defined(__GNUC__) && !defined(__clang__)
    // #795: GCC's float trmv and forward substitution at W >= 4.
    if constexpr (std::is_same_v<T, float> &&
                  (std::is_same_v<Kernel, kernel::trmv> || std::is_same_v<Kernel, kernel::lower_trisolve>))
        return 2;
#  endif
    return 4;
#else
    return 6;
#endif
}

/// The default width for T and Kernel, before any override (see the top of
/// this file).
template <typename T, typename Kernel>
consteval std::size_t default_interleave_rows() {
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
        constexpr std::size_t measured = measured_interleave_rows<T, Kernel>();
        constexpr std::size_t cap = vector_registers / 2;   // the rest hold A and x
        return measured < cap ? measured : cap;
    } else {
        return 4;
    }
}

/// The width with the -D overrides applied.
template <typename T, typename Kernel>
consteval std::size_t configured_interleave_rows() {
#if defined(MTL5_INTERLEAVE_ROWS_F32)
    if constexpr (std::is_same_v<T, float>) return MTL5_INTERLEAVE_ROWS_F32;
#endif
#if defined(MTL5_INTERLEAVE_ROWS_F64)
    if constexpr (std::is_same_v<T, double>) return MTL5_INTERLEAVE_ROWS_F64;
#endif
#if defined(MTL5_INTERLEAVE_ROWS)
    return MTL5_INTERLEAVE_ROWS;
#else
    return default_interleave_rows<T, Kernel>();
#endif
}

/// How many rows Kernel interleaves for accumulator type T. Specialize it to
/// set the width for a number type or a single kernel.
template <typename T, typename Kernel>
struct interleave_rows {
    static constexpr std::size_t value = configured_interleave_rows<T, Kernel>();
};

template <typename T, typename Kernel>
inline constexpr std::size_t interleave_rows_v = [] {
    constexpr std::size_t w = interleave_rows<std::remove_cvref_t<T>, Kernel>::value;
    static_assert(w >= 1 && w <= 64, "interleave_rows must be between 1 and 64");
    return w;
}();

} // namespace mtl::interface
