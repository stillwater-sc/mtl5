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
//     scalar loads a row apart. That costs more than it saves: with it off,
//     f32 gemv runs about 25% faster at every W >= 4. Which W suffers least
//     differs between compilers and -march levels.
//
// So the default for float and double is measured, not derived: 6, capped at
// half the vector registers the target has. On an i7-12700K (n = 4096, one
// core), 6 matched or beat the pre-#772 width of 4 for gemv, trmv and the
// forward substitution, in f32 and f64: row-major under GCC (plain and
// -march=native) and Clang, column-major under GCC. Row-major gemv under GCC
// took 3.70 against 3.80 ms in f64 and 3.03 against 3.70 ms in f32. 8 was
// faster still for f64, but 13% slower for f32 gemv under GCC -march=native.
// The default is 4 for every other type: long double (x87 has eight stack
// slots), integers, and software number types, whose cost only the user knows.
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

/// The default width for T, before any override (see the top of this file).
template <typename T>
consteval std::size_t default_interleave_rows() {
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
        constexpr std::size_t measured = 6;
        constexpr std::size_t cap = vector_registers / 2;   // the rest hold A and x
        return measured < cap ? measured : cap;
    } else {
        return 4;
    }
}

/// The width with the -D overrides applied.
template <typename T>
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
    return default_interleave_rows<T>();
#endif
}

/// How many rows Kernel interleaves for accumulator type T. Specialize it to
/// set the width for a number type or a single kernel.
template <typename T, typename Kernel>
struct interleave_rows {
    static constexpr std::size_t value = configured_interleave_rows<T>();
};

template <typename T, typename Kernel>
inline constexpr std::size_t interleave_rows_v = [] {
    constexpr std::size_t w = interleave_rows<std::remove_cvref_t<T>, Kernel>::value;
    static_assert(w >= 1 && w <= 64, "interleave_rows must be between 1 and 64");
    return w;
}();

} // namespace mtl::interface
