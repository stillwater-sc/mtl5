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
//     GCC's float trmv and forward substitution were held at 2 until #795:
//     the keep_scalar barrier (#776) made them 5.6-7.2x slower at 4. Without
//     it on AArch64 (detail/scalar_chain.hpp), 4 is their fastest width too
//     (Ventuno Q: trmv 3.12 ms at 4 against 4.71 at 2).
//   - Other targets: 6, the i7-12700K figure that was the default for every
//     target before (#772).
// Each is capped at half the vector registers the target has: 32-bit x86 and
// unrecognized targets get 4. The default is 4 for every other type: long
// double (x87 has eight stack slots), integers, and software number types,
// whose cost only the user knows.
//
// Run-time selection (#778, interleave_runtime.hpp) recovers the rest: the
// Xeon's 4 and GCC's 12 on the i7, which a portable x86-64 binary can't know
// at compile time. It is off unless MTL5_INTERLEAVE_RUNTIME is defined (the
// CMake option of that name passes it), because it compiles each kernel at
// five widths. Any override below turns it off for the types it covers.
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

#if defined(MTL5_INTERLEAVE_RUNTIME)
#  include <mtl/interface/interleave_runtime.hpp>
#endif

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
    /// Marks the primary template: a specialization lacks it, and so turns
    /// run-time selection off for its type and kernel (#778).
    static constexpr bool is_measured_default = true;
};

template <typename T, typename Kernel>
inline constexpr std::size_t interleave_rows_v = [] {
    constexpr std::size_t w = interleave_rows<std::remove_cvref_t<T>, Kernel>::value;
    static_assert(w >= 1 && w <= 64, "interleave_rows must be between 1 and 64");
    return w;
}();

/// Whether Kernel picks its width for T at run time (#778): only with
/// MTL5_INTERLEAVE_RUNTIME, on x86-64 or AArch64, for float and double, and
/// only where neither a -D override nor a specialization fixed the width.
template <typename T, typename Kernel>
consteval bool interleave_rows_at_runtime() {
#if defined(MTL5_INTERLEAVE_RUNTIME) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__aarch64__) || defined(_M_ARM64))
    using U = std::remove_cvref_t<T>;
    if constexpr (!requires { interleave_rows<U, Kernel>::is_measured_default; }) {
        return false;
    } else if constexpr (std::is_same_v<U, float>) {
#  if defined(MTL5_INTERLEAVE_ROWS) || defined(MTL5_INTERLEAVE_ROWS_F32)
        return false;
#  else
        return true;
#  endif
    } else if constexpr (std::is_same_v<U, double>) {
#  if defined(MTL5_INTERLEAVE_ROWS) || defined(MTL5_INTERLEAVE_ROWS_F64)
        return false;
#  else
        return true;
#  endif
    } else {
        return false;
    }
#else
    return false;
#endif
}

template <typename T, typename Kernel>
inline constexpr bool interleave_rows_at_runtime_v = interleave_rows_at_runtime<T, Kernel>();

namespace detail {

template <typename Kernel>
consteval int interleave_kernel_index() {
    if constexpr (std::is_same_v<Kernel, kernel::gemv>) return 0;
    else if constexpr (std::is_same_v<Kernel, kernel::trmv>) return 1;
    else return 2;
}

} // namespace detail

/// The width Kernel uses for T in this process: the run-time choice where
/// there is one, else interleave_rows_v.
template <typename T, typename Kernel>
std::size_t selected_interleave_rows() noexcept {
#if defined(MTL5_INTERLEAVE_RUNTIME)
    if constexpr (interleave_rows_at_runtime_v<T, Kernel>) {
        constexpr int type = std::is_same_v<std::remove_cvref_t<T>, float> ? 1 : 0;
        if (const std::size_t w = runtime_interleave_width(type, detail::interleave_kernel_index<Kernel>()))
            return w;
    }
#endif
    return interleave_rows_v<T, Kernel>;
}

/// Call f(std::integral_constant<std::size_t, W>{}) with the width Kernel uses
/// for T: interleave_rows_v, or with run-time selection, the width chosen for
/// this CPU, dispatched over runtime_interleave_widths.
template <typename T, typename Kernel, typename F>
decltype(auto) with_interleave_rows(F&& f) {
    constexpr std::size_t W = interleave_rows_v<T, Kernel>;
#if defined(MTL5_INTERLEAVE_RUNTIME)
    if constexpr (interleave_rows_at_runtime_v<T, Kernel>) {
        switch (selected_interleave_rows<T, Kernel>()) {
        case 2:  return f(std::integral_constant<std::size_t, 2>{});
        case 4:  return f(std::integral_constant<std::size_t, 4>{});
        case 6:  return f(std::integral_constant<std::size_t, 6>{});
        case 8:  return f(std::integral_constant<std::size_t, 8>{});
        case 12: return f(std::integral_constant<std::size_t, 12>{});
        default: break;
        }
    }
#endif
    return f(std::integral_constant<std::size_t, W>{});
}

} // namespace mtl::interface
