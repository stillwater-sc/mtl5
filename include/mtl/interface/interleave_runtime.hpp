#pragma once
// MTL5 -- choosing the interleave width at run time (#778)
//
// interface::interleave_rows fixes the width of the generic gemv, trmv and
// forward substitution at compile time, per ISA class (#772, #780). A binary
// built for a portable target runs on CPUs whose fastest width differs: an
// x86-64 build defaults to 8, where the Ivy Bridge Xeon wants 4 and GCC on the
// i7-12700K wants 12 for double. With MTL5_INTERLEAVE_RUNTIME defined, float
// and double compile each kernel at every width in runtime_interleave_widths
// and pick one on first use, from the table below.
//
// Every width gives the same bits (interleave_rows.hpp), so the choice can
// only change the time, never a result.
//
// The table is the #780 measurements (docs/benchmarks/interleave-width.md):
// for each CPU it lists, the fastest width of each kernel and type, per
// compiler, in the build those were measured in. That was the generic build:
// no -march, so SSE2 on x86-64 and NEON on AArch64. Hence:
//   - An x86-64 build with AVX or more uses no entry. Those widths were not
//     measured, and the compile-time default stands.
//   - A CPU, or a compiler, the table doesn't list gets the compile-time
//     default, so it is never slower than a build without run-time selection.
//   - A CPU is identified by its core, not its exact model: the entries cover
//     the measured core's sibling models (Alder Lake's desktop and mobile
//     parts, say). Each case below names the machine it was measured on.
//   - On a hybrid CPU the entry is for the big cores, which is where the
//     sweeps ran. On Linux on AArch64, the first core /proc/cpuinfo lists
//     that the table knows is the one used (the Ventuno's A78C, not its A55).
//
// force_interleave_rows(w) makes every run-time-selected kernel use width w
// (one of runtime_interleave_widths), and force_interleave_rows(0) undoes it.
// It is for tests and benchmarks: it changes which instantiation runs, never
// the result, and it is a relaxed atomic, so a kernel already running keeps
// the width it started with.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include <mtl/util/cpuid.hpp>

#if defined(__linux__) && defined(__aarch64__)
#  include <cstdlib>
#  include <fstream>
#  include <string>
#endif

namespace mtl::interface {

/// The widths run-time selection compiles each kernel at.
inline constexpr std::size_t runtime_interleave_widths[] = {2, 4, 6, 8, 12};

/// Whether w is one of runtime_interleave_widths.
constexpr bool is_runtime_interleave_width(std::size_t w) noexcept {
    for (std::size_t c : runtime_interleave_widths)
        if (c == w) return true;
    return false;
}

/// The cores the table knows.
enum class interleave_cpu : unsigned char {
    unknown,
    alder_lake,     ///< Intel family 6, model 0x97/0x9A: Golden Cove P-cores (i7-12700K)
    ivy_bridge,     ///< Intel family 6, model 0x3A/0x3E (Xeon E5-2420 v2)
    zen4,           ///< AMD family 0x19, model 0x60-0x7F: client Zen 4 (Ryzen 7 8845HS, Ryzen 9 8945HS)
    cortex_a78ae,   ///< Arm part 0xd42 (Jetson Orin)
    cortex_a78c,    ///< Arm part 0xd4b (Arduino Ventuno Q)
};

constexpr const char* to_string(interleave_cpu c) noexcept {
    switch (c) {
    case interleave_cpu::alder_lake:   return "alder-lake";
    case interleave_cpu::ivy_bridge:   return "ivy-bridge";
    case interleave_cpu::zen4:         return "zen4";
    case interleave_cpu::cortex_a78ae: return "cortex-a78ae";
    case interleave_cpu::cortex_a78c:  return "cortex-a78c";
    default:                           return "unknown";
    }
}

/// A table entry: width[type][kernel], type 0 = double and 1 = float, kernel
/// 0 = gemv, 1 = trmv and 2 = the forward substitution. 0 means none: the
/// compile-time default.
struct interleave_choice {
    std::uint8_t width[2][3];
};

/// The measured widths for cpu under this compiler and build (see the top of
/// this file). All zero where nothing was measured.
constexpr interleave_choice interleave_table(interleave_cpu cpu) noexcept {
    using c = interleave_cpu;
    // Each row: {double: gemv, trmv, trisolve}, {float: gemv, trmv, trisolve}.
#if (defined(__x86_64__) || defined(_M_X64)) && defined(__AVX__)
    static_cast<void>(cpu);
    return {};   // only the SSE2 build was measured
#elif defined(__GNUC__) && !defined(__clang__)
    // GCC 15 on x86-64, GCC 11 (Jetson) and 13 (Ventuno) on AArch64.
    switch (cpu) {
    case c::alder_lake:   return {{{12, 12, 12}, {6, 8, 8}}};
    case c::ivy_bridge:   return {{{4, 6, 4}, {8, 4, 4}}};
    // f32 trmv and trisolve: 4 since #795 (Jetson 6.52 and 5.88 ms at 4, 7.31
    // and 7.07 at 2); the rest from #780.
    case c::cortex_a78ae: return {{{4, 2, 2}, {4, 4, 4}}};
    case c::cortex_a78c:  return {{{4, 4, 4}, {4, 4, 4}}};
    default:              return {};
    }
#elif defined(__clang__) && !defined(_MSC_VER)
    // Clang 21 on x86-64, Clang 18 on the Ventuno. The Jetson has no Clang.
    switch (cpu) {
    case c::alder_lake:   return {{{8, 8, 8}, {6, 6, 6}}};
    case c::ivy_bridge:   return {{{4, 4, 4}, {6, 6, 6}}};
    case c::cortex_a78c:  return {{{4, 4, 4}, {4, 4, 4}}};
    default:              return {};
    }
#elif defined(_MSC_VER)
    // MSVC 19.52 and clang-cl 22. clang-cl's f64 trisolve on the Ryzen 7 was
    // 0.2% faster at 6; 8 everywhere else.
    switch (cpu) {
    case c::zen4:         return {{{8, 8, 8}, {8, 8, 8}}};
    default:              return {};
    }
#else
    static_cast<void>(cpu);
    return {};
#endif
}

/// The core this thread's CPU is, as far as the table goes.
inline interleave_cpu detect_interleave_cpu() noexcept {
#if MTL5_HAS_X86_CPUID && (defined(__x86_64__) || defined(_M_X64))
    unsigned r[4];
    util::cpuidex(0, 0, r);
    if (r[0] < 1) return interleave_cpu::unknown;
    const bool intel = r[1] == 0x756e6547u && r[3] == 0x49656e69u && r[2] == 0x6c65746eu;   // GenuineIntel
    const bool amd   = r[1] == 0x68747541u && r[3] == 0x69746e65u && r[2] == 0x444d4163u;   // AuthenticAMD
    util::cpuidex(1, 0, r);
    unsigned family = (r[0] >> 8) & 0xFu;
    unsigned model = (r[0] >> 4) & 0xFu;
    if (family == 0xFu) family += (r[0] >> 20) & 0xFFu;
    if (family == 6u || family >= 0xFu) model |= ((r[0] >> 16) & 0xFu) << 4;
    if (intel && family == 6u) {
        if (model == 0x97u || model == 0x9Au) return interleave_cpu::alder_lake;
        if (model == 0x3Au || model == 0x3Eu) return interleave_cpu::ivy_bridge;
    }
    if (amd && family == 0x19u && model >= 0x60u && model <= 0x7Fu) return interleave_cpu::zen4;
    return interleave_cpu::unknown;
#elif defined(__linux__) && defined(__aarch64__)
    // MIDR fields, one per core: "CPU implementer : 0x41", "CPU part : 0xd4b".
    std::ifstream in("/proc/cpuinfo");
    std::string line;
    unsigned long implementer = 0;
    while (std::getline(in, line)) {
        const auto colon = line.find(':');
        if (colon == std::string::npos) continue;
        const unsigned long v = std::strtoul(line.c_str() + colon + 1, nullptr, 0);
        if (line.rfind("CPU implementer", 0) == 0) {
            implementer = v;
        } else if (line.rfind("CPU part", 0) == 0 && implementer == 0x41) {   // Arm Ltd
            if (v == 0xd42) return interleave_cpu::cortex_a78ae;
            if (v == 0xd4b) return interleave_cpu::cortex_a78c;
        }
    }
    return interleave_cpu::unknown;
#else
    return interleave_cpu::unknown;
#endif
}

/// The core detected on first use, cached.
inline interleave_cpu runtime_interleave_cpu() noexcept {
    static const interleave_cpu cpu = detect_interleave_cpu();
    return cpu;
}

namespace detail {

inline std::atomic<std::size_t>& forced_interleave_rows() noexcept {
    static std::atomic<std::size_t> w{0};
    return w;
}

inline const interleave_choice& runtime_interleave_choice() noexcept {
    static const interleave_choice choice = interleave_table(runtime_interleave_cpu());
    return choice;
}

} // namespace detail

/// Make every run-time-selected kernel use width w; 0 goes back to the table.
inline void force_interleave_rows(std::size_t w) {
    if (w != 0 && !is_runtime_interleave_width(w))
        throw std::invalid_argument("force_interleave_rows: not one of runtime_interleave_widths");
    detail::forced_interleave_rows().store(w, std::memory_order_relaxed);
}

/// The width run-time selection picks for type (0 = double, 1 = float) and
/// kernel (0 = gemv, 1 = trmv, 2 = the forward substitution): the forced one,
/// else the table's, else 0 for the compile-time default.
inline std::size_t runtime_interleave_width(int type, int kernel) noexcept {
    if (const std::size_t f = detail::forced_interleave_rows().load(std::memory_order_relaxed)) return f;
    return detail::runtime_interleave_choice().width[type][kernel];
}

} // namespace mtl::interface
