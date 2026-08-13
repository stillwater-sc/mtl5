#pragma once
// MTL5 -- runtime cache-hierarchy detection for the GEMM blocking model (#222).
//
// simd::derive_blocking sizes the cache blocks kc/mc/nc from a hw_traits
// description whose figures were hardcoded to a Haswell-class core (32 KB L1 /
// 256 KB L2 / 8 MB L3). This reads the real hierarchy at runtime so those blocks
// follow the machine the binary is running on rather than the machine the model
// was written on.
//
// SCOPE. Only the quantities derive_blocking actually consumes are detected:
// L1d, L2, L3 and the line size (plus L1 associativity, which comes free from
// the same CPUID registers). The FMA latency/issue width and the vector register
// file are deliberately NOT detected -- they select the mr x nr register tile,
// which is a compile-time template argument of the micro-kernel, so a runtime
// value could not be applied to the compiled kernel anyway. Page size is not
// detected because derive_blocking does not read it.
//
// This header is reachable from the core (blocking.hpp -> gemm), so unlike
// system_info.hpp it deliberately does NOT include <windows.h>: the core must
// not force the Win32 headers on every translation unit. That is a coverage gap
// rather than a compromise -- on x86 the CPUID path covers every OS, Windows
// included, and the one uncovered configuration is non-x86 Windows (ARM64),
// which reports "unknown" and therefore keeps the compile-time defaults.
//
// Every field is best effort. 0 means NOT DETECTED and callers must fall back;
// it never means "a cache of size zero".

#include <cstddef>

#include <mtl/util/cpuid.hpp>

#if !MTL5_HAS_X86_CPUID
#  if defined(__APPLE__)
#    include <sys/sysctl.h>
#  elif !defined(_WIN32)
#    include <cstdlib>
#    include <fstream>
#    include <string>
#  endif
#endif

namespace mtl::util {

/// Detected data-cache hierarchy, in bytes. 0 = not detected.
struct cache_info {
    std::size_t l1d_bytes  = 0;   ///< per-core L1 DATA cache (never the I-cache)
    std::size_t l1d_assoc  = 0;   ///< L1d ways
    std::size_t l2_bytes   = 0;   ///< per-core L2
    std::size_t l3_bytes   = 0;   ///< shared L3 (0 is legitimate: many cores have none)
    std::size_t line_bytes = 0;   ///< cache line
};

namespace detail {

#if MTL5_HAS_X86_CPUID

/// CPUID deterministic-cache-parameters walk. Intel uses leaf 4; AMD publishes
/// the identical encoding at extended leaf 0x8000001D. Sub-leaves enumerate the
/// caches until the type field reads 0 (null).
inline void fill_caches_x86(cache_info& c) {
    unsigned regs[4];

    cpuidex(0, 0, regs);
    const unsigned max_leaf = regs[0];
    char vendor[13] = {};
    // EBX, EDX, ECX -- the order the vendor string is returned in.
    vendor[0]  = static_cast<char>(regs[1] & 0xff);
    vendor[1]  = static_cast<char>((regs[1] >> 8) & 0xff);
    vendor[2]  = static_cast<char>((regs[1] >> 16) & 0xff);
    vendor[3]  = static_cast<char>((regs[1] >> 24) & 0xff);
    vendor[4]  = static_cast<char>(regs[3] & 0xff);
    vendor[5]  = static_cast<char>((regs[3] >> 8) & 0xff);
    vendor[6]  = static_cast<char>((regs[3] >> 16) & 0xff);
    vendor[7]  = static_cast<char>((regs[3] >> 24) & 0xff);
    vendor[8]  = static_cast<char>(regs[2] & 0xff);
    vendor[9]  = static_cast<char>((regs[2] >> 8) & 0xff);
    vendor[10] = static_cast<char>((regs[2] >> 16) & 0xff);
    vendor[11] = static_cast<char>((regs[2] >> 24) & 0xff);

    bool is_amd = vendor[0] == 'A' && vendor[1] == 'u' && vendor[2] == 't';  // AuthenticAMD
    // Older AMD parts return zeros for leaf 4 and only populate 0x8000001D.
    int leaf = 4;
    if (is_amd) {
        cpuidex(static_cast<int>(0x80000000u), 0, regs);
        if (regs[0] >= 0x8000001Du) leaf = static_cast<int>(0x8000001Du);
    }
    if (leaf == 4 && max_leaf < 4) return;   // no deterministic-cache leaf at all

    for (int i = 0; i < 32; ++i) {
        cpuidex(leaf, i, regs);
        const unsigned type = regs[0] & 0x1fu;
        if (type == 0) break;                         // 0 = null: end of enumeration
        const unsigned level = (regs[0] >> 5) & 0x7u;
        const std::size_t line  = static_cast<std::size_t>(regs[1] & 0xfffu) + 1;
        const std::size_t parts = static_cast<std::size_t>((regs[1] >> 12) & 0x3ffu) + 1;
        const std::size_t ways  = static_cast<std::size_t>((regs[1] >> 22) & 0x3ffu) + 1;
        const std::size_t sets  = static_cast<std::size_t>(regs[2]) + 1;
        const std::size_t size  = ways * parts * line * sets;

        if (c.line_bytes == 0) c.line_bytes = line;
        // type 1 = data, 2 = instruction, 3 = unified. The I-cache must never be
        // mistaken for L1d: they are the same size on most parts, so the error
        // would be invisible in the numbers.
        if (level == 1 && type == 1) { c.l1d_bytes = size; c.l1d_assoc = ways; }
        else if (level == 2 && type != 2) { c.l2_bytes = size; }
        else if (level == 3 && type != 2) { c.l3_bytes = size; }
    }
}

#elif defined(__APPLE__)

inline std::size_t sysctl_size(const char* name) {
    std::size_t v = 0, len = sizeof(v);
    if (::sysctlbyname(name, &v, &len, nullptr, 0) == 0) return v;
    return 0;
}

/// Apple Silicon reports per-performance-level caches; hw.l*cachesize is the
/// Intel-era spelling and is absent or zero on M-series.
inline void fill_caches_apple(cache_info& c) {
    c.l1d_bytes = sysctl_size("hw.perflevel0.l1dcachesize");
    if (c.l1d_bytes == 0) c.l1d_bytes = sysctl_size("hw.l1dcachesize");
    c.l2_bytes = sysctl_size("hw.perflevel0.l2cachesize");
    if (c.l2_bytes == 0) c.l2_bytes = sysctl_size("hw.l2cachesize");
    c.l3_bytes    = sysctl_size("hw.l3cachesize");   // typically absent on M-series
    c.line_bytes  = sysctl_size("hw.cachelinesize");
}

#elif !defined(_WIN32)

/// Read a whole sysfs attribute; empty on failure.
inline std::string read_sysfs(const std::string& path) {
    std::ifstream in(path);
    std::string   s;
    if (in) std::getline(in, s);
    return s;
}

/// Parse the kernel's cache size spelling: a decimal with an optional K/M/G.
inline std::size_t parse_size(const std::string& s) {
    if (s.empty()) return 0;
    char*             end = nullptr;
    const unsigned long long v = std::strtoull(s.c_str(), &end, 10);
    if (end == s.c_str()) return 0;
    std::size_t mult = 1;
    if (end && *end) {
        switch (*end) {
            case 'K': case 'k': mult = 1024; break;
            case 'M': case 'm': mult = 1024 * 1024; break;
            case 'G': case 'g': mult = 1024 * 1024 * 1024; break;
            default: break;
        }
    }
    return static_cast<std::size_t>(v) * mult;
}

/// Linux sysfs walk, for non-x86 (AArch64, RISC-V, POWER) where there is no
/// CPUID. cpu0 is representative for the purposes of the blocking model; on a
/// heterogeneous core layout it describes whichever core cpu0 is.
inline void fill_caches_sysfs(cache_info& c) {
    const std::string base = "/sys/devices/system/cpu/cpu0/cache/index";
    for (int i = 0; i < 10; ++i) {
        const std::string dir = base + std::to_string(i);
        const std::string lvl = read_sysfs(dir + "/level");
        if (lvl.empty()) break;                       // no more index<N> entries
        const std::string type = read_sysfs(dir + "/type");
        const std::size_t size = parse_size(read_sysfs(dir + "/size"));
        const std::size_t line = parse_size(read_sysfs(dir + "/coherency_line_size"));
        const int level = std::atoi(lvl.c_str());

        if (c.line_bytes == 0 && line != 0) c.line_bytes = line;
        const bool is_instruction = (type == "Instruction");
        if (level == 1 && type == "Data") {
            c.l1d_bytes = size;
            c.l1d_assoc = parse_size(read_sysfs(dir + "/ways_of_associativity"));
        } else if (level == 2 && !is_instruction) {
            c.l2_bytes = size;
        } else if (level == 3 && !is_instruction) {
            c.l3_bytes = size;
        }
    }
}

#endif

} // namespace detail

/// Detect the data-cache hierarchy. Best effort: any field that could not be
/// determined stays 0, and callers fall back rather than believing the zero.
inline cache_info detect_caches() {
    cache_info c;
#if MTL5_HAS_X86_CPUID
    detail::fill_caches_x86(c);
#elif defined(__APPLE__)
    detail::fill_caches_apple(c);
#elif !defined(_WIN32)
    detail::fill_caches_sysfs(c);
#endif
    // Non-x86 Windows falls through with everything 0 -- see the header note.
    return c;
}

/// Detect once per process and reuse. The hierarchy cannot change under a
/// running binary, and the CPUID walk / sysfs reads should not sit in any hot
/// path. Thread-safe initialization is guaranteed by the static local.
inline const cache_info& cached_cache_info() {
    static const cache_info c = detect_caches();
    return c;
}

} // namespace mtl::util
