#pragma once
// MTL5 -- System identification for benchmark tagging
//
// Self-identifies the machine a binary is running on so every benchmark result
// can be tagged with the processor, OS, and compiler it was produced by. This
// is what lets docs/benchmarks/systems.md entries be generated rather than
// hand-authored.
//
// Header-only and self-contained. It is deliberately NOT included by mtl.hpp:
// it pulls in <windows.h> / <sys/utsname.h> / <sys/sysctl.h> on the respective
// platforms, which the core library must not force on every translation unit.
// Include it directly where you need it (benchmark drivers, diagnostics).
//
// Portability: CPU brand/vendor/feature flags use CPUID on x86-64. On other
// ISAs (e.g. AArch64) CPUID does not exist, so the brand is read from the OS
// (/proc/cpuinfo on Linux, sysctl on macOS, the registry on Windows) and the
// x86 feature flags are simply reported as unavailable.

#include <cstdint>
#include <cstring>
#include <string>
#include <thread>

#include <mtl/util/cpuid.hpp>   // MTL5_HAS_X86_CPUID, util::cpuidex

// ---------------------------------------------------------------------------
// Architecture detection (compile time)
// ---------------------------------------------------------------------------
// The CPUID wrapper and its <cpuid.h>/<intrin.h> include moved to cpuid.hpp so
// cache_info.hpp can share them (#222); MTL5_SYSINFO_X86 is kept as the name the
// rest of this header uses.
#define MTL5_SYSINFO_X86 MTL5_HAS_X86_CPUID

#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#  if defined(_MSC_VER)
// RegGetValueA lives in Advapi32; auto-link it so header-only consumers on MSVC
// need no extra link flags. Non-MSVC Windows toolchains link -ladvapi32 in CMake.
#    pragma comment(lib, "advapi32")
#  endif
#elif defined(__APPLE__)
#  include <sys/sysctl.h>
#  include <sys/utsname.h>
#else // assume POSIX / Linux
#  include <fstream>
#  include <sys/utsname.h>
#endif

namespace mtl::util {

/// CPU identity and the SIMD feature flags that matter for kernel dispatch.
/// `logical_cores` is the number of hardware threads the OS exposes; physical
/// core / P-E topology is not portably derivable and stays out of scope here.
struct cpu_info {
    std::string  brand;          ///< marketing brand string, e.g. "AMD Ryzen 9 8945HS"
    std::string  vendor;         ///< "GenuineIntel", "AuthenticAMD", or "" if unknown
    std::string  arch;           ///< "x86_64", "aarch64", ...
    unsigned     logical_cores = 0;

    // x86 SIMD features (all false on non-x86, where they are meaningless).
    bool sse2    = false;
    bool avx     = false;
    bool avx2    = false;
    bool fma     = false;
    bool avx512f = false;
};

/// Operating-system name and a best-effort version/build string.
struct os_info {
    std::string name;            ///< "Windows", "Linux", "macOS"
    std::string version;         ///< best-effort, e.g. "10.0.26200" or "Ubuntu 24.04 (6.8.0)"
};

/// Compiler identity, resolved entirely at compile time.
struct compiler_info {
    std::string name;            ///< "MSVC", "GCC", "Clang", "Apple Clang"
    std::string version;         ///< dotted version, e.g. "14.51.36231", "13.3.0"
    long        cpp_standard = 0;///< value of __cplusplus (e.g. 202002)
    std::string build_type;      ///< "Release" (NDEBUG) or "Debug"
};

struct system_info {
    cpu_info      cpu;
    os_info       os;
    compiler_info compiler;
};

namespace detail {

#if MTL5_SYSINFO_X86
// `cpuidex` resolves to mtl::util::cpuidex (cpuid.hpp) by enclosing-namespace
// lookup from mtl::util::detail.
inline void fill_cpu_x86(cpu_info& cpu) {
    unsigned regs[4];

    // Leaf 0: highest basic leaf + vendor string (EBX, EDX, ECX order).
    cpuidex(0, 0, regs);
    const unsigned max_leaf = regs[0];
    char vendor[13] = {};
    std::memcpy(vendor + 0, &regs[1], 4);
    std::memcpy(vendor + 4, &regs[3], 4);
    std::memcpy(vendor + 8, &regs[2], 4);
    cpu.vendor = vendor;

    // Leaf 1: SSE2 (EDX bit 26), AVX (ECX bit 28), FMA (ECX bit 12).
    if (max_leaf >= 1) {
        cpuidex(1, 0, regs);
        cpu.sse2 = (regs[3] & (1u << 26)) != 0;
        cpu.avx  = (regs[2] & (1u << 28)) != 0;
        cpu.fma  = (regs[2] & (1u << 12)) != 0;
    }
    // Leaf 7: AVX2 (EBX bit 5), AVX-512F (EBX bit 16).
    if (max_leaf >= 7) {
        cpuidex(7, 0, regs);
        cpu.avx2    = (regs[1] & (1u << 5))  != 0;
        cpu.avx512f = (regs[1] & (1u << 16)) != 0;
    }

    // Extended leaves 0x80000002..4: 48-char brand string, if supported.
    cpuidex(static_cast<int>(0x80000000u), 0, regs);
    const unsigned max_ext = regs[0];
    if (max_ext >= 0x80000004u) {
        char brand[49] = {};
        for (unsigned i = 0; i < 3; ++i) {
            cpuidex(static_cast<int>(0x80000002u + i), 0, regs);
            std::memcpy(brand + i * 16, regs, 16);
        }
        // Brand strings are frequently padded with leading spaces; trim.
        std::string b = brand;
        const auto first = b.find_first_not_of(' ');
        const auto last  = b.find_last_not_of(' ');
        cpu.brand = (first == std::string::npos) ? std::string{}
                                                 : b.substr(first, last - first + 1);
    }
}
#endif // MTL5_SYSINFO_X86

#if defined(_WIN32)
/// Read a REG_SZ value from HKLM; returns "" if absent.
inline std::string reg_string(const char* subkey, const char* value) {
    char  buf[512];
    DWORD size = sizeof(buf);
    DWORD type = 0;
    LSTATUS s = ::RegGetValueA(HKEY_LOCAL_MACHINE, subkey, value,
                               RRF_RT_REG_SZ, &type, buf, &size);
    if (s == ERROR_SUCCESS && size > 0) return std::string(buf);
    return {};
}

/// Read a REG_DWORD value from HKLM; returns `fallback` if absent.
inline DWORD reg_dword(const char* subkey, const char* value, DWORD fallback = 0) {
    DWORD data = 0;
    DWORD size = sizeof(data);
    LSTATUS s = ::RegGetValueA(HKEY_LOCAL_MACHINE, subkey, value,
                               RRF_RT_REG_DWORD, nullptr, &data, &size);
    return (s == ERROR_SUCCESS) ? data : fallback;
}
#endif

inline void fill_cpu_fallback_brand(cpu_info& cpu) {
    if (!cpu.brand.empty()) return;
#if defined(_WIN32)
    cpu.brand = reg_string(
        "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
        "ProcessorNameString");
#elif defined(__APPLE__)
    char   buf[256];
    size_t len = sizeof(buf);
    if (::sysctlbyname("machdep.cpu.brand_string", buf, &len, nullptr, 0) == 0)
        cpu.brand.assign(buf, len ? len - 1 : 0); // len includes NUL
    if (cpu.brand.empty()) {
        len = sizeof(buf);
        if (::sysctlbyname("hw.model", buf, &len, nullptr, 0) == 0)
            cpu.brand.assign(buf, len ? len - 1 : 0);
    }
#else // Linux / POSIX: parse /proc/cpuinfo
    std::ifstream in("/proc/cpuinfo");
    std::string   line;
    while (std::getline(in, line)) {
        // "model name" on x86, "Hardware"/"Model" on many ARM boards.
        const char* keys[] = {"model name", "Hardware", "Model", "CPU part"};
        for (const char* key : keys) {
            if (line.rfind(key, 0) == 0) {
                const auto colon = line.find(':');
                if (colon != std::string::npos) {
                    auto v = line.substr(colon + 1);
                    const auto f = v.find_first_not_of(" \t");
                    if (f != std::string::npos) { cpu.brand = v.substr(f); return; }
                }
            }
        }
    }
#endif
}

inline os_info detect_os() {
    os_info os;
#if defined(_WIN32)
    os.name = "Windows";
    // The Win32 GetVersionEx path is shimmed for unmanifested apps; read the
    // authoritative version straight from the registry instead. Major/minor are
    // DWORDs (the legacy string "CurrentVersion" is frozen at "6.3" since 8.1),
    // while the build number and marketing name are strings.
    const char* key = "SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion";
    DWORD major = reg_dword(key, "CurrentMajorVersionNumber", 10);
    DWORD minor = reg_dword(key, "CurrentMinorVersionNumber", 0);
    std::string build   = reg_string(key, "CurrentBuildNumber");
    std::string display = reg_string(key, "DisplayVersion");
    os.version = std::to_string(major) + "." + std::to_string(minor);
    if (!build.empty())   os.version += "." + build;
    if (!display.empty()) os.version += " (" + display + ")";
#elif defined(__APPLE__)
    os.name = "macOS";
    char   buf[256];
    size_t len = sizeof(buf);
    if (::sysctlbyname("kern.osproductversion", buf, &len, nullptr, 0) == 0)
        os.version.assign(buf, len ? len - 1 : 0);
    if (os.version.empty()) {
        len = sizeof(buf);
        if (::sysctlbyname("kern.osrelease", buf, &len, nullptr, 0) == 0)
            os.version.assign(buf, len ? len - 1 : 0);
    }
#else // Linux / POSIX
    os.name = "Linux";
    // Prefer PRETTY_NAME from /etc/os-release, append kernel release.
    std::ifstream in("/etc/os-release");
    std::string   line, pretty;
    while (std::getline(in, line)) {
        if (line.rfind("PRETTY_NAME=", 0) == 0) {
            pretty = line.substr(std::string("PRETTY_NAME=").size());
            if (pretty.size() >= 2 && pretty.front() == '"' && pretty.back() == '"')
                pretty = pretty.substr(1, pretty.size() - 2);
            break;
        }
    }
    struct utsname u{};
    std::string kernel = (::uname(&u) == 0) ? u.release : "";
    if (!pretty.empty()) os.version = pretty;
    if (!kernel.empty()) os.version += (os.version.empty() ? "" : " ") + ("(" + kernel + ")");
#endif
    return os;
}

inline compiler_info detect_compiler() {
    compiler_info c;
    // MSVC leaves __cplusplus at 199711L unless /Zc:__cplusplus is passed;
    // _MSVC_LANG always reflects the real -std level, so prefer it there.
#if defined(_MSVC_LANG)
    c.cpp_standard = _MSVC_LANG;
#else
    c.cpp_standard = __cplusplus;
#endif
#if defined(NDEBUG)
    c.build_type = "Release";
#else
    c.build_type = "Debug";
#endif

    // Order matters: Clang and Apple Clang also define __GNUC__, and clang-cl
    // additionally defines _MSC_VER, so test the most specific macros first.
#if defined(__apple_build_version__)
    c.name = "Apple Clang";
    c.version = std::to_string(__clang_major__) + "." +
                std::to_string(__clang_minor__) + "." +
                std::to_string(__clang_patchlevel__);
#elif defined(__clang__)
    c.name = "Clang";
    c.version = std::to_string(__clang_major__) + "." +
                std::to_string(__clang_minor__) + "." +
                std::to_string(__clang_patchlevel__);
#elif defined(_MSC_VER)
    c.name = "MSVC";
    // _MSC_FULL_VER packs the toolset version, e.g. 195136231 -> 19.51.36231.
    const long full = _MSC_FULL_VER;
    const long maj  = full / 10000000;
    const long min  = (full / 100000) % 100;
    const long bld  = full % 100000;
    c.version = std::to_string(maj) + "." + std::to_string(min) + "." + std::to_string(bld);
#elif defined(__GNUC__)
    c.name = "GCC";
    c.version = std::to_string(__GNUC__) + "." +
                std::to_string(__GNUC_MINOR__) + "." +
                std::to_string(__GNUC_PATCHLEVEL__);
#else
    c.name = "unknown";
#endif
    return c;
}

} // namespace detail

/// Detect the current machine's CPU, OS, and compiler identity.
inline system_info identify() {
    system_info si;

#if defined(__x86_64__) || defined(_M_X64)
    si.cpu.arch = "x86_64";
#elif defined(__i386__) || defined(_M_IX86)
    si.cpu.arch = "x86";
#elif defined(__aarch64__) || defined(_M_ARM64)
    si.cpu.arch = "aarch64";
#elif defined(__arm__) || defined(_M_ARM)
    si.cpu.arch = "arm";
#else
    si.cpu.arch = "unknown";
#endif

#if MTL5_SYSINFO_X86
    detail::fill_cpu_x86(si.cpu);
#endif
    detail::fill_cpu_fallback_brand(si.cpu);
    if (si.cpu.brand.empty()) si.cpu.brand = "unknown";

    si.cpu.logical_cores = std::thread::hardware_concurrency();

    si.os       = detail::detect_os();
    si.compiler = detail::detect_compiler();
    return si;
}

/// Space-separated list of the detected x86 SIMD features (empty on non-x86).
inline std::string simd_feature_list(const cpu_info& c) {
    std::string s;
    auto add = [&](bool on, const char* name) {
        if (on) { if (!s.empty()) s += ' '; s += name; }
    };
    add(c.sse2, "SSE2");
    add(c.avx, "AVX");
    add(c.avx2, "AVX2");
    add(c.fma, "FMA");
    add(c.avx512f, "AVX512F");
    return s;
}

/// The ISA the BINARY WAS COMPILED FOR, read from the compiler's own predefined
/// macros (empty if none apply).
///
/// `simd_feature_list` above says what the MACHINE can do; this says what the
/// code is actually allowed to use, and the two differ exactly when an intended
/// flag did not take effect. That is not hypothetical: the Zen 4 A/B in #430 was
/// measured as an AVX2 build when AVX-512 was intended, and nothing in the CSV
/// could say so -- the flags reached the compiler through target_compile_options
/// and CMAKE_CXX_FLAGS_<CONFIG>, so recording CMAKE_CXX_FLAGS alone shows an
/// empty string on a build that is anything but default.
///
/// Asking the compiler what it defined sidesteps every one of those paths: it is
/// the effect, not the intent.
inline std::string build_isa_list() {
    std::string s;
    auto add = [&](const char* name) {
        if (!s.empty()) s += ' ';
        s += name;
    };
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    // x86-64 guarantees SSE2; 32-bit needs _M_IX86_FP >= 2 under MSVC.
  #if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
    add("SSE2");
  #endif
  #if defined(__AVX__)
    add("AVX");
  #endif
  #if defined(__AVX2__)
    add("AVX2");
  #endif
  #if defined(__FMA__) || (defined(_MSC_VER) && defined(__AVX2__))
    // MSVC has no __FMA__; /arch:AVX2 enables FMA, so infer it there.
    add("FMA");
  #endif
  #if defined(__AVX512F__)
    add("AVX512F");
  #endif
#elif defined(__aarch64__) || defined(_M_ARM64)
    add("NEON");   // mandatory on AArch64
  #if defined(__ARM_FEATURE_SVE)
    add("SVE");
  #endif
#endif
    if (s.empty()) s = "baseline";
    return s;
}

/// Human-readable multi-line summary, suitable for a benchmark output header.
inline std::string to_string(const system_info& si) {
    std::string feats = simd_feature_list(si.cpu);
    std::string s;
    s += "CPU:      " + si.cpu.brand;
    if (!si.cpu.vendor.empty()) s += " [" + si.cpu.vendor + "]";
    s += "\n";
    s += "Arch:     " + si.cpu.arch +
         ", " + std::to_string(si.cpu.logical_cores) + " logical cores";
    if (!feats.empty()) s += " (" + feats + ")";
    s += "\n";
    s += "OS:       " + si.os.name + " " + si.os.version + "\n";
    s += "Compiler: " + si.compiler.name + " " + si.compiler.version +
         " (C++" + std::to_string((si.compiler.cpp_standard / 100L) % 100L) +
         ", " + si.compiler.build_type + ")";
    return s;
}

/// Flat, machine-parseable key=value lines -- for embedding in CSV headers or
/// building a tag for docs/benchmarks/systems.md.
inline std::string to_keyvals(const system_info& si) {
    std::string s;
    s += "cpu_brand="   + si.cpu.brand   + "\n";
    s += "cpu_vendor="  + si.cpu.vendor  + "\n";
    s += "cpu_arch="    + si.cpu.arch    + "\n";
    s += "cpu_logical_cores=" + std::to_string(si.cpu.logical_cores) + "\n";
    s += "cpu_simd="    + simd_feature_list(si.cpu) + "\n";
    // What the machine can do (above) and what this binary was built to use
    // (below) are different facts, and only the second explains the numbers.
    s += "build_isa="   + build_isa_list() + "\n";
    s += "os_name="     + si.os.name     + "\n";
    s += "os_version="  + si.os.version  + "\n";
    s += "compiler="    + si.compiler.name + "\n";
    s += "compiler_version=" + si.compiler.version + "\n";
    s += "cpp_standard=" + std::to_string(si.compiler.cpp_standard) + "\n";
    s += "build_type="  + si.compiler.build_type + "\n";
    return s;
}

} // namespace mtl::util
