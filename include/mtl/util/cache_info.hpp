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

#if defined(__linux__)
#  include <cerrno>
#  include <cstdlib>
#  include <fstream>
#  include <sched.h>
#  include <set>
#  include <string>
#  include <utility>
#  include <vector>
#elif defined(__APPLE__)
#  include <sys/sysctl.h>
#endif

namespace mtl::util {

/// Detected data-cache hierarchy, in bytes. 0 = not detected.
///
/// `*_bytes` is the cache's TOTAL size as the hardware reports it, and
/// `*_sharing_cores` is how many distinct PHYSICAL cores share that instance, so
/// the per-core budget a model should use is `bytes / sharing_cores`. The two are
/// reported separately rather than pre-divided: this struct describes the
/// machine, and how to spend a shared cache is a modelling decision that belongs
/// with the model (simd::with_detected_caches makes it).
///
/// Sharing is counted in CORES, not logical CPUs, and the difference is not
/// pedantic. An SMT pair shares its L1d and L2, so counting CPUs would halve
/// both on any hyperthreaded machine -- while the benchmark pinning policy runs
/// one thread per physical core and leaves the sibling idle. Counting cores gives
/// 1 there (no division) and 4 for a 4-core E-cluster sharing one L2, which is
/// the case that actually needs discounting.
struct cache_info {
    std::size_t l1d_bytes         = 0;  ///< L1 DATA cache (never the I-cache)
    std::size_t l1d_assoc         = 0;  ///< L1d ways
    std::size_t l1d_sharing_cores = 0;  ///< physical cores sharing it (0 = unknown)
    std::size_t l2_bytes          = 0;
    std::size_t l2_sharing_cores  = 0;
    std::size_t l3_bytes          = 0;  ///< 0 is legitimate: many parts have none
    std::size_t l3_sharing_cores  = 0;
    std::size_t line_bytes        = 0;
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

#endif // MTL5_HAS_X86_CPUID

#if defined(__linux__)

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

/// Expand a sysfs cpu list ("0-3,8" or "0,6") to the ids it names.
inline std::vector<int> parse_cpu_list(const std::string& s) {
    std::vector<int> out;
    std::size_t pos = 0;
    while (pos < s.size()) {
        std::size_t comma = s.find(',', pos);
        if (comma == std::string::npos) comma = s.size();
        const std::string part = s.substr(pos, comma - pos);
        const std::size_t dash = part.find('-');
        if (dash == std::string::npos) {
            if (!part.empty()) out.push_back(std::atoi(part.c_str()));
        } else {
            const int lo = std::atoi(part.substr(0, dash).c_str());
            const int hi = std::atoi(part.substr(dash + 1).c_str());
            for (int v = lo; v <= hi; ++v) out.push_back(v);
        }
        pos = comma + 1;
    }
    return out;
}

/// How many distinct PHYSICAL cores appear in a cpu list. SMT siblings collapse
/// to one; a 4-core cluster counts 4. Identity is (package, core_id), because
/// core_id is only unique within a package.
inline std::size_t distinct_cores(const std::vector<int>& cpus) {
    std::set<std::pair<int, int>> cores;
    for (int cpu : cpus) {
        const std::string topo =
            "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/";
        const std::string core = read_sysfs(topo + "core_id");
        if (core.empty()) continue;                 // topology unavailable
        const std::string pkg = read_sysfs(topo + "physical_package_id");
        cores.insert({pkg.empty() ? 0 : std::atoi(pkg.c_str()), std::atoi(core.c_str())});
    }
    return cores.size();
}

/// CPUs this process may actually be scheduled on. Under `taskset` that is the
/// pinned set, which is what makes detection agree with where the work will run.
///
/// A plain `cpu_set_t` is fixed at CPU_SETSIZE (1024) logical CPUs, and on a
/// host configured with more `sched_getaffinity` fails with EINVAL. Falling back
/// to cpu0 there would be worse than useless: cpu0 need not even be in the mask,
/// so detection would describe a core the process cannot run on. Grow a
/// dynamically sized set until it fits instead.
inline std::vector<int> allowed_cpus() {
    std::vector<int> out;
#if defined(CPU_ALLOC)
    for (int ncpus = CPU_SETSIZE; ncpus <= (1 << 20); ncpus *= 2) {
        cpu_set_t* set = CPU_ALLOC(ncpus);
        if (set == nullptr) break;
        const std::size_t sz = CPU_ALLOC_SIZE(ncpus);
        CPU_ZERO_S(sz, set);
        errno = 0;
        const bool ok = (::sched_getaffinity(0, sz, set) == 0);
        if (ok)
            for (int i = 0; i < ncpus; ++i)
                if (CPU_ISSET_S(static_cast<std::size_t>(i), sz, set)) out.push_back(i);
        const bool too_small = (!ok && errno == EINVAL);
        CPU_FREE(set);
        if (ok || !too_small) break;                // success, or a real failure
    }
#else
    cpu_set_t set;                                  // libc without CPU_ALLOC
    CPU_ZERO(&set);
    if (::sched_getaffinity(0, sizeof(set), &set) == 0)
        for (int i = 0; i < CPU_SETSIZE; ++i)
            if (CPU_ISSET(i, &set)) out.push_back(i);
#endif
    if (out.empty()) out.push_back(0);              // affinity unavailable: assume cpu0
    return out;
}

/// Linux sysfs walk. Used on EVERY ISA including x86, because it fixes the two
/// defects CPUID has here (#432):
///
///   * DETERMINISM. CPUID describes whichever core the calling thread happens to
///     be on, so on a hybrid part the same binary reports a P-core or an E-core
///     hierarchy run to run. sysfs is per-CPU data read by id, so the answer does
///     not depend on where this thread was scheduled.
///   * SHARING. sysfs publishes `shared_cpu_list` per cache, so a cluster L2
///     shared by four cores can be discounted. CPUID's equivalent field counts
///     logical processors and needs threads-per-core to interpret.
///
/// Scans the cpus this process may run on and keeps, per level, the entry with
/// the SMALLEST per-core budget. Under `taskset` that is the pinned set; run
/// unpinned on a hybrid machine it is the whole machine, and taking the minimum
/// means the blocks fit whichever core the work lands on rather than overflowing
/// the smaller kind. Either way it is a property of the machine and the affinity
/// mask, not of the scheduler's whim.
inline void fill_caches_sysfs(cache_info& c) {
    std::size_t best_l1 = static_cast<std::size_t>(-1);
    std::size_t best_l2 = static_cast<std::size_t>(-1);
    std::size_t best_l3 = static_cast<std::size_t>(-1);

    for (int cpu : allowed_cpus()) {
        const std::string base =
            "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/cache/index";
        for (int i = 0; i < 10; ++i) {
            const std::string dir = base + std::to_string(i);
            const std::string lvl = read_sysfs(dir + "/level");
            if (lvl.empty()) break;                 // no more index<N> entries
            const std::string type = read_sysfs(dir + "/type");
            const std::size_t size = parse_size(read_sysfs(dir + "/size"));
            const std::size_t line = parse_size(read_sysfs(dir + "/coherency_line_size"));
            const int level = std::atoi(lvl.c_str());
            if (size == 0) continue;

            // 0 means the topology could not be read, and it is STORED as 0 --
            // the struct's contract is that 0 is "unknown", and reporting an
            // unreadable topology as "shared by exactly one core" would claim
            // knowledge of a private cache we do not have. Only the local divisor
            // treats it as 1; with_detected_caches makes the same assumption at
            // the point where it actually spends the budget.
            const std::size_t sharers = distinct_cores(parse_cpu_list(read_sysfs(dir + "/shared_cpu_list")));
            const std::size_t per_core = size / (sharers ? sharers : 1);

            if (c.line_bytes == 0 && line != 0) c.line_bytes = line;
            const bool is_instruction = (type == "Instruction");
            if (level == 1 && type == "Data") {
                if (per_core < best_l1) {
                    best_l1 = per_core;
                    c.l1d_bytes = size;
                    c.l1d_sharing_cores = sharers;
                    c.l1d_assoc = parse_size(read_sysfs(dir + "/ways_of_associativity"));
                }
            } else if (level == 2 && !is_instruction) {
                if (per_core < best_l2) {
                    best_l2 = per_core;
                    c.l2_bytes = size;
                    c.l2_sharing_cores = sharers;
                }
            } else if (level == 3 && !is_instruction) {
                if (per_core < best_l3) {
                    best_l3 = per_core;
                    c.l3_bytes = size;
                    c.l3_sharing_cores = sharers;
                }
            }
        }
    }
}

#endif // __linux__

} // namespace detail

/// Detect the data-cache hierarchy. Best effort: any field that could not be
/// determined stays 0, and callers fall back rather than believing the zero.
inline cache_info detect_caches() {
    cache_info c;
#if defined(__linux__)
    // Preferred everywhere on Linux, x86 included: deterministic and it knows
    // about sharing (#432).
    detail::fill_caches_sysfs(c);
#  if MTL5_HAS_X86_CPUID
    // Last resort if sysfs is unavailable (a stripped container). This path is
    // per-CURRENT-CORE and reports no sharing, so on a hybrid machine it is not
    // reproducible -- which is why it is the fallback and not the default.
    if (c.l1d_bytes == 0) detail::fill_caches_x86(c);
#  endif
#elif defined(__APPLE__)
    detail::fill_caches_apple(c);
#elif MTL5_HAS_X86_CPUID
    // Windows / other x86: CPUID only, so the hybrid caveat above applies. The
    // machines this project measures on are all Linux, so the fix lands where it
    // is needed; extending it here needs GetLogicalProcessorInformationEx, which
    // the core must not pull <windows.h> in for.
    detail::fill_caches_x86(c);
#endif
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
