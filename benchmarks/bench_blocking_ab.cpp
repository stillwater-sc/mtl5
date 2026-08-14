// MTL5 -- A/B the detected cache blocking against the compile-time defaults.
//
// One question: on this machine, is GEMM faster with kc/mc derived from the
// DETECTED cache hierarchy than with the hardcoded Haswell-class figures MTL5
// ships? Both arms are this same source; they differ only by
// MTL5_ENABLE_CACHE_DETECTION, so nothing but the blocking parameters changes.
// benchmarks/run_blocking_ab.sh builds both and interleaves them.
//
// The answer on an i7-12700K was NO -- detection lost on 9 of 10 points, which
// is why it is opt-in (see simd/blocking.hpp). This harness is how that verdict
// gets revisited on other hardware (#430) rather than assumed either way.
//
// Shapes are NOT hardcoded. The regime where the jc (n-dimension) loop actually
// parallelizes is machine-dependent -- it needs
//
//     nib = ceil(m/mc) <= T/2      (so the ic loop does not consume the budget)
//     njb = ceil(n/nc) >= 2        (so there is more than one jc block)
//
// and both bounds contain this machine's own mc and nc. A fixed shape list
// silently tests nothing on some machines: every measurement in #426 was square,
// which is precisely the regime where jc_nt is structurally 1 and the effects
// being hunted cannot appear. `--suggest-shapes` derives the list from the
// running machine; the script derives it ONCE and passes the identical list to
// both arms, so the two are never measured on different shapes.

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include <mtl/detail/gemm_blocked.hpp>
#include <mtl/detail/thread_pool.hpp>
#include <mtl/simd/blocking.hpp>
#include <mtl/util/cache_info.hpp>
#include <mtl/util/system_info.hpp>

using clk = std::chrono::steady_clock;

namespace {

struct shape { std::size_t m, n, k; };

bool mul_overflows(std::size_t a, std::size_t b) {
    return a != 0 && b > std::numeric_limits<std::size_t>::max() / a;
}

/// Parse "m,n,k;m,n,k". Rejects anything whose buffer products would wrap: a
/// wrapped m*k allocates a short vector while gemm_blocked still walks the
/// original extents, which is an out-of-bounds write rather than a failed run.
/// A shape that is merely too large for RAM is left alone -- that fails loudly
/// as bad_alloc, which is a fine way to find out.
std::vector<shape> parse_shapes(const std::string& spec) {
    std::vector<shape> out;
    std::size_t pos = 0;
    while (pos < spec.size()) {
        std::size_t end = spec.find(';', pos);
        if (end == std::string::npos) end = spec.size();
        const std::string one = spec.substr(pos, end - pos);
        std::size_t m = 0, n = 0, k = 0;
        if (std::sscanf(one.c_str(), "%zu,%zu,%zu", &m, &n, &k) != 3 || !m || !n || !k) {
            std::fprintf(stderr, "bad shape '%s' (want m,n,k, all > 0)\n", one.c_str());
            std::exit(2);
        }
        if (mul_overflows(m, k) || mul_overflows(k, n) || mul_overflows(m, n)) {
            std::fprintf(stderr, "shape %zu,%zu,%zu overflows size_t\n", m, n, k);
            std::exit(2);
        }
        out.push_back({m, n, k});
        pos = end + 1;
    }
    return out;
}

/// Shapes derived from THIS machine's blocking parameters and thread budget:
/// a square set (where jc_nt is normally 1 -- the control), and a wide/short set
/// chosen to put the ic block count under half the budget so the jc loop is the
/// one that parallelizes.
template <typename T>
std::vector<shape> suggest_shapes(unsigned threads) {
    const auto& bp = mtl::simd::runtime_blocking<T>();
    std::vector<shape> s;

    for (std::size_t n : {std::size_t{1024}, std::size_t{2048}, std::size_t{4096}})
        s.push_back({n, n, n});                       // square control

    // Wide/short: m small enough that nib <= T/2, n past a couple of jc blocks.
    const std::size_t m_jc = std::max<std::size_t>(bp.mr, bp.mc * std::max(1u, threads) / 2);
    for (std::size_t mult : {std::size_t{2}, std::size_t{3}}) {
        const std::size_t n_jc = bp.nc * mult;
        if (n_jc == 0) continue;
        s.push_back({m_jc, n_jc, 1024});
    }
    return s;
}

template <typename T>
double run_one(const shape& sh, unsigned nt, int reps, double& checksum) {
    std::vector<T> A(sh.m * sh.k), B(sh.k * sh.n), C(sh.m * sh.n, T(0));
    for (std::size_t i = 0; i < A.size(); ++i) A[i] = T(1) + T(i % 17) * T(0.03125);
    for (std::size_t i = 0; i < B.size(); ++i) B[i] = T(0.5) - T(i % 23) * T(0.015625);

    double best = 1e30;
    for (int r = 0; r < reps; ++r) {
        const auto t0 = clk::now();
        mtl::detail::gemm_blocked<T, T>(
            sh.m, sh.n, sh.k, T(1),
            A.data(), static_cast<std::ptrdiff_t>(sh.k), 1,
            B.data(), static_cast<std::ptrdiff_t>(sh.n), 1,
            T(0), C.data(), sh.n, nt);
        const auto t1 = clk::now();
        best = std::min(best, std::chrono::duration<double>(t1 - t0).count());
    }
    double sum = 0.0;                       // control: both arms must agree exactly
    for (std::size_t i = 0; i < C.size(); i += 4097) sum += static_cast<double>(C[i]);
    checksum = sum;
    return best;
}

void usage() {
    std::printf(
        "Usage: bench_blocking_ab [options]\n"
        "  --label <name>      arm label recorded in the CSV (e.g. detected|default)\n"
        "  --csv <file>        append results here (header written if new)\n"
        "  --threads <list>    comma list, default \"1\"\n"
        "  --reps <n>          repetitions per point, min reported (default 5)\n"
        "  --dtype <double|float>\n"
        "  --shapes <m,n,k;..> explicit shapes; both arms MUST be given the same list\n"
        "  --suggest-shapes    print shapes derived from this machine and exit\n");
}

} // namespace

int main(int argc, char** argv) {
    std::string label = "unlabelled", csv, shapes_spec, dtype = "double";
    std::vector<unsigned> threads{1};
    int reps = 5;
    bool suggest = false;

    for (int i = 1; i < argc; ++i) {
        const auto need = [&](const char* f) -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", f); std::exit(2); }
            return argv[++i];
        };
        if      (!std::strcmp(argv[i], "--label"))   label = need("--label");
        else if (!std::strcmp(argv[i], "--csv"))     csv = need("--csv");
        else if (!std::strcmp(argv[i], "--reps"))    reps = std::atoi(need("--reps").c_str());
        else if (!std::strcmp(argv[i], "--dtype"))   dtype = need("--dtype");
        else if (!std::strcmp(argv[i], "--shapes"))  shapes_spec = need("--shapes");
        else if (!std::strcmp(argv[i], "--suggest-shapes")) suggest = true;
        else if (!std::strcmp(argv[i], "--threads")) {
            threads.clear();
            const std::string list = need("--threads");
            std::size_t p = 0;
            while (p < list.size()) {
                std::size_t e = list.find(',', p);
                if (e == std::string::npos) e = list.size();
                threads.push_back(static_cast<unsigned>(std::strtoul(list.substr(p, e - p).c_str(), nullptr, 10)));
                p = e + 1;
            }
        }
        else if (!std::strcmp(argv[i], "--help")) { usage(); return 0; }
        else { std::fprintf(stderr, "unknown option: %s\n", argv[i]); usage(); return 2; }
    }

    // Validate before anything reaches a measurement. Each of these otherwise
    // fails silently in a way that still writes a plausible-looking CSV row:
    // an empty --threads makes max_element dereference an empty range; --reps 0
    // leaves the 1e30 sentinel as the "time"; an unrecognised --dtype runs the
    // double path while labelling the rows with whatever was asked for.
    if (threads.empty()) {
        std::fprintf(stderr, "--threads: needs at least one count\n"); return 2;
    }
    for (unsigned t : threads)
        if (t == 0) { std::fprintf(stderr, "--threads: counts must be > 0\n"); return 2; }
    if (reps <= 0) { std::fprintf(stderr, "--reps must be > 0\n"); return 2; }
    if (dtype != "double" && dtype != "float") {
        std::fprintf(stderr, "--dtype must be double or float (got '%s')\n", dtype.c_str());
        return 2;
    }

    const unsigned tmax = *std::max_element(threads.begin(), threads.end());
    if (suggest) {
        const auto s = (dtype == "float") ? suggest_shapes<float>(tmax)
                                          : suggest_shapes<double>(tmax);
        for (std::size_t i = 0; i < s.size(); ++i)
            std::printf("%s%zu,%zu,%zu", i ? ";" : "", s[i].m, s[i].n, s[i].k);
        std::printf("\n");
        return 0;
    }

    const auto shapes = shapes_spec.empty()
        ? ((dtype == "float") ? suggest_shapes<float>(tmax) : suggest_shapes<double>(tmax))
        : parse_shapes(shapes_spec);
    if (shapes.empty()) { std::fprintf(stderr, "no shapes\n"); return 2; }

    // Header: what this arm actually compiled to, so a CSV can be audited later.
    const auto& bpd = mtl::simd::runtime_blocking<double>();
    const auto& bpf = mtl::simd::runtime_blocking<float>();
    const auto  ci  = mtl::util::detect_caches();
    std::fprintf(stderr, "arm=%s dtype=%s\n", label.c_str(), dtype.c_str());
    std::fprintf(stderr, "  detected l1d=%zu l2=%zu l3=%zu line=%zu\n",
                 ci.l1d_bytes, ci.l2_bytes, ci.l3_bytes, ci.line_bytes);
#if defined(MTL5_ENABLE_CACHE_DETECTION)
    std::fprintf(stderr, "  MTL5_ENABLE_CACHE_DETECTION is ON -- blocking follows the detected caches\n");
#else
    std::fprintf(stderr, "  detection not enabled -- blocking uses the compile-time defaults (shipped)\n");
#endif
    std::fprintf(stderr, "  fp64 mr=%zu nr=%zu kc=%zu mc=%zu nc=%zu\n",
                 bpd.mr, bpd.nr, bpd.kc, bpd.mc, bpd.nc);
    std::fprintf(stderr, "  fp32 mr=%zu nr=%zu kc=%zu mc=%zu nc=%zu\n",
                 bpf.mr, bpf.nr, bpf.kc, bpf.mc, bpf.nc);

    std::vector<std::string> rows;
    for (unsigned nt : threads) {
        for (const auto& sh : shapes) {
            double chk = 0.0;
            const double secs = (dtype == "float") ? run_one<float>(sh, nt, reps, chk)
                                                   : run_one<double>(sh, nt, reps, chk);
            const auto& bp = (dtype == "float") ? bpf : bpd;
            const double gf = 2.0 * double(sh.m) * double(sh.n) * double(sh.k) / secs / 1e9;
            char buf[512];
            // `threads` is what was ASKED for; `pool` is what the process can
            // actually use. thread_pool clamps MTL5_NUM_THREADS to
            // hardware_concurrency, so on a machine with fewer cores than the
            // request the two differ -- and every grid calculation depends on the
            // second. Recording only the request cost a wrong analysis of the
            // Jetson run, where 8 was asked for on a 6-core part and the shortfall
            // was visible only as a suspiciously low speedup.
            std::snprintf(buf, sizeof buf,
                "%s,%s,%zu,%zu,%zu,%u,%u,%zu,%zu,%zu,%zu,%zu,%d,%.6f,%.3f,%.6f",
                label.c_str(), dtype.c_str(), sh.m, sh.n, sh.k, nt,
                mtl::detail::thread_pool::instance().size(),
                bp.mr, bp.nr, bp.kc, bp.mc, bp.nc, reps, secs, gf, chk);
            rows.emplace_back(buf);
            std::fprintf(stderr, "  m=%zu n=%zu k=%zu T=%u (pool %u)  %.4f s  %.2f GFLOP/s\n",
                         sh.m, sh.n, sh.k, nt,
                         mtl::detail::thread_pool::instance().size(), secs, gf);
        }
    }

    if (csv.empty()) {
        for (const auto& r : rows) std::printf("%s\n", r.c_str());
        return 0;
    }
    const bool fresh = !std::ifstream(csv).good();
    std::ofstream out(csv, std::ios::app);
    if (!out) { std::fprintf(stderr, "cannot write %s\n", csv.c_str()); return 1; }
    if (fresh)
        out << "arm,dtype,m,n,k,threads,pool,mr,nr,kc,mc,nc,reps,min_s,gflops,checksum\n";
    for (const auto& r : rows) out << r << "\n";

    // Sidecar, matching the convention bench_all uses, plus the cache figures --
    // a result is not interpretable without the hierarchy it was blocked for.
    if (std::ofstream si{csv + ".sysinfo"}) {
        si << "label=" << label << "\n"
           << mtl::util::to_keyvals(mtl::util::identify())
           << "cache_l1d_bytes="  << ci.l1d_bytes  << "\n"
           << "cache_l1d_assoc="  << ci.l1d_assoc  << "\n"
           << "cache_l2_bytes="   << ci.l2_bytes   << "\n"
           << "cache_l3_bytes="   << ci.l3_bytes   << "\n"
           << "cache_line_bytes=" << ci.line_bytes << "\n";
    }
    return 0;
}
