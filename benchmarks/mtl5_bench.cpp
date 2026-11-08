// mtl5_bench -- the platform benchmark (#721).
//
// One binary that characterizes a machine's linear-algebra capability as MTL5
// sees it, the same way on every machine: probe the machine's ceilings (FMA
// peak, triad bandwidth), run every registered operation at a fixed problem
// size, check every result, and report each rate as a percentage of the
// roofline for that operation's arithmetic intensity.
//
//   mtl5_bench [--class small|medium|large] [--json out.json] [--label name]
//              [--filter substr] [--list]
//
// --json writes the results and, beside them, out.json.sysinfo: the machine,
// the build (commit, flags, SIMD target) and the thread count, the same
// provenance keys bench_all writes (#442), so benchmarks/check_sidecars.sh can
// hold a committed result to it.
//
// Exit status: 0 if every check passed, 1 if any check failed, an entry threw,
// or the JSON or its sidecar could not be written; 2 on a usage error. A
// failed check prints FAIL in place of the rate's verdict and still reports
// the timing, so a fast wrong kernel is visible.
//
// The build is the backend (benchmarks/README.md): build this binary once per
// configuration -- native, native-fast, BLAS -- and run each.

#include <benchmarks/platform/entries_blas.hpp>
#include <benchmarks/platform/entries_dense.hpp>
#include <benchmarks/platform/entries_sparse.hpp>
#include <benchmarks/platform/entries_iterative.hpp>
#include <benchmarks/platform/entries_mixed.hpp>
#include <benchmarks/platform/probes.hpp>
#include <benchmarks/platform/registry.hpp>
#include <benchmarks/platform/run.hpp>

#include <mtl/build_info.hpp>
#include <mtl/detail/thread_pool.hpp>
#include <mtl/simd/batch.hpp>
#include <mtl/util/system_info.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace bp = mtl::bench::platform;

namespace {

#if defined(MTL5_HAS_BLAS)
constexpr const char* kDefaultLabel = "blas";
#elif defined(MTL5_NATIVE_FAST_GEMM) && defined(MTL5_HAS_HIGHWAY)
constexpr const char* kDefaultLabel = "native-fast";
#else
constexpr const char* kDefaultLabel = "native";
#endif

void usage(std::ostream& os) {
    os << "usage: mtl5_bench [--class small|medium|large] [--json FILE] [--card FILE]\n"
          "                  [--label NAME] [--machine NAME] [--filter SUBSTR] [--list]\n"
          "  --class    problem sizes (default: medium)\n"
          "  --json     write results to FILE and provenance to FILE.sysinfo\n"
          "  --card     write a markdown capability card to FILE\n"
          "  --label    build label recorded in the output (default: from the build)\n"
          "  --machine  machine name for the card and the records (default: the CPU brand)\n"
          "  --baseline FILE.json from the same build at fewer threads (normally 1):\n"
          "             report each entry's speedup and parallel efficiency against it\n"
          "  --filter   run only entries whose area/op/dtype contains SUBSTR\n"
          "  --list     print the registered entries and exit\n";
}

std::string json_escape(const std::string& s) {
    std::string o;
    for (char c : s) {
        switch (c) {
        case '"':  o += "\\\""; break;
        case '\\': o += "\\\\"; break;
        case '\n': o += "\\n";  break;
        case '\t': o += "\\t";  break;
        default:
            if (static_cast<unsigned char>(c) < 0x20) {
                char buf[8];
                std::snprintf(buf, sizeof buf, "\\u%04x", c);
                o += buf;
            } else {
                o += c;
            }
        }
    }
    return o;
}

/// JSON numbers cannot be inf or nan.
std::string num(double v) {
    if (!(v == v) || v > 1e300 || v < -1e300) return "null";
    std::ostringstream os;
    os << std::setprecision(6) << v;
    return os.str();
}

std::string id_of(const bp::entry& e) { return e.area + "/" + e.op + "/" + e.dtype; }

/// A --baseline run: its thread count and each entry's median, read back from
/// the JSON this program writes (its own format, so a small scanner suffices).
struct baseline {
    unsigned threads = 0;
    std::string cls;   // the size class it ran at: medians compare only within one
    std::map<std::string, double> median;   // "area/op/dtype" -> seconds
};

std::string json_string_after(const std::string& s, const std::string& key, std::size_t& pos) {
    const std::string k = "\"" + key + "\": \"";
    const std::size_t a = s.find(k, pos);
    if (a == std::string::npos) return {};
    const std::size_t b = s.find('"', a + k.size());
    pos = b;
    return s.substr(a + k.size(), b - a - k.size());
}

bool read_baseline(const std::string& path, baseline& out) {
    std::ifstream in(path);
    if (!in) return false;
    std::stringstream ss;
    ss << in.rdbuf();
    const std::string s = ss.str();
    const std::size_t t = s.find("\"threads\": ");
    if (t == std::string::npos) return false;
    out.threads = static_cast<unsigned>(std::strtoul(s.c_str() + t + 11, nullptr, 10));
    std::size_t cp = 0;
    out.cls = json_string_after(s, "class", cp);
    std::size_t pos = s.find("\"results\"");
    while (pos != std::string::npos) {
        const std::size_t start = s.find("{\"area\": \"", pos);
        if (start == std::string::npos) break;
        pos = start;
        const std::string area = json_string_after(s, "area", pos);
        const std::string op = json_string_after(s, "op", pos);
        const std::string dtype = json_string_after(s, "dtype", pos);
        const std::size_t m = s.find("\"median_s\": ", pos);
        if (m == std::string::npos) break;
        const double v = std::strtod(s.c_str() + m + 12, nullptr);   // "null" parses as 0
        if (v > 0.0) out.median[area + "/" + op + "/" + dtype] = v;
        pos = m;
    }
    return out.threads > 0;
}

/// Returns false if the file could not be opened or a write failed.
bool write_json(const std::string& path, const std::string& label, const std::string& machine,
                bp::size_class cls, const bp::probe_results& pr, const std::vector<bp::record>& rs) {
    std::ofstream os(path);
    if (!os) return false;
    os << "{\n"
       << "  \"schema\": \"mtl5-bench/1\",\n"
       << "  \"label\": \"" << json_escape(label) << "\",\n"
       << "  \"machine\": \"" << json_escape(machine) << "\",\n"
       << "  \"class\": \"" << bp::to_string(cls) << "\",\n"
       << "  \"simd_backend\": \"" << json_escape(mtl::simd::backend_name()) << "\",\n"
       << "  \"probes\": {\n"
       << "    \"threads\": " << pr.threads << ",\n"
       << "    \"fma_peak_f64_gflops\": " << num(pr.fma_peak_f64_gflops) << ",\n"
       << "    \"fma_peak_f32_gflops\": " << num(pr.fma_peak_f32_gflops) << ",\n"
       << "    \"build_fma_peak_f64_gflops\": " << num(pr.build_fma_peak_f64_gflops) << ",\n"
       << "    \"build_fma_peak_f32_gflops\": " << num(pr.build_fma_peak_f32_gflops) << ",\n"
       << "    \"hw_fma_isa\": \"" << json_escape(pr.hw_isa) << "\",\n"
       << "    \"hw_fma_peak_f64_gflops\": " << num(pr.hw_fma_peak_f64_gflops) << ",\n"
       << "    \"hw_fma_peak_f32_gflops\": " << num(pr.hw_fma_peak_f32_gflops) << ",\n"
       << "    \"triad_gbytes_per_s\": " << num(pr.triad_gbytes_per_s) << ",\n"
       << "    \"read_gbytes_per_s\": " << num(pr.read_gbytes_per_s) << ",\n"
       << "    \"update_gbytes_per_s\": " << num(pr.update_gbytes_per_s) << ",\n"
       << "    \"bandwidth_gbytes_per_s\": " << num(pr.bandwidth_gbytes_per_s()) << ",\n"
       << "    \"triad_array_bytes\": " << pr.triad_array_bytes << ",\n"
       << "    \"last_level_cache_bytes\": " << bp::last_level_cache_bytes() << "\n"
       << "  },\n"
       << "  \"results\": [\n";
    for (std::size_t i = 0; i < rs.size(); ++i) {
        const auto& r = rs[i];
        os << "    {\"area\": \"" << json_escape(r.area) << "\", \"op\": \"" << json_escape(r.op)
           << "\", \"dtype\": \"" << json_escape(r.dtype) << "\", \"shape\": \"" << json_escape(r.shape)
           << "\",\n     \"median_s\": " << num(r.median_s) << ", \"min_s\": " << num(r.min_s)
           << ", \"samples\": " << r.samples
           << ", \"flops\": " << num(r.flops) << ", \"bytes\": " << num(r.bytes)
           << ",\n     \"gflops\": " << num(r.gflops) << ", \"gbytes_per_s\": " << num(r.gbytes_per_s)
           << ", \"roofline_gflops\": " << num(r.roofline_gflops)
           << ", \"roofline_pct\": " << (r.no_roofline() ? std::string("null") : num(r.roofline_pct))
           << ", \"cache_resident\": " << (r.cache_resident ? "true" : "false")
           << ", \"speedup\": " << num(r.speedup) << ", \"efficiency\": " << num(r.efficiency)
           << ",\n     \"check\": {\"ok\": " << (r.check.ok ? "true" : "false")
           << ", \"error\": " << num(r.check.error) << ", \"tolerance\": " << num(r.check.tolerance)
           << ", \"metric\": \"" << json_escape(r.check.what) << "\"}";
        if (!r.error.empty()) os << ", \"exception\": \"" << json_escape(r.error) << "\"";
        os << "}" << (i + 1 < rs.size() ? "," : "") << "\n";
    }
    os << "  ]\n}\n";
    os.flush();
    return static_cast<bool>(os);
}

/// Returns false if the file could not be opened or a write failed.
bool write_sidecar(const std::string& path, const std::string& label, const std::string& machine,
                   bp::size_class cls, const bp::probe_results& pr) {
    std::ofstream si(path);
    if (!si) return false;
    si << "label=" << label << "\n"
       << "machine=" << machine << "\n"
       << "benchmark=mtl5_bench\n"
       << "class=" << bp::to_string(cls) << "\n"
       << mtl::util::to_keyvals(mtl::util::identify())
       << "simd_backend=" << mtl::simd::backend_name() << "\n"
       << "git_commit=" << mtl::build_git_commit << "\n"
       << "git_dirty=" << mtl::build_git_dirty << "\n"
       << "cxx_flags=" << mtl::build_cxx_flags << "\n"
       << "cmake_build_type=" << mtl::build_cmake_type << "\n"
       << "pool_size=" << pr.threads << "\n";
    si.flush();
    return static_cast<bool>(si);
}

/// A markdown capability card: one run's machine, build, ceilings and results,
/// in the form docs/benchmarks/systems.md publishes. Area rows are grouped and
/// summarized (median and best %roof), so a reader sees where the build reaches
/// the machine and where it does not without reading every row.
bool write_card(const std::string& path, const std::string& label, const std::string& machine,
                bp::size_class cls, const bp::probe_results& pr, const std::vector<bp::record>& rs) {
    std::ofstream md(path);
    if (!md) return false;
    const auto si = mtl::util::identify();
    md << std::fixed;
    md << "## " << machine << " -- " << label << ", " << pr.threads
       << (pr.threads == 1 ? " thread" : " threads") << ", class " << bp::to_string(cls) << "\n\n"
       << "| | |\n|---|---|\n"
       << "| CPU | " << (si.cpu.brand.empty() ? "unknown" : si.cpu.brand) << " (" << si.cpu.arch
       << ", " << si.cpu.logical_cores << " logical cores) |\n"
       << "| OS | " << si.os.name << " " << si.os.version << " |\n"
       << "| compiler | " << si.compiler.name << " " << si.compiler.version << " |\n"
       << "| build | `" << mtl::build_cxx_flags << "`, SIMD " << mtl::simd::backend_name() << " |\n"
       << "| commit | `" << mtl::build_git_commit << "`" << (std::string(mtl::build_git_dirty) == "1" ? " (dirty)" : "")
       << " |\n" << std::setprecision(1)
       << "| FMA peak (ceiling) | f64 " << pr.fma_peak_f64_gflops << " GFLOP/s, f32 " << pr.fma_peak_f32_gflops
       << " GFLOP/s |\n"
       << "| ... hardware, " << pr.hw_isa << " | f64 " << pr.hw_fma_peak_f64_gflops << ", f32 "
       << pr.hw_fma_peak_f32_gflops << " GFLOP/s |\n"
       << "| ... this build, " << mtl::simd::backend_name() << " | f64 " << pr.build_fma_peak_f64_gflops
       << ", f32 " << pr.build_fma_peak_f32_gflops << " GFLOP/s |\n"
       << "| bandwidth | read " << pr.read_gbytes_per_s << ", update " << pr.update_gbytes_per_s
       << ", triad " << pr.triad_gbytes_per_s << " GB/s (ceiling " << pr.bandwidth_gbytes_per_s() << ") |\n"
       << "| last-level cache | " << (bp::last_level_cache_bytes() >> 20) << " MiB |\n\n";

    // Per-area summary.
    bool scaled = false;
    for (const auto& r : rs) scaled |= r.speedup == r.speedup;
    md << "| area | entries | median %roof | best %roof | failed checks |"
       << (scaled ? " median parallel efficiency |\n|---|---:|---:|---:|---:|---:|\n" : "\n|---|---:|---:|---:|---:|\n");
    std::vector<std::string> areas;
    for (const auto& r : rs)
        if (std::find(areas.begin(), areas.end(), r.area) == areas.end()) areas.push_back(r.area);
    for (const auto& a : areas) {
        std::vector<double> pct, eff;
        int failed = 0;
        std::size_t entries = 0;
        for (const auto& r : rs)
            if (r.area == a) {
                if (!r.no_roofline()) pct.push_back(r.roofline_pct);
                if (r.efficiency == r.efficiency) eff.push_back(r.efficiency);
                ++entries;
                failed += !(r.check.ok && r.error.empty());
            }
        std::sort(pct.begin(), pct.end());
        md << "| " << a << " | " << entries << " | " << std::setprecision(0);
        if (pct.empty()) md << "n/a | n/a";
        else md << pct[pct.size() / 2] << "% | " << pct.back() << "%";
        md << " | " << failed << " |";
        if (scaled) {
            std::sort(eff.begin(), eff.end());
            if (eff.empty()) md << " n/a |";
            else md << " " << eff[eff.size() / 2] * 100.0 << "% |";
        }
        md << "\n";
    }

    md << "\n| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |"
       << (scaled ? " speedup (efficiency) |\n|---|---|---|---|---:|---:|---:|---:|---|---:|\n"
                  : "\n|---|---|---|---|---:|---:|---:|---:|---|\n");
    bool any_cached = false;
    for (const auto& r : rs) {
        const bool ok = r.check.ok && r.error.empty();
        any_cached |= r.cache_resident;
        md << "| " << r.area << " | " << r.op << " | " << r.dtype << " | " << r.shape << " | "
           << std::setprecision(3) << r.median_s * 1e3 << " ms | " << std::setprecision(2) << r.gflops
           << " | " << r.gbytes_per_s << " | " << std::setprecision(0);
        if (r.no_roofline()) md << "n/a";
        else md << r.roofline_pct << "%" << (r.cache_resident ? "\\*" : "");
        md << " | " << (ok ? "ok" : "**FAIL**") << " |";
        if (scaled) {
            if (r.speedup == r.speedup)
                md << " " << std::setprecision(1) << r.speedup << "x (" << std::setprecision(0)
                   << r.efficiency * 100.0 << "%) |";
            else md << " n/a |";
        }
        md << "\n";
    }
    if (any_cached)
        md << "\n\\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.\n";
    md << "\n";
    md.flush();
    return static_cast<bool>(md);
}

} // namespace

int main(int argc, char** argv) {
    bp::size_class cls = bp::size_class::medium;
    std::string json_path, card_path, filter, machine, baseline_path, label = kDefaultLabel;
    bool list = false;
    for (int i = 1; i < argc; ++i) {
        auto need = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) { std::cerr << flag << " needs a value\n"; usage(std::cerr); std::exit(2); }
            return argv[++i];
        };
        if (!std::strcmp(argv[i], "--class")) {
            const std::string c = need("--class");
            if (c == "small") cls = bp::size_class::small;
            else if (c == "medium") cls = bp::size_class::medium;
            else if (c == "large") cls = bp::size_class::large;
            else { std::cerr << "unknown class '" << c << "'\n"; usage(std::cerr); return 2; }
        } else if (!std::strcmp(argv[i], "--json"))   json_path = need("--json");
        else if (!std::strcmp(argv[i], "--card"))   card_path = need("--card");
        else if (!std::strcmp(argv[i], "--label"))  label = need("--label");
        else if (!std::strcmp(argv[i], "--machine")) machine = need("--machine");
        else if (!std::strcmp(argv[i], "--baseline")) baseline_path = need("--baseline");
        else if (!std::strcmp(argv[i], "--filter")) filter = need("--filter");
        else if (!std::strcmp(argv[i], "--list"))   list = true;
        else if (!std::strcmp(argv[i], "--help") || !std::strcmp(argv[i], "-h")) { usage(std::cout); return 0; }
        else { std::cerr << "unknown argument '" << argv[i] << "'\n"; usage(std::cerr); return 2; }
    }

    bp::registry reg;
    bp::blas::register_all(reg);
    bp::dense::register_all(reg);
    bp::sparse_entries::register_all(reg);
    bp::iterative::register_all(reg);
    bp::mixed::register_all(reg);

    std::vector<const bp::entry*> selected;
    for (const auto& e : reg.entries())
        if (filter.empty() || id_of(e).find(filter) != std::string::npos) selected.push_back(&e);

    if (list) {
        for (const auto* e : selected) std::cout << id_of(*e) << "\n";
        return 0;
    }
    if (selected.empty()) { std::cerr << "no entry matches '" << filter << "'\n"; return 2; }

    if (machine.empty()) {
        machine = mtl::util::identify().cpu.brand;
        if (machine.empty()) machine = "unknown machine";
    }
    const unsigned threads = static_cast<unsigned>(mtl::detail::thread_pool::instance().size());
    baseline base;
    if (!baseline_path.empty()) {
        if (!read_baseline(baseline_path, base)) {
            std::cerr << "error: could not read a baseline from " << baseline_path << "\n";
            return 2;
        }
        if (base.cls != bp::to_string(cls)) {
            // Different problem sizes: a "speedup" would compare unrelated work.
            std::cerr << "error: baseline " << baseline_path << " is class '" << base.cls << "', this run is '"
                      << bp::to_string(cls) << "'\n";
            return 2;
        }
        if (base.threads == threads) {
            std::cerr << "warning: baseline " << baseline_path << " ran at the same thread count (" << threads
                      << "); scaling is not reported\n";
            base.median.clear();
        }
    }
    std::cout << "mtl5_bench  label=" << label << "  class=" << bp::to_string(cls)
              << "  simd=" << mtl::simd::backend_name() << "  threads=" << threads
              << "  commit=" << mtl::build_git_commit << "\n\n";

    std::cout << "probing the machine..." << std::flush;
    const bp::probe_results pr = bp::run_probes(threads);
    std::cout << "\r" << std::fixed << std::setprecision(1)
              << "probe  fma peak f64 " << std::setw(8) << pr.fma_peak_f64_gflops << " GFLOP/s"
              << "   (hardware " << pr.hw_isa << " " << pr.hw_fma_peak_f64_gflops
              << ", this build " << pr.build_fma_peak_f64_gflops << ")\n"
              << "probe  fma peak f32 " << std::setw(8) << pr.fma_peak_f32_gflops << " GFLOP/s"
              << "   (hardware " << pr.hw_isa << " " << pr.hw_fma_peak_f32_gflops
              << ", this build " << pr.build_fma_peak_f32_gflops << ")\n"
              << "probe  read         " << std::setw(8) << pr.read_gbytes_per_s << " GB/s\n"
              << "probe  update       " << std::setw(8) << pr.update_gbytes_per_s << " GB/s\n"
              << "probe  triad        " << std::setw(8) << pr.triad_gbytes_per_s << " GB/s  ("
              << (pr.triad_array_bytes >> 20) << " MiB arrays)\n\n";

    std::cout << std::left << std::setw(12) << "area" << std::setw(23) << "op" << std::setw(8) << "type"
              << std::setw(30) << "shape" << std::right << std::setw(11) << "median"
              << std::setw(11) << "GFLOP/s" << std::setw(10) << "GB/s" << std::setw(9) << "%roof"
              << "  check\n";

    std::vector<bp::record> results;
    int failures = 0;
    for (const auto* e : selected) {
        bp::record r = bp::run_entry(*e, cls, pr);
        if (const auto it = base.median.find(id_of(*e)); it != base.median.end() && r.median_s > 0.0) {
            r.speedup = it->second / r.median_s;
            r.efficiency = r.speedup / (double(threads) / double(base.threads));
        }
        const bool ok = r.check.ok && r.error.empty();
        failures += !ok;
        std::ostringstream t;
        t << std::fixed << std::setprecision(r.median_s < 1e-3 ? 1 : 2)
          << (r.median_s < 1e-3 ? r.median_s * 1e6 : r.median_s * 1e3) << (r.median_s < 1e-3 ? "us" : "ms");
        std::cout << std::left << std::setw(12) << r.area << std::setw(23) << r.op << std::setw(8) << r.dtype
                  << std::setw(30) << r.shape << std::right << std::setw(11) << t.str()
                  << std::fixed << std::setprecision(2) << std::setw(11) << r.gflops
                  << std::setw(10) << r.gbytes_per_s << std::setprecision(0) << std::setw(8);
        if (r.no_roofline()) std::cout << "n/a" << "   ";
        else std::cout << r.roofline_pct << (r.cache_resident ? "%* " : "%  ");
        std::cout << (ok ? "ok" : "FAIL");
        if (r.speedup == r.speedup)
            std::cout << std::setprecision(1) << "  x" << r.speedup << std::setprecision(0) << " ("
                      << r.efficiency * 100.0 << "% eff.)";
        if (!ok) {
            std::cout << "  (" << (r.error.empty() ? r.check.what : r.error) << ": "
                      << std::scientific << std::setprecision(2) << r.check.error << " > "
                      << r.check.tolerance << ")" << std::fixed;
        }
        std::cout << "\n" << std::flush;
        results.push_back(std::move(r));
    }

    bool any_cached = false;
    for (const auto& r : results) any_cached |= r.cache_resident;
    if (any_cached)
        std::cout << "\n* working set fits in the last-level cache ("
                  << (bp::last_level_cache_bytes() >> 20)
                  << " MiB): a cache rate, shown as a % of the DRAM roofline\n";

    if (!json_path.empty()) {
        // A result without its sidecar is the provenance loss check_sidecars.sh
        // exists to prevent, so a failed write of EITHER file fails the run.
        const bool json_ok = write_json(json_path, label, machine, cls, pr, results);
        const bool side_ok = json_ok && write_sidecar(json_path + ".sysinfo", label, machine, cls, pr);
        if (!json_ok || !side_ok) {
            std::cerr << "\nerror: could not write " << (json_ok ? json_path + ".sysinfo" : json_path) << "\n";
            return 1;
        }
        std::cout << "\nwrote " << json_path << " and " << json_path << ".sysinfo\n";
    }
    if (!card_path.empty()) {
        if (!write_card(card_path, label, machine, cls, pr, results)) {
            std::cerr << "\nerror: could not write " << card_path << "\n";
            return 1;
        }
        std::cout << "wrote " << card_path << "\n";
    }
    std::cout << "\n" << results.size() - failures << " of " << results.size() << " checks passed\n";
    return failures == 0 ? 0 : 1;
}
