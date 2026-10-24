// MTL5 Benchmark -- a vendor BLAS called from inside an MTL5 region (#640)
//
// A batch of independent n x n dgemms, split across the tids of one MTL5
// region, with the vendor library set (by its environment variable) to a team
// of its own. Arms:
//
//   uncapped  each tid calls the vendor's dgemm_ directly: every tid starts a
//             vendor team, pool-size x vendor-size threads on the cores (what
//             MTL5 did before #640)
//   capped    each tid calls mtl::interface::blas::gemm, which runs the vendor
//             on one thread inside a region (MKL, BLIS; OpenBLAS is unchanged)
//   process1  OpenBLAS only: the process-wide openblas_set_num_threads(1),
//             then as `uncapped` -- the configuration the docs give OpenBLAS
//   direct    the same batch from the calling thread, outside a region: the
//             vendor's team is the only parallelism
//
//   ddot      one ddot of length 256 per call inside a region, direct and
//             through the wrapper: what the cap costs a small call
//
// Run each arm in a process of its own (--arms): a vendor's idle team keeps
// spinning after its last call (Intel OpenMP for 200 ms by default), and an arm
// that followed `uncapped` in the same process would be timed against it.
//
// Usage (one vendor per build tree, MTL5_WITH_BLAS=ON):
//   MTL5_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 bench_vendor_nested [--csv f]
//       [--arms uncapped,capped,process1,direct,ddot] [--sizes 128,256,512]
//       [--batch 64] [--reps 7]
// The CSV is appended to, with a header when it is new.
// benchmarks/run_vendor_nested.sh builds the three vendor trees and runs them.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <mtl/parallel/thread_pool.hpp>
#include <mtl/util/system_info.hpp>
#include <mtl/build_info.hpp>

#ifdef MTL5_HAS_BLAS
#include <mtl/interface/blas.hpp>
#include <mtl/interface/vendor_threads.hpp>

#if defined(MTL5_BLAS_VENDOR_OPENBLAS)
extern "C" void openblas_set_num_threads(int);
#endif

namespace {

using clk = std::chrono::steady_clock;
namespace iface = mtl::interface;

struct batch {
    int n;
    std::vector<std::vector<double>> A, B, C;
    batch(int n_, int count) : n(n_) {
        const std::size_t e = static_cast<std::size_t>(n) * n;
        for (int b = 0; b < count; ++b) {
            A.emplace_back(e); B.emplace_back(e); C.emplace_back(e, 0.0);
            for (std::size_t i = 0; i < e; ++i) {
                A.back()[i] = static_cast<double>((i * 7 + b) % 11) - 5.0;
                B.back()[i] = static_cast<double>((i * 5 + b) % 13) - 6.0;
            }
        }
    }
    void raw(int b) {
        const char t = 'N'; const double one = 1.0, zero = 0.0;
        dgemm_(&t, &t, &n, &n, &n, &one, A[b].data(), &n, B[b].data(), &n,
               &zero, C[b].data(), &n);
    }
    void wrapped(int b) {
        iface::blas::gemm('N', 'N', n, n, n, 1.0, A[b].data(), n, B[b].data(), n,
                          0.0, C[b].data(), n);
    }
    double checksum() const {
        double s = 0.0;
        for (const auto& c : C) for (double v : c) s += v;
        return s;
    }
};

template <typename F>
double median_seconds(int reps, F&& f) {
    f();                                    // warm-up: first touch, vendor pool start
    std::vector<double> t;
    for (int r = 0; r < reps; ++r) {
        const auto t0 = clk::now();
        f();
        t.push_back(std::chrono::duration<double>(clk::now() - t0).count());
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

std::vector<std::string> parse_names(const char* s) {
    std::vector<std::string> v;
    std::string cur;
    for (const char* p = s;; ++p) {
        if (*p == ',' || *p == '\0') { if (!cur.empty()) v.push_back(cur); cur.clear(); }
        else cur += *p;
        if (*p == '\0') break;
    }
    return v;
}

std::vector<int> parse_list(const char* s) {
    std::vector<int> v;
    for (const char* p = s; *p;) {
        v.push_back(std::atoi(p));
        while (*p && *p != ',') ++p;
        if (*p == ',') ++p;
    }
    return v;
}

const char* vendor_env() {
    for (const char* name : {"MKL_NUM_THREADS", "BLIS_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                             "OMP_NUM_THREADS"})
        if (const char* v = std::getenv(name)) { static std::string s; s = std::string(name) + "=" + v; return s.c_str(); }
    return "(vendor default)";
}

} // namespace

int main(int argc, char** argv) {
    std::vector<int> sizes{128, 256, 512};
    int count = 64, reps = 7;
    std::string csv;
    std::vector<std::string> arms{"uncapped", "capped", "process1", "direct", "ddot"};
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--sizes") && i + 1 < argc) sizes = parse_list(argv[++i]);
        else if (!std::strcmp(argv[i], "--batch") && i + 1 < argc) count = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--reps") && i + 1 < argc) reps = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--csv") && i + 1 < argc) csv = argv[++i];
        else if (!std::strcmp(argv[i], "--arms") && i + 1 < argc) arms = parse_names(argv[++i]);
    }
    auto want = [&](const char* a) { return std::find(arms.begin(), arms.end(), a) != arms.end(); };

    auto& pool = mtl::parallel::pool();
    const unsigned P = pool.size();
    const std::string vendor = iface::blas_vendor()[0] ? iface::blas_vendor() : "unknown";
    std::printf("vendor %s, %s, MTL5 pool %u, vendor threads on the caller %d, "
                "capped in regions: %s\n", vendor.c_str(), vendor_env(), P,
                iface::blas_vendor_threads(), iface::blas_capped_in_region() ? "yes" : "no");

    std::ofstream out;
    if (!csv.empty()) {
        const bool fresh = !std::ifstream(csv).good();
        out.open(csv, std::ios::app);
        if (fresh) out << "vendor,pool,vendor_env,arm,n,batch,seconds,gflops,checksum\n";
    }
    auto row = [&](const char* arm, int n, double s, double sum) {
        const double gf = 2.0 * n * n * static_cast<double>(n) * count / s * 1e-9;
        std::printf("  %-9s n=%-5d %9.4f s %8.1f GFLOP/s  checksum %.6e\n", arm, n, s, gf, sum);
        if (out) out << vendor << ',' << P << ',' << vendor_env() << ',' << arm << ','
                     << n << ',' << count << ',' << s << ',' << gf << ',' << sum << '\n';
    };

    for (int n : sizes) {
        batch bt(n, count);
        // tid t runs batch entries t, t+P, ...: a static split, so every arm
        // gives each tid the same entries.
        auto region = [&](bool wrapped) {
            pool.run(P, [&](unsigned tid) {
                for (int b = static_cast<int>(tid); b < count; b += static_cast<int>(P))
                    wrapped ? bt.wrapped(b) : bt.raw(b);
            });
        };
        // The time first, then the checksum: as two arguments of one call
        // their order is unspecified, and GCC reads C before the arm writes it.
        auto arm = [&](const char* name, auto&& f) {
            if (!want(name)) return;
            const double s = median_seconds(reps, f);
            row(name, n, s, bt.checksum());
        };
        arm("uncapped", [&] { region(false); });
        arm("capped",   [&] { region(true); });
        arm("direct",   [&] { for (int b = 0; b < count; ++b) bt.wrapped(b); });
#if defined(MTL5_BLAS_VENDOR_OPENBLAS)
        if (want("process1")) {
            const int keep = openblas_get_num_threads();
            openblas_set_num_threads(1);
            arm("process1", [&] { region(false); });
            openblas_set_num_threads(keep);
        }
#endif
    }

    // What the cap costs one small call made inside a region.
    if (want("ddot")) {
        const int n = 256, calls = 200000;
        std::vector<double> x(n, 0.5), y(n, 0.25);
        std::vector<double> sink(P, 0.0);
        auto per_call = [&](bool wrapped) {
            const double s = median_seconds(reps, [&] {
                pool.run(P, [&](unsigned tid) {
                    const int one = 1;
                    double acc = 0.0;
                    for (int c = 0; c < calls; ++c)
                        acc += wrapped ? iface::blas::dot(n, x.data(), 1, y.data(), 1)
                                       : ddot_(&n, x.data(), &one, y.data(), &one);
                    sink[tid] += acc;
                });
            });
            return s / calls * 1e9;
        };
        const double raw = per_call(false), wrapped = per_call(true);
        std::printf("  ddot n=%d inside a region: raw %.1f ns/call, wrapped %.1f ns/call\n",
                    n, raw, wrapped);
        if (out) {
            out << vendor << ',' << P << ',' << vendor_env() << ",ddot_raw_ns," << n << ','
                << calls << ',' << raw << ",0,0\n";
            out << vendor << ',' << P << ',' << vendor_env() << ",ddot_wrapped_ns," << n << ','
                << calls << ',' << wrapped << ",0,0\n";
        }
    }
    // Provenance (#442): the machine, the build, and which vendor library and
    // thread settings the numbers are for.
    if (!csv.empty()) {
        if (std::ofstream si{csv + ".sysinfo"}) {
            si << "label=vendor-nested-" << vendor << "\n"
               << mtl::util::to_keyvals(mtl::util::identify())
               << "git_commit="  << mtl::build_git_commit  << "\n"
               << "git_dirty="   << mtl::build_git_dirty   << "\n"
               << "cxx_flags="   << mtl::build_cxx_flags   << "\n"
               << "cmake_build_type=" << mtl::build_cmake_type << "\n"
               << "pool_size="   << P << "\n"
               << "blas_vendor=" << vendor << "\n"
               << "vendor_env="  << vendor_env() << "\n"
               << "vendor_threads_caller=" << iface::blas_vendor_threads() << "\n"
               << "capped_in_region=" << (iface::blas_capped_in_region() ? 1 : 0) << "\n"
               << "batch=" << count << "\n"
               << "reps=" << reps << "\n";
            for (const char* name : {"MKL_NUM_THREADS", "BLIS_NUM_THREADS",
                                     "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                                     "MTL5_NUM_THREADS", "BENCH_PCPUS"})
                if (const char* v = std::getenv(name)) si << "env_" << name << '=' << v << "\n";
        }
    }
    return 0;
}

#else
int main() {
    std::puts("bench_vendor_nested needs a build with -DMTL5_WITH_BLAS=ON");
    return 0;
}
#endif
