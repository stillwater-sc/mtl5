// MTL5 Benchmark Suite -- Policy-based backend comparison
// Build with: cmake -B build -DMTL5_BUILD_BENCHMARKS=ON [-DMTL5_WITH_BLAS=ON ...]
// Run:        ./build/benchmarks/bench_all [--csv out.csv] [--sizes ...] [--sweep ...]
//
// When compiled without BLAS/LAPACK, only the native backend is benchmarked.
// When compiled with BLAS/LAPACK enabled, both native and accelerated paths
// are benchmarked side-by-side in the same binary for fair comparison.
//
// Sizes can be given explicitly (--sizes) or generated as a sweep (--sweep).
// The default size set deliberately brackets powers of two with odd and
// 1.5x neighbours so a plain run measures padding / odd-size overhead.

#include <benchmarks/harness/runner.hpp>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Default size sets: a mix of powers of two, their +/-1 neighbours, and 1.5x
// midpoints, so the out-of-the-box run exercises many non-power-of-2 sizes and
// exposes padding / leading-dimension overhead. Override with --sizes/--sweep.
static const std::vector<std::size_t> kDefaultBlasSizes = {
    48, 64, 65, 96, 128, 129, 192, 255, 256, 257, 384, 512, 513, 768, 1024};
static const std::vector<std::size_t> kDefaultLapackSizes = {
    48, 64, 65, 96, 128, 129, 192, 255, 256, 257, 384, 512};

// Parse an explicit comma-separated size list, e.g. "64,100,256".
static std::vector<std::size_t> parse_sizes(const char* arg) {
    std::vector<std::size_t> sizes;
    std::istringstream ss(arg);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (token.empty()) continue;
        sizes.push_back(std::stoull(token));
    }
    if (sizes.empty()) throw std::runtime_error("empty size list");
    return sizes;
}

// Parse a sweep spec and expand it to a sorted, de-duplicated size list.
//   START:STOP:STEP    linear, inclusive    (e.g. 16:1024:16  -> 16,32,...,1024)
//   START:STOP:xFACTOR geometric, inclusive (e.g. 16:1024:x2  -> 16,32,...,1024)
// STEP / FACTOR are deliberately free-form so non-power-of-2 sweeps are easy,
// e.g. 33:1024:97 yields only odd, non-power-of-2 sizes.
static std::vector<std::size_t> parse_sweep(const std::string& spec) {
    std::vector<std::string> parts;
    std::istringstream ss(spec);
    std::string tok;
    while (std::getline(ss, tok, ':')) parts.push_back(tok);
    if (parts.size() != 3)
        throw std::runtime_error("sweep must be START:STOP:STEP or START:STOP:xFACTOR (got '" + spec + "')");

    const std::size_t start = std::stoull(parts[0]);
    const std::size_t stop  = std::stoull(parts[1]);
    if (start == 0) throw std::runtime_error("sweep START must be > 0");
    if (stop < start) throw std::runtime_error("sweep STOP must be >= START");

    std::set<std::size_t> uniq;
    if (!parts[2].empty() && (parts[2][0] == 'x' || parts[2][0] == 'X')) {
        const double factor = std::stod(parts[2].substr(1));
        if (factor <= 1.0) throw std::runtime_error("sweep FACTOR must be > 1");
        double n = static_cast<double>(start);
        while (static_cast<std::size_t>(n) <= stop) {
            uniq.insert(static_cast<std::size_t>(n));
            n *= factor;
        }
    } else {
        const std::size_t step = std::stoull(parts[2]);
        if (step == 0) throw std::runtime_error("sweep STEP must be > 0");
        for (std::size_t n = start; n <= stop; n += step) uniq.insert(n);
    }
    return {uniq.begin(), uniq.end()};
}

// One binary == one backend, fixed by the build flags. With no BLAS/LAPACK the
// public mtl:: ops run the generic C++ path; otherwise they dispatch. The label
// (passed via --label, default below) names this build in the output.
#if defined(MTL5_HAS_BLAS) || defined(MTL5_HAS_LAPACK)
static constexpr const char* kDefaultLabel = "blas";
#else
static constexpr const char* kDefaultLabel = "native";
#endif

static void print_build_info(const std::string& label) {
    std::cout << "MTL5 Benchmark Suite\n";
    std::cout << "====================\n";
    std::cout << "Backend (build): " << label << "  [compiled with:";
#ifdef MTL5_HAS_BLAS
    std::cout << " BLAS";
#endif
#ifdef MTL5_HAS_LAPACK
    std::cout << " LAPACK";
#endif
#ifdef MTL5_HAS_HIGHWAY
    std::cout << " Highway";
#endif
#ifdef MTL5_NATIVE_FAST_GEMM
    std::cout << " native-fast-gemm";
#endif
#if !defined(MTL5_HAS_BLAS) && !defined(MTL5_HAS_LAPACK) && !defined(MTL5_NATIVE_FAST_GEMM)
    std::cout << " generic-only";
#endif
    std::cout << " ]\n\n";
}

static void print_usage() {
    std::cout <<
        "Usage: bench_all [options]\n"
        "  --csv <file>            Write results to CSV\n"
        "  --sizes <n,n,...>       Explicit sizes for all suites\n"
        "  --blas-sizes <n,...>    Explicit sizes for BLAS suites\n"
        "  --lapack-sizes <n,...>  Explicit sizes for LAPACK suites\n"
        "  --sweep <spec>          Generated sizes for all suites\n"
        "  --blas-sweep <spec>     Generated sizes for BLAS suites\n"
        "  --lapack-sweep <spec>   Generated sizes for LAPACK suites\n"
        "  --suite <name>          Suite or group to run (default: all)\n"
        "  --label <name>          Backend label recorded in output\n"
        "                          (default: this build's config)\n"
        "\n"
        "Sweep spec:\n"
        "  START:STOP:STEP         linear, inclusive    (e.g. 16:1024:16)\n"
        "  START:STOP:xFACTOR      geometric, inclusive (e.g. 16:1024:x2)\n"
        "  STEP/FACTOR are free-form, so odd sweeps are easy:\n"
        "    33:1024:97       -> only odd, non-power-of-2 sizes\n"
        "    250:1030:x1.01   -> dense sweep bracketing the 256/512/1024 cliffs\n"
        "\n"
        "Suites: all, blas (=l1+l2+l3), lapack,\n"
        "        l1 (dot+nrm2+axpy+scal), l2 (gemv), l3 (gemm),\n"
        "        dot, nrm2, axpy, scal, gemv, gemm, lu, qr, cholesky, eig\n";
}

int main(int argc, char* argv[]) {
    std::vector<std::size_t> blas_sizes   = kDefaultBlasSizes;
    std::vector<std::size_t> lapack_sizes = kDefaultLapackSizes;
    std::string csv_path;
    std::string suite = "all";
    std::string label = kDefaultLabel;

    try {
        for (int i = 1; i < argc; ++i) {
            auto need_arg = [&](const char* opt) -> const char* {
                if (i + 1 >= argc) throw std::runtime_error(std::string(opt) + " requires an argument");
                return argv[++i];
            };
            if (std::strcmp(argv[i], "--csv") == 0) {
                csv_path = need_arg("--csv");
            } else if (std::strcmp(argv[i], "--sizes") == 0) {
                blas_sizes = lapack_sizes = parse_sizes(need_arg("--sizes"));
            } else if (std::strcmp(argv[i], "--blas-sizes") == 0) {
                blas_sizes = parse_sizes(need_arg("--blas-sizes"));
            } else if (std::strcmp(argv[i], "--lapack-sizes") == 0) {
                lapack_sizes = parse_sizes(need_arg("--lapack-sizes"));
            } else if (std::strcmp(argv[i], "--sweep") == 0) {
                blas_sizes = lapack_sizes = parse_sweep(need_arg("--sweep"));
            } else if (std::strcmp(argv[i], "--blas-sweep") == 0) {
                blas_sizes = parse_sweep(need_arg("--blas-sweep"));
            } else if (std::strcmp(argv[i], "--lapack-sweep") == 0) {
                lapack_sizes = parse_sweep(need_arg("--lapack-sweep"));
            } else if (std::strcmp(argv[i], "--suite") == 0) {
                suite = need_arg("--suite");
            } else if (std::strcmp(argv[i], "--label") == 0) {
                label = need_arg("--label");
            } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
                print_usage();
                return 0;
            } else {
                throw std::runtime_error(std::string("unknown option: ") + argv[i]);
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n\n";
        print_usage();
        return 1;
    }

    print_build_info(label);

    mtl::bench::reporter rep;
    namespace b = mtl::bench;

    if (suite == "all") {
        b::run_all(rep, label, blas_sizes, lapack_sizes);
    } else if (suite == "blas") {
        std::cout << "=== BLAS Level 1 ===" << std::endl;
        b::bench_dot(rep, label, blas_sizes);
        b::bench_nrm2(rep, label, blas_sizes);
        b::bench_axpy(rep, label, blas_sizes);
        b::bench_scal(rep, label, blas_sizes);
        std::cout << "=== BLAS Level 2 ===" << std::endl;
        b::bench_gemv(rep, label, blas_sizes);
        std::cout << "=== BLAS Level 3 ===" << std::endl;
        b::bench_gemm(rep, label, blas_sizes);
    } else if (suite == "l1") {
        std::cout << "=== BLAS Level 1 ===" << std::endl;
        b::bench_dot(rep, label, blas_sizes);
        b::bench_nrm2(rep, label, blas_sizes);
        b::bench_axpy(rep, label, blas_sizes);
        b::bench_scal(rep, label, blas_sizes);
    } else if (suite == "l2") {
        std::cout << "=== BLAS Level 2 ===" << std::endl;
        b::bench_gemv(rep, label, blas_sizes);
    } else if (suite == "l3") {
        std::cout << "=== BLAS Level 3 ===" << std::endl;
        b::bench_gemm(rep, label, blas_sizes);
    } else if (suite == "lapack") {
        std::cout << "=== LAPACK Factorizations ===" << std::endl;
        b::bench_lu(rep, label, lapack_sizes);
        b::bench_qr(rep, label, lapack_sizes);
        b::bench_cholesky(rep, label, lapack_sizes);
        b::bench_eigenvalue(rep, label, lapack_sizes);
    } else if (suite == "dot") {
        b::bench_dot(rep, label, blas_sizes);
    } else if (suite == "nrm2") {
        b::bench_nrm2(rep, label, blas_sizes);
    } else if (suite == "axpy") {
        b::bench_axpy(rep, label, blas_sizes);
    } else if (suite == "scal") {
        b::bench_scal(rep, label, blas_sizes);
    } else if (suite == "gemv") {
        b::bench_gemv(rep, label, blas_sizes);
    } else if (suite == "gemm") {
        b::bench_gemm(rep, label, blas_sizes);
    } else if (suite == "lu") {
        b::bench_lu(rep, label, lapack_sizes);
    } else if (suite == "qr") {
        b::bench_qr(rep, label, lapack_sizes);
    } else if (suite == "cholesky") {
        b::bench_cholesky(rep, label, lapack_sizes);
    } else if (suite == "eig") {
        b::bench_eigenvalue(rep, label, lapack_sizes);
    } else {
        std::cerr << "Unknown suite: " << suite << "\n\n";
        print_usage();
        return 1;
    }

    rep.print_table();

    if (!csv_path.empty()) {
        rep.write_csv(csv_path);
    }

    return 0;
}
