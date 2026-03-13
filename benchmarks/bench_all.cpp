// MTL5 Benchmark Suite -- Policy-based backend comparison
// Build with: cmake --preset dev -DMTL5_BUILD_BENCHMARKS=ON [-DMTL5_ENABLE_BLAS=ON ...]
// Run:        ./build/benchmarks/bench_all [--csv output.csv] [--sizes 64,128,256,512,1024]
//
// When compiled without BLAS/LAPACK, only the native backend is benchmarked.
// When compiled with BLAS/LAPACK enabled, both native and accelerated paths
// are benchmarked side-by-side in the same binary for fair comparison.

#include <benchmarks/harness/runner.hpp>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

static std::vector<std::size_t> parse_sizes(const char* arg) {
    std::vector<std::size_t> sizes;
    std::istringstream ss(arg);
    std::string token;
    while (std::getline(ss, token, ',')) {
        sizes.push_back(std::stoull(token));
    }
    return sizes;
}

static void print_backend_info() {
    std::cout << "MTL5 Benchmark Suite\n";
    std::cout << "====================\n";
    std::cout << "Available backends: native";
#ifdef MTL5_HAS_BLAS
    std::cout << ", blas";
#endif
#ifdef MTL5_HAS_LAPACK
    std::cout << ", lapack";
#endif
#ifdef MTL5_HAS_UMFPACK
    std::cout << ", umfpack";
#endif
    std::cout << "\n\n";
}

int main(int argc, char* argv[]) {
    std::vector<std::size_t> blas_sizes   = {64, 128, 256, 512, 1024};
    std::vector<std::size_t> lapack_sizes = {64, 128, 256, 512};
    std::string csv_path;
    std::string suite = "all";

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--csv") == 0 && i + 1 < argc) {
            csv_path = argv[++i];
        } else if (std::strcmp(argv[i], "--sizes") == 0 && i + 1 < argc) {
            blas_sizes = parse_sizes(argv[++i]);
            lapack_sizes = blas_sizes;
        } else if (std::strcmp(argv[i], "--blas-sizes") == 0 && i + 1 < argc) {
            blas_sizes = parse_sizes(argv[++i]);
        } else if (std::strcmp(argv[i], "--lapack-sizes") == 0 && i + 1 < argc) {
            lapack_sizes = parse_sizes(argv[++i]);
        } else if (std::strcmp(argv[i], "--suite") == 0 && i + 1 < argc) {
            suite = argv[++i];
        } else if (std::strcmp(argv[i], "--help") == 0) {
            std::cout << "Usage: bench_all [options]\n"
                      << "  --csv <file>           Write results to CSV\n"
                      << "  --sizes <n,n,...>       Matrix sizes for all suites\n"
                      << "  --blas-sizes <n,...>    Override sizes for BLAS suites\n"
                      << "  --lapack-sizes <n,...>  Override sizes for LAPACK suites\n"
                      << "  --suite <name>         Run specific suite: all, dot, nrm2,\n"
                      << "                         gemv, gemm, lu, qr, cholesky, eig\n";
            return 0;
        }
    }

    print_backend_info();

    mtl::bench::reporter rep;

    if (suite == "all") {
        mtl::bench::run_all(rep, blas_sizes, lapack_sizes);
    } else if (suite == "dot") {
        mtl::bench::bench_dot(rep, blas_sizes);
    } else if (suite == "nrm2") {
        mtl::bench::bench_nrm2(rep, blas_sizes);
    } else if (suite == "gemv") {
        mtl::bench::bench_gemv(rep, blas_sizes);
    } else if (suite == "gemm") {
        mtl::bench::bench_gemm(rep, blas_sizes);
    } else if (suite == "lu") {
        mtl::bench::bench_lu(rep, lapack_sizes);
    } else if (suite == "qr") {
        mtl::bench::bench_qr(rep, lapack_sizes);
    } else if (suite == "cholesky") {
        mtl::bench::bench_cholesky(rep, lapack_sizes);
    } else if (suite == "eig") {
        mtl::bench::bench_eigenvalue(rep, lapack_sizes);
    } else {
        std::cerr << "Unknown suite: " << suite << '\n';
        return 1;
    }

    rep.print_table();

    if (!csv_path.empty()) {
        rep.write_csv(csv_path);
    }

    return 0;
}
