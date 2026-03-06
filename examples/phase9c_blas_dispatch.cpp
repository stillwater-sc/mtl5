// phase9c_blas_dispatch.cpp - BLAS/LAPACK Dispatch Architecture
//
// This example demonstrates:
//   1. How MTL5 optionally dispatches to hardware BLAS/LAPACK
//   2. The compile-time #ifdef pattern for optional acceleration
//   3. Manual GEMM vs BLAS-accelerated GEMM performance comparison
//   4. How DL frameworks use the same dispatch strategy for MKL/cuBLAS
//
// Key insight: A linear algebra library operates at two levels:
//   (a) Generic C++ template code that works everywhere (our default)
//   (b) Vendor-optimized BLAS/LAPACK routines for hot paths (optional)
//
// MTL5 uses compile-time guards:
//   - MTL5_HAS_BLAS   → enables hardware BLAS dispatch
//   - MTL5_HAS_LAPACK → enables hardware LAPACK dispatch
//
// This mirrors exactly how PyTorch, TensorFlow, and NumPy work:
//   - PyTorch: links against MKL/OpenBLAS/cuBLAS at build time
//   - NumPy: numpy.show_config() reveals linked BLAS backend
//   - JAX: dispatches to cuBLAS on GPU, MKL on CPU
//
// Build with BLAS enabled:
//   cmake -B build -DMTL5_ENABLE_BLAS=ON -DMTL5_ENABLE_LAPACK=ON
//   cmake --build build

#include <mtl/mtl.hpp>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <vector>

using namespace mtl;

// Prevent dead-code elimination
template <typename T>
void do_not_optimize(const T& val) {
    volatile auto sink = val;
    (void)sink;
}

// ── Manual GEMM (always available) ─────────────────────────────────────

/// Textbook C = alpha * A * B + beta * C, row-major.
/// This is what MTL5 uses when no BLAS is available.
void manual_gemm(double alpha,
                 const mat::dense2D<double>& A,
                 const mat::dense2D<double>& B,
                 double beta,
                 mat::dense2D<double>& C) {
    auto M = A.num_rows();
    auto K = A.num_cols();
    auto N = B.num_cols();

    // Scale C by beta
    for (std::size_t i = 0; i < M; ++i)
        for (std::size_t j = 0; j < N; ++j)
            C(i, j) *= beta;

    // C += alpha * A * B
    for (std::size_t i = 0; i < M; ++i)
        for (std::size_t k = 0; k < K; ++k) {
            double a_ik = alpha * A(i, k);
            for (std::size_t j = 0; j < N; ++j)
                C(i, j) += a_ik * B(k, j);
        }
}

// ── BLAS GEMM (conditionally available) ────────────────────────────────

#ifdef MTL5_HAS_BLAS

/// Dispatch to hardware dgemm.
/// Assumes row-major layout: BLAS is column-major, so we compute
/// B^T * A^T = (A * B)^T and read the result as row-major.
void blas_gemm(double alpha,
               const mat::dense2D<double>& A,
               const mat::dense2D<double>& B,
               double beta,
               mat::dense2D<double>& C) {
    int M = static_cast<int>(A.num_rows());
    int K = static_cast<int>(A.num_cols());
    int N = static_cast<int>(B.num_cols());

    // Row-major trick: dgemm('N','N', N, M, K, alpha, B, N, A, K, beta, C, N)
    // This computes C^T = alpha * B^T * A^T + beta * C^T in column-major,
    // which is C = alpha * A * B + beta * C in row-major.
    interface::blas::gemm('N', 'N', N, M, K,
                          alpha, B.data(), N,
                          A.data(), K,
                          beta, C.data(), N);
}

#endif // MTL5_HAS_BLAS

int main() {
    std::cout << "=============================================================\n";
    std::cout << " Phase 9C: BLAS/LAPACK Dispatch Architecture\n";
    std::cout << "=============================================================\n\n";

    // ══════════════════════════════════════════════════════════════════════
    // 1. Compile-Time Detection
    // ══════════════════════════════════════════════════════════════════════
    std::cout << "=== 1. Build Configuration ===\n\n";

#ifdef MTL5_HAS_BLAS
    std::cout << "  BLAS:   ENABLED (hardware-accelerated GEMM, dot, nrm2, ...)\n";
#else
    std::cout << "  BLAS:   disabled (using generic C++ implementation)\n";
    std::cout << "          To enable: cmake -DMTL5_ENABLE_BLAS=ON\n";
#endif

#ifdef MTL5_HAS_LAPACK
    std::cout << "  LAPACK: ENABLED (hardware-accelerated LU, QR, SVD, ...)\n";
#else
    std::cout << "  LAPACK: disabled (using generic C++ implementation)\n";
    std::cout << "          To enable: cmake -DMTL5_ENABLE_LAPACK=ON\n";
#endif
    std::cout << "\n";

    // ══════════════════════════════════════════════════════════════════════
    // 2. The Dispatch Pattern
    // ══════════════════════════════════════════════════════════════════════
    std::cout << "=== 2. The Dispatch Pattern ===\n\n";

    std::cout << "MTL5 uses the same compile-time dispatch strategy as every\n";
    std::cout << "major numerical framework:\n\n";

    std::cout << "  +------------------+------------------+--------------------+\n";
    std::cout << "  | Framework        | Guard Macro      | BLAS Backend       |\n";
    std::cout << "  +------------------+------------------+--------------------+\n";
    std::cout << "  | MTL5             | MTL5_HAS_BLAS    | OpenBLAS/MKL       |\n";
    std::cout << "  | NumPy/SciPy      | auto-detected    | OpenBLAS/MKL       |\n";
    std::cout << "  | PyTorch (CPU)    | USE_BLAS=1       | MKL/OpenBLAS       |\n";
    std::cout << "  | PyTorch (GPU)    | USE_CUDA=1       | cuBLAS             |\n";
    std::cout << "  | TensorFlow       | auto-detected    | Eigen/MKL/cuBLAS   |\n";
    std::cout << "  | JAX              | jaxlib backend   | MKL/cuBLAS/rocBLAS |\n";
    std::cout << "  +------------------+------------------+--------------------+\n\n";

    std::cout << "The pattern in code:\n\n";
    std::cout << "  // In mtl/interface/blas.hpp:\n";
    std::cout << "  #ifdef MTL5_HAS_BLAS\n";
    std::cout << "  extern \"C\" {\n";
    std::cout << "      void dgemm_(const char*, const char*, const int*, ...);\n";
    std::cout << "  }\n";
    std::cout << "  namespace mtl::interface::blas {\n";
    std::cout << "      void gemm(char transa, char transb, int m, int n, int k,\n";
    std::cout << "                double alpha, const double* A, int lda,\n";
    std::cout << "                const double* B, int ldb,\n";
    std::cout << "                double beta, double* C, int ldc);\n";
    std::cout << "  }\n";
    std::cout << "  #endif\n\n";

    std::cout << "Key design decisions:\n";
    std::cout << "  - Fortran name mangling: dgemm_ (trailing underscore)\n";
    std::cout << "  - Column-major convention: BLAS assumes Fortran layout\n";
    std::cout << "  - Pass-by-pointer: Fortran ABI requires pointer args\n";
    std::cout << "  - C++ wrappers: type-safe overloads for float/double\n\n";

    // ══════════════════════════════════════════════════════════════════════
    // 3. Performance Comparison
    // ══════════════════════════════════════════════════════════════════════
    std::cout << "=== 3. Performance: Manual C++ vs BLAS ===\n\n";

    std::vector<std::size_t> sizes = {64, 128, 256, 512};
    const int warmup = 1;
    const int trials = 3;

    std::cout << std::setw(6)  << "N"
              << std::setw(14) << "Manual(ms)"
#ifdef MTL5_HAS_BLAS
              << std::setw(14) << "BLAS(ms)"
              << std::setw(10) << "Speedup"
#endif
              << std::setw(12) << "GFLOP/s"
              << "\n";
    std::cout << std::string(
#ifdef MTL5_HAS_BLAS
        56
#else
        32
#endif
        , '-') << "\n";

    for (auto N : sizes) {
        mat::dense2D<double> A(N, N), B(N, N), C(N, N);
        for (std::size_t i = 0; i < N; ++i)
            for (std::size_t j = 0; j < N; ++j) {
                A(i, j) = 1.0 + 0.001 * static_cast<double>(i * N + j);
                B(i, j) = 2.0 - 0.001 * static_cast<double>(i * N + j);
                C(i, j) = 0.0;
            }

        double flops = 2.0 * static_cast<double>(N) * static_cast<double>(N)
                          * static_cast<double>(N);

        // Warmup
        for (int w = 0; w < warmup; ++w) {
            manual_gemm(1.0, A, B, 0.0, C);
            do_not_optimize(C(0, 0));
        }

        // Time manual GEMM
        double manual_ms = 0.0;
        for (int t = 0; t < trials; ++t) {
            for (std::size_t i = 0; i < N; ++i)
                for (std::size_t j = 0; j < N; ++j)
                    C(i, j) = 0.0;
            auto t0 = std::chrono::steady_clock::now();
            manual_gemm(1.0, A, B, 0.0, C);
            auto t1 = std::chrono::steady_clock::now();
            do_not_optimize(C(N/2, N/2));
            manual_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        manual_ms /= trials;
        double manual_gflops = flops / (manual_ms * 1e6);

#ifdef MTL5_HAS_BLAS
        // Warmup BLAS
        for (int w = 0; w < warmup; ++w) {
            blas_gemm(1.0, A, B, 0.0, C);
            do_not_optimize(C(0, 0));
        }

        // Time BLAS GEMM
        double blas_ms = 0.0;
        for (int t = 0; t < trials; ++t) {
            for (std::size_t i = 0; i < N; ++i)
                for (std::size_t j = 0; j < N; ++j)
                    C(i, j) = 0.0;
            auto t0 = std::chrono::steady_clock::now();
            blas_gemm(1.0, A, B, 0.0, C);
            auto t1 = std::chrono::steady_clock::now();
            do_not_optimize(C(N/2, N/2));
            blas_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        blas_ms /= trials;
        double blas_gflops = flops / (blas_ms * 1e6);
#endif

        std::cout << std::setw(6)  << N
                  << std::setw(14) << std::fixed << std::setprecision(2) << manual_ms
#ifdef MTL5_HAS_BLAS
                  << std::setw(14) << std::fixed << std::setprecision(2) << blas_ms
                  << std::setw(9)  << std::fixed << std::setprecision(0)
                  << (manual_ms / blas_ms) << "x"
#endif
                  << std::setw(12) << std::fixed << std::setprecision(1)
#ifdef MTL5_HAS_BLAS
                  << blas_gflops
#else
                  << manual_gflops
#endif
                  << "\n";
    }
    std::cout << "\n";

#ifndef MTL5_HAS_BLAS
    std::cout << "  (BLAS not enabled - showing manual C++ GFLOP/s only)\n";
    std::cout << "  Typical BLAS speedup: 10-50x over naive C++ loops\n";
    std::cout << "  MKL on modern x86: ~50-200 GFLOP/s for large DGEMM\n\n";
#endif

    // ══════════════════════════════════════════════════════════════════════
    // 4. LAPACK Integration
    // ══════════════════════════════════════════════════════════════════════
    std::cout << "=== 4. LAPACK Integration ===\n\n";

    std::cout << "MTL5 wraps the core LAPACK routines:\n\n";
    std::cout << "  +--------------------+------------+---------------------------+\n";
    std::cout << "  | MTL5 Operation     | LAPACK     | Use Case                  |\n";
    std::cout << "  +--------------------+------------+---------------------------+\n";
    std::cout << "  | lu()               | dgetrf_    | General linear systems    |\n";
    std::cout << "  | cholesky()         | dpotrf_    | SPD systems (fast)        |\n";
    std::cout << "  | qr()               | dgeqrf_    | Least-squares, rank       |\n";
    std::cout << "  | svd()              | dgesdd_    | Low-rank approx, PCA      |\n";
    std::cout << "  | eigenvalue_sym()   | dsyev_     | Symmetric eigenproblems   |\n";
    std::cout << "  +--------------------+------------+---------------------------+\n\n";

#ifdef MTL5_HAS_LAPACK
    std::cout << "  LAPACK is enabled - factorizations dispatch to vendor routines.\n\n";
#else
    std::cout << "  LAPACK is disabled - MTL5 uses its own C++ implementations.\n";
    std::cout << "  These are correct but not SIMD-optimized like MKL/OpenBLAS.\n\n";
#endif

    // ══════════════════════════════════════════════════════════════════════
    // 5. Architecture: How It All Fits Together
    // ══════════════════════════════════════════════════════════════════════
    std::cout << "=== 5. The Full Picture ===\n\n";

    std::cout << "  User code:     C = A * B;   // or lu(A), qr(A), svd(A)\n";
    std::cout << "                     |\n";
    std::cout << "                     v\n";
    std::cout << "  MTL5 operation layer (operation/mult.hpp, operation/lu.hpp, ...)\n";
    std::cout << "                     |\n";
    std::cout << "        +------------+------------+\n";
    std::cout << "        |                         |\n";
    std::cout << "   #ifdef MTL5_HAS_BLAS     Generic C++ fallback\n";
    std::cout << "        |                    (always works,\n";
    std::cout << "        v                     ~1-5 GFLOP/s)\n";
    std::cout << "   interface/blas.hpp\n";
    std::cout << "   interface/lapack.hpp\n";
    std::cout << "        |\n";
    std::cout << "        v\n";
    std::cout << "   Vendor library\n";
    std::cout << "   (MKL, OpenBLAS,\n";
    std::cout << "    cuBLAS, rocBLAS)\n";
    std::cout << "   ~50-200 GFLOP/s\n\n";

    std::cout << "This two-tier design means:\n";
    std::cout << "  - MTL5 works out of the box on any C++20 compiler\n";
    std::cout << "  - Flip one CMake flag to get 10-50x speedup on hot paths\n";
    std::cout << "  - No code changes needed - dispatch is compile-time\n";
    std::cout << "  - Same binary can be profiled to identify BLAS bottlenecks\n";

    return EXIT_SUCCESS;
}
