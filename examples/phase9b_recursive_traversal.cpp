// phase9b_recursive_traversal.cpp - Block-Recursive GEMM
//
// This example demonstrates:
//   1. Why naive ijk-loop GEMM thrashes cache for large matrices
//   2. How block-recursive subdivision keeps working sets in L1/L2
//   3. The performance crossover where blocking wins decisively
//   4. Connection to real-world tiling in DL compilers (XLA, TVM, Triton)
//
// Key insight: Matrix multiplication C += A * B with the naive ijk loop has
// poor temporal locality - each row of B is streamed through cache N times,
// one per row of A. Block-recursive GEMM subdivides A, B, C into quadrants
// and multiplies the sub-blocks, keeping the working set small enough to
// fit in cache. This is the fundamental strategy behind tiled GEMM in every
// DL compiler and BLAS library.
//
// The recursator subdivides matrices into quadrants:
//
//   C_NW = A_NW * B_NW + A_NE * B_SW     (top-left of C)
//   C_NE = A_NW * B_NE + A_NE * B_SE     (top-right of C)
//   C_SW = A_SW * B_NW + A_SE * B_SW     (bottom-left of C)
//   C_SE = A_SW * B_NE + A_SE * B_SE     (bottom-right of C)
//
// At the base case (block fits in cache), we use the naive ijk loop.
// This is cache-oblivious: no tuning parameter for cache size needed.

#include <mtl/mtl.hpp>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <cmath>

using namespace mtl;

// Prevent dead-code elimination
template <typename T>
void do_not_optimize(const T& val) {
    volatile auto sink = val;
    (void)sink;
}

// ── Naive ijk GEMM ─────────────────────────────────────────────────────

/// C += A * B using the textbook triple-nested loop.
/// Poor cache behavior: B columns are accessed with stride = num_cols.
void naive_gemm(mat::dense2D<double>& C,
                const mat::dense2D<double>& A,
                const mat::dense2D<double>& B) {
    auto M = A.num_rows();
    auto K = A.num_cols();
    auto N = B.num_cols();
    for (std::size_t i = 0; i < M; ++i)
        for (std::size_t j = 0; j < N; ++j)
            for (std::size_t k = 0; k < K; ++k)
                C(i, j) += A(i, k) * B(k, j);
}

// ── Block-Recursive GEMM via recursator ────────────────────────────────

/// Recursive GEMM: C += A * B
/// Subdivides all three matrices into quadrants and recurses.
/// At base case (any dimension <= threshold), falls through to naive ijk.
void recursive_gemm(recursion::recursator<mat::dense2D<double>> rC,
                    recursion::recursator<mat::dense2D<double>> rA,
                    recursion::recursator<mat::dense2D<double>> rB,
                    std::size_t threshold) {
    auto M = rA.num_rows();
    auto K = rA.num_cols();
    auto N = rB.num_cols();

    // Base case: block is small enough, use naive multiply
    if (M <= threshold || K <= threshold || N <= threshold) {
        for (std::size_t i = 0; i < M; ++i)
            for (std::size_t j = 0; j < N; ++j)
                for (std::size_t k = 0; k < K; ++k)
                    rC(i, j) += rA(i, k) * rB(k, j);
        return;
    }

    // Recursive case: split all three matrices into quadrants.
    //
    //   C_NW += A_NW * B_NW + A_NE * B_SW
    //   C_NE += A_NW * B_NE + A_NE * B_SE
    //   C_SW += A_SW * B_NW + A_SE * B_SW
    //   C_SE += A_SW * B_NE + A_SE * B_SE
    //
    // This gives 8 recursive half-size multiplications,
    // matching the structure of Strassen (without the clever additions).

    recursive_gemm(rC.north_west(), rA.north_west(), rB.north_west(), threshold);
    recursive_gemm(rC.north_west(), rA.north_east(), rB.south_west(), threshold);

    recursive_gemm(rC.north_east(), rA.north_west(), rB.north_east(), threshold);
    recursive_gemm(rC.north_east(), rA.north_east(), rB.south_east(), threshold);

    recursive_gemm(rC.south_west(), rA.south_west(), rB.north_west(), threshold);
    recursive_gemm(rC.south_west(), rA.south_east(), rB.south_west(), threshold);

    recursive_gemm(rC.south_east(), rA.south_west(), rB.north_east(), threshold);
    recursive_gemm(rC.south_east(), rA.south_east(), rB.south_east(), threshold);
}

/// Top-level wrapper: C = A * B using block-recursive algorithm
void blocked_gemm(mat::dense2D<double>& C,
                  const mat::dense2D<double>& A,
                  const mat::dense2D<double>& B,
                  std::size_t block_size) {
    // Zero C first (recursive_gemm accumulates with +=)
    for (std::size_t i = 0; i < C.num_rows(); ++i)
        for (std::size_t j = 0; j < C.num_cols(); ++j)
            C(i, j) = 0.0;

    // The recursator takes non-const refs, so we need mutable copies
    // of A and B. In a production library, we'd have const_recursator.
    // For this pedagogical example, we cast (the recursator only reads A,B).
    auto& A_mut = const_cast<mat::dense2D<double>&>(A);
    auto& B_mut = const_cast<mat::dense2D<double>&>(B);

    recursion::recursator<mat::dense2D<double>> rC(C);
    recursion::recursator<mat::dense2D<double>> rA(A_mut);
    recursion::recursator<mat::dense2D<double>> rB(B_mut);

    recursive_gemm(rC, rA, rB, block_size);
}

int main() {
    std::cout << "=============================================================\n";
    std::cout << " Phase 9B: Block-Recursive GEMM - Cache-Oblivious Tiling\n";
    std::cout << "=============================================================\n\n";

    // ══════════════════════════════════════════════════════════════════════
    // 1. The Problem: Why Naive GEMM Is Slow
    // ══════════════════════════════════════════════════════════════════════
    std::cout << "=== 1. The Problem: Cache Behavior of ijk GEMM ===\n\n";

    std::cout << "For C += A * B with N×N matrices:\n\n";
    std::cout << "  Naive ijk loop:\n";
    std::cout << "    for i in [0,N):        // row of A, row of C\n";
    std::cout << "      for j in [0,N):      // col of B, col of C\n";
    std::cout << "        for k in [0,N):    // col of A, row of B\n";
    std::cout << "          C(i,j) += A(i,k) * B(k,j)\n\n";

    std::cout << "  Cache analysis (row-major storage):\n";
    std::cout << "  - A(i,k): sequential in k → good spatial locality\n";
    std::cout << "  - B(k,j): stride-N access in k → column traversal = BAD\n";
    std::cout << "  - Each element of B is loaded N times total\n";
    std::cout << "  - When N > sqrt(cache_size / sizeof(double)),\n";
    std::cout << "    B's columns don't fit in cache → capacity misses\n\n";

    std::cout << "  Block-recursive GEMM:\n";
    std::cout << "  - Recursively subdivide A, B, C into quadrants\n";
    std::cout << "  - At base case, all three sub-blocks fit in L1/L2 cache\n";
    std::cout << "  - Working set: 3 * B^2 * 8 bytes (B = block size)\n";
    std::cout << "  - Cache-OBLIVIOUS: adapts to any cache hierarchy\n";
    std::cout << "  - Same strategy used by XLA, TVM, Triton, cuBLAS\n\n";

    // ══════════════════════════════════════════════════════════════════════
    // 2. Correctness Verification
    // ══════════════════════════════════════════════════════════════════════
    std::cout << "=== 2. Correctness: Naive vs Block-Recursive ===\n\n";

    const std::size_t Nv = 64;
    mat::dense2D<double> Av(Nv, Nv), Bv(Nv, Nv);
    mat::dense2D<double> Cn(Nv, Nv), Cb(Nv, Nv);

    for (std::size_t i = 0; i < Nv; ++i)
        for (std::size_t j = 0; j < Nv; ++j) {
            Av(i, j) = static_cast<double>((i + j) % 7) - 3.0;
            Bv(i, j) = static_cast<double>((i * 3 + j) % 11) - 5.0;
            Cn(i, j) = 0.0;
        }

    naive_gemm(Cn, Av, Bv);
    blocked_gemm(Cb, Av, Bv, 8);

    double max_err = 0.0;
    for (std::size_t i = 0; i < Nv; ++i)
        for (std::size_t j = 0; j < Nv; ++j)
            max_err = std::max(max_err, std::abs(Cn(i, j) - Cb(i, j)));

    std::cout << "  Matrix size: " << Nv << "x" << Nv << "\n";
    std::cout << "  Block size:  8\n";
    std::cout << "  Max |naive - blocked|: " << std::scientific << max_err << "\n";
    std::cout << "  Match: " << (max_err < 1e-10 ? "PASS" : "FAIL") << "\n\n";

    // ══════════════════════════════════════════════════════════════════════
    // 3. Performance Benchmark
    // ══════════════════════════════════════════════════════════════════════
    std::cout << "=== 3. Performance: Naive vs Block-Recursive ===\n\n";

    // Sizes chosen to show the crossover. For small N, recursion overhead
    // dominates. For large N, cache effects dominate and blocking wins.
    std::vector<std::size_t> sizes = {64, 128, 256, 512};
    const int warmup = 1;
    const int trials = 3;
    const std::size_t block_size = 32;  // ~32×32 = 8KB per block, fits in L1

    std::cout << "  Block size for recursive: " << block_size
              << "  (3 blocks = "
              << 3 * block_size * block_size * 8 / 1024 << " KB, fits L1)\n\n";

    std::cout << std::setw(6)  << "N"
              << std::setw(14) << "Naive(ms)"
              << std::setw(14) << "Blocked(ms)"
              << std::setw(10) << "Speedup"
              << std::setw(14) << "3*N^2*8(KB)"
              << "\n";
    std::cout << std::string(58, '-') << "\n";

    for (auto N : sizes) {
        mat::dense2D<double> A(N, N), B(N, N), C(N, N);
        for (std::size_t i = 0; i < N; ++i)
            for (std::size_t j = 0; j < N; ++j) {
                A(i, j) = 1.0 + 0.001 * static_cast<double>(i * N + j);
                B(i, j) = 2.0 - 0.001 * static_cast<double>(i * N + j);
            }

        // Warmup
        for (int w = 0; w < warmup; ++w) {
            for (std::size_t i = 0; i < N; ++i)
                for (std::size_t j = 0; j < N; ++j)
                    C(i, j) = 0.0;
            naive_gemm(C, A, B);
            do_not_optimize(C(0, 0));
        }

        // Time naive
        double naive_ms = 0.0;
        for (int t = 0; t < trials; ++t) {
            for (std::size_t i = 0; i < N; ++i)
                for (std::size_t j = 0; j < N; ++j)
                    C(i, j) = 0.0;
            auto t0 = std::chrono::steady_clock::now();
            naive_gemm(C, A, B);
            auto t1 = std::chrono::steady_clock::now();
            do_not_optimize(C(N/2, N/2));
            naive_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        naive_ms /= trials;

        // Time blocked
        double blocked_ms = 0.0;
        for (int t = 0; t < trials; ++t) {
            auto t0 = std::chrono::steady_clock::now();
            blocked_gemm(C, A, B, block_size);
            auto t1 = std::chrono::steady_clock::now();
            do_not_optimize(C(N/2, N/2));
            blocked_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        blocked_ms /= trials;

        double speedup = naive_ms / blocked_ms;
        double working_set_kb = 3.0 * N * N * 8.0 / 1024.0;

        std::cout << std::setw(6)  << N
                  << std::setw(14) << std::fixed << std::setprecision(2) << naive_ms
                  << std::setw(14) << std::fixed << std::setprecision(2) << blocked_ms
                  << std::setw(9)  << std::fixed << std::setprecision(2) << speedup << "x"
                  << std::setw(14) << std::fixed << std::setprecision(0) << working_set_kb
                  << "\n";
    }

    // ══════════════════════════════════════════════════════════════════════
    // 4. Block Decomposition Visualization
    // ══════════════════════════════════════════════════════════════════════
    std::cout << "\n=== 4. Recursator Decomposition ===\n\n";

    std::cout << "  8x8 matrix decomposed with min_dim_test(2):\n\n";

    mat::dense2D<double> D(8, 8);
    for (std::size_t i = 0; i < 8; ++i)
        for (std::size_t j = 0; j < 8; ++j)
            D(i, j) = 0.0;

    // Label each cell with its base-case block ID
    int block_id = 0;
    recursion::for_each(
        recursion::recursator<mat::dense2D<double>>(D),
        [&](auto& sub) {
            for (std::size_t i = 0; i < sub.num_rows(); ++i)
                for (std::size_t j = 0; j < sub.num_cols(); ++j)
                    sub(i, j) = static_cast<double>(block_id);
            ++block_id;
        },
        recursion::min_dim_test(2));

    std::cout << "  Block IDs (each 2x2 block processed as a unit):\n";
    for (std::size_t i = 0; i < 8; ++i) {
        std::cout << "  ";
        for (std::size_t j = 0; j < 8; ++j) {
            std::cout << std::setw(4) << static_cast<int>(D(i, j));
        }
        std::cout << "\n";
    }
    std::cout << "\n  " << block_id << " base-case blocks, each fitting in registers/L1\n";

    // ══════════════════════════════════════════════════════════════════════
    // 5. Connection to DL Compilers
    // ══════════════════════════════════════════════════════════════════════
    std::cout << "\n=== 5. From Recursators to DL Compilers ===\n\n";

    std::cout << "The block-recursive GEMM strategy is the foundation of every\n";
    std::cout << "high-performance matrix multiply in modern DL frameworks:\n\n";

    std::cout << "  cuBLAS / cuDNN:\n";
    std::cout << "    - Thread blocks tile the output matrix into ~128x128 blocks\n";
    std::cout << "    - Each thread block loads A,B tiles into shared memory\n";
    std::cout << "    - Warp-level tiles (~64x64) use tensor cores\n";
    std::cout << "    - 3-level tiling: grid → block → warp (like our recursion)\n\n";

    std::cout << "  XLA (TensorFlow/JAX compiler):\n";
    std::cout << "    - Emits tiled loops for matmul on CPU and GPU\n";
    std::cout << "    - Tile sizes chosen to match L1/L2/shared memory capacity\n";
    std::cout << "    - Fusion of elementwise ops within each tile\n\n";

    std::cout << "  TVM / Apache TVM:\n";
    std::cout << "    - schedule.tile(i, j, bi, bj, 32, 32) is explicit blocking\n";
    std::cout << "    - Auto-tuning searches over tile sizes per hardware target\n\n";

    std::cout << "  Triton (OpenAI):\n";
    std::cout << "    - tl.program_id(0/1) selects output tile\n";
    std::cout << "    - K-dimension reduced in blocks via tl.arange(0, BLOCK_K)\n";
    std::cout << "    - Programmer specifies block shape; Triton manages memory\n\n";

    std::cout << "Our cache-OBLIVIOUS recursion is the theoretical ideal:\n";
    std::cout << "no tuning parameters needed. Production libraries use\n";
    std::cout << "cache-AWARE tiling (fixed block sizes) for predictability,\n";
    std::cout << "but the underlying principle - divide until blocks fit in\n";
    std::cout << "fast memory - is identical.\n";

    return EXIT_SUCCESS;
}
