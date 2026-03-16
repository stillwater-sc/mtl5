// expression_benchmark.cpp - Cache-Efficient Expression Templates
//
// This example demonstrates:
//   1. How expression templates fuse element-wise operations into a single pass
//   2. The cache-efficiency advantage: fewer temporaries = fewer cache misses
//   3. Performance crossover at L3 cache boundary sizes
//
// Key insight: C = 2.0*A + 3.0*B requires 3 NxN matrices (A, B, C).
// With expression templates (lazy), only these 3 matrices must be in cache
// during the single fused pass. With eager evaluation, 2 extra temporaries
// are created (one for 2.0*A, one for 3.0*B), totaling 5 matrices - which
// can exceed L3 cache capacity and cause a performance cliff.

#include <mtl/mtl.hpp>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>

using namespace mtl;

// Prevent dead-code elimination: force the compiler to "use" a value
template <typename T>
void do_not_optimize(const T& val) {
    volatile auto sink = val;
    (void)sink;
}

int main() {
    std::cout << "=============================================================\n";
    std::cout << " Phase 8A: Expression Template Benchmark - Cache Efficiency\n";
    std::cout << "=============================================================\n\n";

    // ======================================================================
    // Background: Why Expression Templates Help
    // ======================================================================
    std::cout << "=== Background ===\n";
    std::cout << "Consider the expression:  C = 2.0 * A + 3.0 * B\n\n";
    std::cout << "Without expression templates (eager evaluation):\n";
    std::cout << "  temp1 = 2.0 * A       // allocate NxN, write all elements\n";
    std::cout << "  temp2 = 3.0 * B       // allocate NxN, write all elements\n";
    std::cout << "  C     = temp1 + temp2  // allocate NxN, read temp1+temp2\n";
    std::cout << "  Total: 5 matrices in flight (A, B, temp1, temp2, C)\n\n";
    std::cout << "With expression templates (lazy/fused evaluation):\n";
    std::cout << "  C(i,j) = 2.0*A(i,j) + 3.0*B(i,j)  // single fused loop\n";
    std::cout << "  Total: 3 matrices in flight (A, B, C)\n\n";
    std::cout << "When 3 matrices fit in L3 cache but 5 do not, the lazy\n";
    std::cout << "path wins decisively.\n\n";

    // ======================================================================
    // Benchmark: Sweep Matrix Sizes
    // ======================================================================
    std::cout << "=== Benchmark: Lazy vs. Eager Across Matrix Sizes ===\n\n";

    // Sizes chosen to span the L3 cache boundary (typically 8-12 MB)
    // 3 matrices of N=700 doubles = 3 x 700^2 x 8 ~= 11.2 MB  (fits)
    // 5 matrices of N=700 doubles = 5 x 700^2 x 8 ~= 18.7 MB  (exceeds)
    std::vector<std::size_t> sizes = {200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1500};
    const int warm_up = 2;
    const int trials  = 5;

    std::cout << std::setw(6)  << "N"
              << std::setw(12) << "Lazy(ms)"
              << std::setw(12) << "Eager(ms)"
              << std::setw(10) << "Ratio"
              << std::setw(14) << "3*N^2*8(MB)"
              << std::setw(14) << "5*N^2*8(MB)"
              << "\n";
    std::cout << std::string(68, '-') << "\n";

    for (auto N : sizes) {
        // Allocate and initialize source matrices
        mat::dense2D<double> A(N, N);
        mat::dense2D<double> B(N, N);
        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t j = 0; j < N; ++j) {
                A(i, j) = 1.0 + 0.001 * static_cast<double>(i * N + j);
                B(i, j) = 2.0 - 0.001 * static_cast<double>(i * N + j);
            }
        }

        // -- Warm-up passes ---------------------------------------------
        for (int w = 0; w < warm_up; ++w) {
            mat::dense2D<double> Cw = 2.0 * A + 3.0 * B;
            do_not_optimize(Cw(0, 0));
        }

        // -- Lazy path (expression templates) ---------------------------
        // The expression 2.0*A + 3.0*B builds an expression tree.
        // Assignment to dense2D<double> triggers a single fused loop.
        double lazy_ms = 0.0;
        for (int t = 0; t < trials; ++t) {
            auto t0 = std::chrono::steady_clock::now();
            mat::dense2D<double> C = 2.0 * A + 3.0 * B;
            auto t1 = std::chrono::steady_clock::now();
            do_not_optimize(C(N/2, N/2));
            lazy_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        lazy_ms /= trials;

        // -- Eager path (forced temporaries via evaluate()) -------------
        // evaluate() materializes each sub-expression into a new dense2D,
        // creating the 2 extra temporaries that bloat memory footprint.
        double eager_ms = 0.0;
        for (int t = 0; t < trials; ++t) {
            auto t0 = std::chrono::steady_clock::now();
            auto t1_mat = evaluate(2.0 * A);   // temp1: NxN allocation + write
            auto t2_mat = evaluate(3.0 * B);   // temp2: NxN allocation + write
            mat::dense2D<double> C = t1_mat + t2_mat;  // read temp1, temp2, write C
            auto t1_clk = std::chrono::steady_clock::now();
            do_not_optimize(C(N/2, N/2));
            eager_ms += std::chrono::duration<double, std::milli>(t1_clk - t0).count();
        }
        eager_ms /= trials;

        double ratio = eager_ms / lazy_ms;
        double mem_lazy_mb  = 3.0 * N * N * 8.0 / (1024.0 * 1024.0);
        double mem_eager_mb = 5.0 * N * N * 8.0 / (1024.0 * 1024.0);

        std::cout << std::setw(6)  << N
                  << std::setw(12) << std::fixed << std::setprecision(2) << lazy_ms
                  << std::setw(12) << std::fixed << std::setprecision(2) << eager_ms
                  << std::setw(9)  << std::fixed << std::setprecision(2) << ratio << "x"
                  << std::setw(14) << std::fixed << std::setprecision(1) << mem_lazy_mb
                  << std::setw(14) << std::fixed << std::setprecision(1) << mem_eager_mb
                  << "\n";
    }

    // ======================================================================
    // Correctness Verification
    // ======================================================================
    std::cout << "\n=== Correctness Verification ===\n";
    const std::size_t Nv = 100;
    mat::dense2D<double> Av(Nv, Nv), Bv(Nv, Nv);
    for (std::size_t i = 0; i < Nv; ++i)
        for (std::size_t j = 0; j < Nv; ++j) {
            Av(i, j) = static_cast<double>(i + j);
            Bv(i, j) = static_cast<double>(i * j);
        }

    // Lazy result
    mat::dense2D<double> C_lazy = 2.0 * Av + 3.0 * Bv;

    // Eager result
    auto t1_v = evaluate(2.0 * Av);
    auto t2_v = evaluate(3.0 * Bv);
    mat::dense2D<double> C_eager = t1_v + t2_v;

    // Manual result
    double max_err = 0.0;
    for (std::size_t i = 0; i < Nv; ++i)
        for (std::size_t j = 0; j < Nv; ++j) {
            double expected = 2.0 * static_cast<double>(i + j) + 3.0 * static_cast<double>(i * j);
            double err_lazy  = std::abs(C_lazy(i, j) - expected);
            double err_eager = std::abs(C_eager(i, j) - expected);
            double err_match = std::abs(C_lazy(i, j) - C_eager(i, j));
            max_err = std::max({max_err, err_lazy, err_eager, err_match});
        }
    std::cout << "Max error (lazy vs manual):  " << std::scientific << max_err << "\n";
    std::cout << "Lazy == Eager:               " << (max_err < 1e-12 ? "PASS" : "FAIL") << "\n\n";

    // ======================================================================
    // Key Takeaways
    // ======================================================================
    std::cout << "=== Key Takeaways ===\n";
    std::cout << "1. Expression templates eliminate temporary matrices by fusing\n";
    std::cout << "   element-wise operations into a single traversal.\n";
    std::cout << "2. Fewer temporaries = smaller working set = better cache usage.\n";
    std::cout << "3. The advantage is most pronounced at the L3 cache boundary:\n";
    std::cout << "   when 3 matrices fit but 5 do not, lazy evaluation avoids\n";
    std::cout << "   spilling to main memory.\n";
    std::cout << "4. evaluate() forces materialization - useful when you need a\n";
    std::cout << "   concrete matrix, but it defeats the fusion optimization.\n";
    std::cout << "5. For element-wise ops, always prefer the natural syntax:\n";
    std::cout << "       C = 2.0 * A + 3.0 * B;   // fused, cache-friendly\n";
    std::cout << "   over manually materializing intermediates.\n";

    return EXIT_SUCCESS;
}
