// phase4a_laplacian_2d.cpp - 2D Laplacian: Sparse Assembly + GMRES
//
// This example demonstrates:
//   1. How to assemble a sparse matrix with the compressed2D inserter
//   2. The 5-point stencil for the 2D Laplacian
//   3. GMRES with different restart parameters
//   4. Comparison between GMRES and CG on the same SPD system
//
// Physics: steady-state heat equation in 2D:
//   -Laplacian(u) = f on [0,1]^2, u = 0 on boundary
// Known solution: u(x,y) = sin(pi*x)*sin(pi*y)

#include <mtl/mtl.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <numbers>

using namespace mtl;

int main() {
    std::cout << "=============================================================\n";
    std::cout << " Phase 4A: 2D Laplacian - Sparse Assembly + GMRES\n";
    std::cout << "=============================================================\n\n";

    // ── Grid Setup ───────────────────────────────────────────────────────
    // n×n interior grid on [0,1]^2, total N = n^2 unknowns
    const std::size_t n = 10;  // 10x10 interior grid
    const std::size_t N = n * n;
    const double h = 1.0 / (n + 1);
    const double pi = std::numbers::pi;

    std::cout << "Grid: " << n << "x" << n << " interior points = " << N << " unknowns\n";
    std::cout << "Grid spacing h = " << h << "\n\n";

    // ── Sparse Assembly with Inserter ────────────────────────────────────
    std::cout << "--- Assembling 2D Laplacian with 5-point stencil ---\n";
    std::cout << "Stencil:    -1\n";
    std::cout << "         -1  4  -1    (scaled by 1/h^2)\n";
    std::cout << "            -1\n\n";

    // Grid ordering: linear index = ix + iy * n
    mat::compressed2D<double> A(N, N);
    {
        mat::inserter<mat::compressed2D<double>> ins(A, 5);  // ~5 nnz per row

        for (std::size_t iy = 0; iy < n; ++iy) {
            for (std::size_t ix = 0; ix < n; ++ix) {
                std::size_t row = ix + iy * n;

                // Center: 4/h^2
                ins[row][row] << 4.0 / (h * h);

                // Left neighbor (ix-1)
                if (ix > 0)
                    ins[row][row - 1] << -1.0 / (h * h);

                // Right neighbor (ix+1)
                if (ix + 1 < n)
                    ins[row][row + 1] << -1.0 / (h * h);

                // Bottom neighbor (iy-1)
                if (iy > 0)
                    ins[row][row - n] << -1.0 / (h * h);

                // Top neighbor (iy+1)
                if (iy + 1 < n)
                    ins[row][row + n] << -1.0 / (h * h);
            }
        }
    }  // inserter destructor finalizes the CRS structure

    std::cout << "Matrix size: " << N << " x " << N
              << ", nnz = " << A.nnz()
              << " (density = " << std::fixed << std::setprecision(2)
              << 100.0 * A.nnz() / (N * N) << "%)\n\n";

    // ── Build RHS ─────────────────────────────────────────────────────────
    // f(x,y) = 1 (constant forcing - excites all eigenmodes for nontrivial convergence)
    vec::dense_vector<double> b(N, 1.0);

    // Preconditioner: identity (no preconditioning) to show raw solver behavior
    itl::pc::identity<mat::compressed2D<double>> no_pc(A);

    // ── Solve with GMRES (restart=5) ─────────────────────────────────────
    std::cout << "--- Solve 1: GMRES(5) - very small Krylov subspace ---\n";
    std::cout << "Restart limits memory to 5 vectors but needs many restarts.\n\n";

    vec::dense_vector<double> x1(N, 0.0);
    itl::cyclic_iteration<double> iter1(b, 500, 1.0e-8, 10);
    int info1 = itl::gmres(A, x1, b, no_pc, iter1, 5);
    std::cout << "\n";

    // ── Solve with GMRES (restart=20) ────────────────────────────────────
    std::cout << "--- Solve 2: GMRES(20) - larger Krylov subspace ---\n";
    std::cout << "More memory per cycle but typically fewer total iterations.\n\n";

    vec::dense_vector<double> x2(N, 0.0);
    itl::cyclic_iteration<double> iter2(b, 500, 1.0e-8, 10);
    int info2 = itl::gmres(A, x2, b, no_pc, iter2, 20);
    std::cout << "\n";

    // ── Solve with CG (for comparison) ───────────────────────────────────
    std::cout << "--- Solve 3: CG - valid since 2D Laplacian is SPD ---\n";
    std::cout << "CG is optimal for SPD systems: guaranteed convergence in N steps.\n\n";

    vec::dense_vector<double> x3(N, 0.0);
    itl::cyclic_iteration<double> iter3(b, 500, 1.0e-8, 10);
    int info3 = itl::cg(A, x3, b, no_pc, iter3);
    std::cout << "\n";

    // ── Summary table ────────────────────────────────────────────────────
    std::cout << "--- Results Summary ---\n";
    std::cout << std::setw(15) << "Solver"
              << std::setw(12) << "Iterations"
              << std::setw(10) << "Status" << "\n";
    std::cout << std::string(37, '-') << "\n";
    std::cout << std::setw(15) << "GMRES(5)"
              << std::setw(12) << iter1.iterations()
              << std::setw(10) << (info1 == 0 ? "OK" : "FAIL") << "\n";
    std::cout << std::setw(15) << "GMRES(20)"
              << std::setw(12) << iter2.iterations()
              << std::setw(10) << (info2 == 0 ? "OK" : "FAIL") << "\n";
    std::cout << std::setw(15) << "CG"
              << std::setw(12) << iter3.iterations()
              << std::setw(10) << (info3 == 0 ? "OK" : "FAIL") << "\n\n";

    // ── Verify solutions agree ─────────────────────────────────────────
    double max_diff = 0.0;
    for (std::size_t i = 0; i < N; ++i)
        max_diff = std::max(max_diff, std::abs(x1(i) - x3(i)));
    std::cout << "Max |GMRES(5) - CG| solution difference: "
              << std::scientific << max_diff << "\n";
    std::cout << "(All solvers converge to the same answer.)\n\n";

    // ── Commentary ───────────────────────────────────────────────────────
    std::cout << "=== Key Takeaways ===\n";
    std::cout << "1. The inserter pattern: create inserter in a scope, fill with\n";
    std::cout << "   ins[row][col] << value, then destructor finalizes CRS arrays.\n";
    std::cout << "2. GMRES restart parameter trades memory vs convergence speed:\n";
    std::cout << "   - Small restart: less memory, possibly more cycles\n";
    std::cout << "   - Large restart: more memory, fewer cycles\n";
    std::cout << "   - Full GMRES (restart=N): guaranteed convergence in N steps\n";
    std::cout << "3. For SPD systems, CG beats GMRES: simpler, less memory,\n";
    std::cout << "   same convergence guarantee. Use GMRES when A is non-symmetric.\n";
    std::cout << "4. The 5-point stencil gives O(h^2) accuracy. Higher-order\n";
    std::cout << "   stencils (9-point) give O(h^4) but denser matrices.\n";

    return EXIT_SUCCESS;
}
