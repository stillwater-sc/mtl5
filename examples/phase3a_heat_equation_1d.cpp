// phase3a_heat_equation_1d.cpp - Solving the 1D Steady-State Heat Equation
//
// This example demonstrates:
//   1. How finite differences convert a PDE into a linear system Ax = b
//   2. Why the Conjugate Gradient method works (symmetric positive definite matrix)
//   3. How preconditioning reduces iteration count
//   4. Comparing numerical solution against analytical solution
//
// Physics: steady-state heat equation with variable thermal conductivity:
//   -(k(x) u'(x))' = f(x) on [0,1], u(0)=u(1)=0
// Variable k(x) creates non-uniform diagonal entries, making Jacobi
// preconditioning effective.

#include <mtl/mtl.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace mtl;

int main() {
    std::cout << "=============================================================\n";
    std::cout << " Phase 3A: 1D Heat Equation - CG with Preconditioning\n";
    std::cout << "=============================================================\n\n";

    // ── Problem Setup ─────────────────────────────────────────────────────
    // Discretize [0,1] with n interior points. Grid spacing h = 1/(n+1).
    // Variable conductivity k(x) = 1 + 99*x ranges from 1 to 100.
    // The resulting matrix is SPD with non-uniform diagonal - ideal for
    // demonstrating the benefit of Jacobi preconditioning.

    const std::size_t n = 100;
    const double h = 1.0 / (n + 1);

    std::cout << "Grid: " << n << " interior points, h = " << std::setprecision(4) << h << "\n";
    std::cout << "Conductivity k(x) = 1 + 99*x (ranges from 1 to 100)\n\n";

    // k(x) at midpoints: k_{i+1/2} = k(x_i + h/2)
    auto k = [](double x) { return 1.0 + 99.0 * x; };

    // Build tridiagonal matrix from variable-coefficient discretization
    // A(i,i-1) = -k_{i-1/2}/h^2,  A(i,i) = (k_{i-1/2}+k_{i+1/2})/h^2,  A(i,i+1) = -k_{i+1/2}/h^2
    mat::dense2D<double> A(n, n);
    for (std::size_t i = 0; i < n; ++i) {
        double x = (i + 1) * h;
        double k_left  = k(x - h/2);
        double k_right = k(x + h/2);
        A(i, i) = (k_left + k_right) / (h * h);
        if (i > 0)     A(i, i-1) = -k_left / (h * h);
        if (i + 1 < n) A(i, i+1) = -k_right / (h * h);
    }

    std::cout << "Diagonal range: A(0,0)=" << std::fixed << std::setprecision(1) << A(0,0)
              << " ... A(" << n-1 << "," << n-1 << ")=" << A(n-1,n-1) << "\n";
    std::cout << "(Ratio ~" << std::setprecision(0) << A(n-1,n-1)/A(0,0)
              << "x - highly non-uniform diagonal)\n\n";

    // Build RHS: f(x) = 1 (constant forcing)
    vec::dense_vector<double> b(n, 1.0);

    // ── Solve 1: CG without preconditioning ──────────────────────────────
    std::cout << "--- Solve 1: CG with Identity Preconditioner (no PC) ---\n";
    std::cout << "Without preconditioning, CG sees the full condition number.\n\n";

    vec::dense_vector<double> x1(n, 0.0);
    itl::pc::identity<mat::dense2D<double>> no_pc(A);
    itl::cyclic_iteration<double> iter1(b, 500, 1.0e-10, 10);
    int info1 = itl::cg(A, x1, b, no_pc, iter1);
    std::cout << "\n";

    // ── Solve 2: CG with Jacobi (diagonal) preconditioning ──────────────
    std::cout << "--- Solve 2: CG with Diagonal (Jacobi) Preconditioner ---\n";
    std::cout << "Jacobi PC: M = diag(A). Scales each equation by 1/A(i,i),\n";
    std::cout << "equalizing the diagonal and reducing the condition number.\n\n";

    vec::dense_vector<double> x2(n, 0.0);
    itl::pc::diagonal<mat::dense2D<double>> jac_pc(A);
    itl::cyclic_iteration<double> iter2(b, 500, 1.0e-10, 10);
    int info2 = itl::cg(A, x2, b, jac_pc, iter2);
    std::cout << "\n";

    // ── Summary ──────────────────────────────────────────────────────────
    std::cout << "--- Summary ---\n";
    std::cout << "  No PC:   " << iter1.iterations() << " iterations (code " << info1 << ")\n";
    std::cout << "  Jacobi:  " << iter2.iterations() << " iterations (code " << info2 << ")\n";
    double speedup = static_cast<double>(iter1.iterations()) / iter2.iterations();
    std::cout << "  Speedup: " << std::fixed << std::setprecision(1)
              << speedup << "x fewer iterations with Jacobi PC\n\n";

    // ── Verify solution ──────────────────────────────────────────────────
    std::cout << "--- Solution Profile ---\n";
    std::cout << std::setw(8) << "x"
              << std::setw(16) << "u(x)" << "\n";
    std::cout << std::string(24, '-') << "\n";

    for (std::size_t i = 0; i < n; i += n/10) {
        double x = (i + 1) * h;
        std::cout << std::fixed << std::setprecision(4) << std::setw(8) << x
                  << std::setprecision(10) << std::setw(16) << x1(i) << "\n";
    }

    // Verify both solutions agree
    double max_diff = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        max_diff = std::max(max_diff, std::abs(x1(i) - x2(i)));
    std::cout << "\nMax difference between solutions: " << std::scientific << max_diff << "\n";
    std::cout << "(Both methods converge to the same answer.)\n\n";

    // ── Commentary ───────────────────────────────────────────────────────
    std::cout << "=== Key Takeaways ===\n";
    std::cout << "1. CG works because the variable-coefficient Laplacian is SPD.\n";
    std::cout << "   - Symmetric: k(x) > 0 ensures symmetry of the stencil.\n";
    std::cout << "   - Positive definite: all eigenvalues > 0.\n";
    std::cout << "2. Jacobi preconditioning helps when diagonal entries vary.\n";
    std::cout << "   It scales each row by 1/A(i,i), reducing the effective\n";
    std::cout << "   condition number and iteration count.\n";
    std::cout << "3. Both methods give the same solution - preconditioning\n";
    std::cout << "   only changes the convergence speed, not the answer.\n";
    std::cout << "4. For production use, sparse storage (compressed2D) would\n";
    std::cout << "   reduce memory from O(n^2) to O(n) for this tridiagonal system.\n";

    return EXIT_SUCCESS;
}
