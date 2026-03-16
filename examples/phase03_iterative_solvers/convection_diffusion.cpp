// convection_diffusion.cpp - When CG Fails: Non-Symmetric Systems
//
// This example demonstrates:
//   1. How convection creates a non-symmetric matrix
//   2. Why CG fails on non-symmetric systems (it requires SPD)
//   3. Why BiCGSTAB succeeds where CG fails
//   4. Three iteration controllers: basic, cyclic, noisy
//
// Physics: 1D convection-diffusion equation:
//   -epsilon * u''(x) + u'(x) = f(x) on [0,1], u(0)=u(1)=0
// The first derivative u'(x) creates asymmetry in the discrete operator.
// Small epsilon means convection-dominated flow (harder to solve).

#include <mtl/mtl.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace mtl;

/// Build convection-diffusion matrix with central differences
mat::dense2D<double> build_conv_diff(std::size_t n, double epsilon) {
    double h = 1.0 / (n + 1);
    mat::dense2D<double> A(n, n);

    for (std::size_t i = 0; i < n; ++i) {
        // Diffusion: -epsilon * u'' -> epsilon * (2u_i - u_{i-1} - u_{i+1}) / h^2
        A(i, i) = 2.0 * epsilon / (h * h);
        if (i > 0)     A(i, i-1) = -epsilon / (h * h) - 1.0 / (2.0 * h);  // + upwind
        if (i + 1 < n) A(i, i+1) = -epsilon / (h * h) + 1.0 / (2.0 * h);  // + downwind
    }
    return A;
}

int main() {
    std::cout << "=============================================================\n";
    std::cout << " Phase 3B: Convection-Diffusion - CG vs BiCGSTAB\n";
    std::cout << "=============================================================\n\n";

    const std::size_t n = 30;
    const double h = 1.0 / (n + 1);

    // Build RHS: simple constant forcing f=1
    vec::dense_vector<double> b(n, 1.0);

    // -- Regime 1: Diffusion-dominated (epsilon = 1.0) --------------------
    {
        double eps = 1.0;
        std::cout << "=== Regime 1: Diffusion-dominated (epsilon = " << eps << ") ===\n";
        std::cout << "When epsilon is large, the diffusion term -eps*u'' dominates.\n";
        std::cout << "The matrix is 'nearly symmetric' - CG may still work.\n\n";

        auto A = build_conv_diff(n, eps);

        // Show asymmetry: compare A(1,0) vs A(0,1)
        std::cout << "A(0,1) = " << std::setprecision(4) << A(0, 1)
                  << ",  A(1,0) = " << A(1, 0) << "\n";
        std::cout << "Asymmetry ratio: |A(0,1)-A(1,0)| / |A(0,1)+A(1,0)| = "
                  << std::abs(A(0,1) - A(1,0)) / std::abs(A(0,1) + A(1,0)) << "\n\n";

        // Try CG (will likely converge but shouldn't be trusted)
        vec::dense_vector<double> x_cg(n, 0.0);
        itl::pc::identity<mat::dense2D<double>> I(A);
        itl::basic_iteration<double> iter_cg(b, 500, 1.0e-8);
        int info_cg = itl::cg(A, x_cg, b, I, iter_cg);
        std::cout << "CG:      " << iter_cg.iterations() << " iterations, "
                  << "error code = " << info_cg << "\n";

        // BiCGSTAB (correct choice for non-symmetric)
        vec::dense_vector<double> x_bi(n, 0.0);
        itl::basic_iteration<double> iter_bi(b, 500, 1.0e-8);
        int info_bi = itl::bicgstab(A, x_bi, b, I, iter_bi);
        std::cout << "BiCGSTAB: " << iter_bi.iterations() << " iterations, "
                  << "error code = " << info_bi << "\n\n";
    }

    // -- Regime 2: Convection-dominated (epsilon = 0.01) ------------------
    {
        double eps = 0.01;
        std::cout << "=== Regime 2: Convection-dominated (epsilon = " << eps << ") ===\n";
        std::cout << "When epsilon is small, the convection term u' dominates.\n";
        std::cout << "The matrix is strongly non-symmetric - CG should fail.\n\n";

        auto A = build_conv_diff(n, eps);

        std::cout << "A(0,1) = " << std::setprecision(4) << A(0, 1)
                  << ",  A(1,0) = " << A(1, 0) << "\n";
        std::cout << "Asymmetry ratio: "
                  << std::abs(A(0,1) - A(1,0)) / std::abs(A(0,1) + A(1,0)) << "\n\n";

        // CG on strongly non-symmetric system
        vec::dense_vector<double> x_cg(n, 0.0);
        itl::pc::identity<mat::dense2D<double>> I(A);
        itl::basic_iteration<double> iter_cg(b, 500, 1.0e-8);
        int info_cg = itl::cg(A, x_cg, b, I, iter_cg);
        std::cout << "CG:       " << iter_cg.iterations() << " iterations, "
                  << "error code = " << info_cg;
        if (info_cg != 0) std::cout << " (FAILED - expected for non-SPD)";
        std::cout << "\n";

        // BiCGSTAB with diagonal preconditioner
        vec::dense_vector<double> x_bi(n, 0.0);
        itl::pc::diagonal<mat::dense2D<double>> diag_pc(A);
        itl::basic_iteration<double> iter_bi(b, 500, 1.0e-8);
        int info_bi = itl::bicgstab(A, x_bi, b, diag_pc, iter_bi);
        std::cout << "BiCGSTAB: " << iter_bi.iterations() << " iterations, "
                  << "error code = " << info_bi << "\n";

        // Verify solution: compute residual ||b - Ax||
        auto r = A * x_bi;
        double res_norm = 0.0;
        for (std::size_t i = 0; i < n; ++i)
            res_norm += (b(i) - r(i)) * (b(i) - r(i));
        res_norm = std::sqrt(res_norm);
        std::cout << "BiCGSTAB residual ||b - Ax||_2 = " << std::scientific << res_norm << "\n\n";
    }

    // -- Iteration Controllers Demo ---------------------------------------
    std::cout << "=== Iteration Controller Comparison ===\n";
    std::cout << "MTL5 provides three controllers with different verbosity:\n\n";

    double eps = 0.1;
    auto A = build_conv_diff(n, eps);
    itl::pc::identity<mat::dense2D<double>> I(A);

    // 1. basic_iteration - silent
    {
        vec::dense_vector<double> x(n, 0.0);
        itl::basic_iteration<double> iter(b, 200, 1.0e-8);
        itl::bicgstab(A, x, b, I, iter);
        std::cout << "basic_iteration (silent): converged in "
                  << iter.iterations() << " iterations\n";
    }

    // 2. cyclic_iteration - prints every N steps
    {
        std::cout << "\ncyclic_iteration (every 5 steps):\n";
        vec::dense_vector<double> x(n, 0.0);
        itl::cyclic_iteration<double> iter(b, 200, 1.0e-8, 5);
        itl::bicgstab(A, x, b, I, iter);
        std::cout << "\n";
    }

    // 3. noisy_iteration - prints every step
    {
        std::cout << "noisy_iteration (every step):\n";
        vec::dense_vector<double> x(n, 0.0);
        itl::noisy_iteration<double> iter(b, 200, 1.0e-8);
        itl::bicgstab(A, x, b, I, iter);
        std::cout << "\n";
    }

    // -- Commentary -------------------------------------------------------
    std::cout << "=== Key Takeaways ===\n";
    std::cout << "1. CG requires the matrix to be SPD (Symmetric Positive Definite).\n";
    std::cout << "   On non-symmetric systems it may diverge or give wrong answers.\n";
    std::cout << "2. BiCGSTAB handles non-symmetric systems by using two sequences\n";
    std::cout << "   of residuals. It's the go-to solver for general sparse systems.\n";
    std::cout << "3. The Peclet number Pe = |velocity|*h / (2*epsilon) determines\n";
    std::cout << "   whether the problem is diffusion- or convection-dominated.\n";
    std::cout << "4. Choose your iteration controller based on debugging needs:\n";
    std::cout << "   - basic: production code (no overhead)\n";
    std::cout << "   - cyclic: monitoring long runs\n";
    std::cout << "   - noisy: debugging convergence issues\n";

    return EXIT_SUCCESS;
}
