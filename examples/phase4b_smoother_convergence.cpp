// phase4b_smoother_convergence.cpp - Stationary Iterative Methods
//
// This example demonstrates:
//   1. Jacobi, Gauss-Seidel, and SOR as standalone solvers
//   2. How SOR(omega=1) reduces to Gauss-Seidel
//   3. The effect of relaxation parameter omega on convergence
//   4. Why these methods are slow but useful as smoothers in multigrid
//
// We solve the 1D Poisson equation -u'' = f on [0,1] and track
// the relative residual ||b - Ax||_2 / ||b||_2 at each sweep.

#include <mtl/mtl.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <numbers>
#include <vector>

using namespace mtl;

int main() {
    std::cout << "=============================================================\n";
    std::cout << " Phase 4B: Smoother Convergence - Jacobi / GS / SOR\n";
    std::cout << "=============================================================\n\n";

    // -- Build 1D Poisson system (sparse) ---------------------------------
    const std::size_t n = 50;
    const double h = 1.0 / (n + 1);

    mat::compressed2D<double> A(n, n);
    {
        mat::inserter<mat::compressed2D<double>> ins(A, 3);
        for (std::size_t i = 0; i < n; ++i) {
            ins[i][i] << 2.0 / (h * h);
            if (i > 0)     ins[i][i-1] << -1.0 / (h * h);
            if (i + 1 < n) ins[i][i+1] << -1.0 / (h * h);
        }
    }

    // RHS: f(x) = 1 (constant)
    vec::dense_vector<double> b(n, 1.0);
    double b_norm = two_norm(b);

    // Helper: compute relative residual
    auto rel_residual = [&](const vec::dense_vector<double>& x) {
        auto r = A * x;
        vec::dense_vector<double> res(n);
        for (std::size_t i = 0; i < n; ++i)
            res(i) = b(i) - r(i);
        return two_norm(res) / b_norm;
    };

    const int max_sweeps = 100;
    const int print_interval = 10;

    // -- Jacobi -----------------------------------------------------------
    std::cout << "--- Jacobi Method ---\n";
    std::cout << "Updates all x_i simultaneously using old values.\n";
    std::cout << "Each sweep requires one matrix-vector-like pass.\n\n";

    itl::smoother::jacobi<mat::compressed2D<double>> jac(A);
    vec::dense_vector<double> x_jac(n, 0.0);
    std::vector<double> hist_jac;

    for (int sweep = 0; sweep <= max_sweeps; ++sweep) {
        double rr = rel_residual(x_jac);
        hist_jac.push_back(rr);
        if (sweep % print_interval == 0)
            std::cout << "  sweep " << std::setw(3) << sweep
                      << "  rel_resid = " << std::scientific << rr << "\n";
        if (sweep < max_sweeps) jac(x_jac, b);
    }

    // -- Gauss-Seidel -----------------------------------------------------
    std::cout << "\n--- Gauss-Seidel Method ---\n";
    std::cout << "Updates x_i in-place, immediately using newest values.\n";
    std::cout << "Converges ~2x faster than Jacobi for 1D Poisson.\n\n";

    itl::smoother::gauss_seidel<mat::compressed2D<double>> gs(A);
    vec::dense_vector<double> x_gs(n, 0.0);
    std::vector<double> hist_gs;

    for (int sweep = 0; sweep <= max_sweeps; ++sweep) {
        double rr = rel_residual(x_gs);
        hist_gs.push_back(rr);
        if (sweep % print_interval == 0)
            std::cout << "  sweep " << std::setw(3) << sweep
                      << "  rel_resid = " << std::scientific << rr << "\n";
        if (sweep < max_sweeps) gs(x_gs, b);
    }

    // -- SOR with multiple omega values -----------------------------------
    std::cout << "\n--- SOR (Successive Over-Relaxation) ---\n";
    std::cout << "x_new = omega * GS_update + (1-omega) * x_old\n";
    std::cout << "omega = 1.0 is Gauss-Seidel. Optimal omega for 1D Poisson:\n";
    double omega_opt = 2.0 / (1.0 + std::sin(std::numbers::pi * h));
    std::cout << "  omega_opt = 2/(1 + sin(pi*h)) = " << std::fixed
              << std::setprecision(4) << omega_opt << "\n\n";

    std::vector<double> omegas = {0.5, 1.0, 1.2, 1.5, omega_opt};
    std::vector<std::string> omega_names = {"0.50", "1.00", "1.20", "1.50", "opt "};
    std::vector<std::vector<double>> hist_sor(omegas.size());

    for (std::size_t w = 0; w < omegas.size(); ++w) {
        itl::smoother::sor<mat::compressed2D<double>> sor_smoother(A, omegas[w]);
        vec::dense_vector<double> x_sor(n, 0.0);

        for (int sweep = 0; sweep <= max_sweeps; ++sweep) {
            hist_sor[w].push_back(rel_residual(x_sor));
            if (sweep < max_sweeps) sor_smoother(x_sor, b);
        }
    }

    // Print SOR comparison table
    std::cout << std::setw(6) << "Sweep";
    for (std::size_t w = 0; w < omegas.size(); ++w)
        std::cout << std::setw(14) << ("w=" + omega_names[w]);
    std::cout << "\n" << std::string(6 + 14 * omegas.size(), '-') << "\n";

    for (int sweep = 0; sweep <= max_sweeps; sweep += print_interval) {
        std::cout << std::setw(6) << sweep;
        for (std::size_t w = 0; w < omegas.size(); ++w)
            std::cout << std::scientific << std::setw(14) << std::setprecision(3)
                      << hist_sor[w][sweep];
        std::cout << "\n";
    }

    // -- Final Summary ----------------------------------------------------
    std::cout << "\n--- Convergence after " << max_sweeps << " sweeps ---\n";
    std::cout << std::setw(20) << "Method" << std::setw(16) << "Rel Residual" << "\n";
    std::cout << std::string(36, '-') << "\n";
    std::cout << std::setw(20) << "Jacobi"
              << std::scientific << std::setw(16) << hist_jac.back() << "\n";
    std::cout << std::setw(20) << "Gauss-Seidel"
              << std::scientific << std::setw(16) << hist_gs.back() << "\n";
    for (std::size_t w = 0; w < omegas.size(); ++w)
        std::cout << std::setw(14) << ("SOR(w=" + omega_names[w] + ")")
                  << std::scientific << std::setw(16) << hist_sor[w].back() << "\n";

    // -- Commentary -------------------------------------------------------
    std::cout << "\n=== Key Takeaways ===\n";
    std::cout << "1. Jacobi is the slowest: spectral radius rho(M_J) close to 1.\n";
    std::cout << "2. Gauss-Seidel ~ 2x faster: rho(M_GS) = rho(M_J)^2.\n";
    std::cout << "3. SOR with optimal omega is dramatically faster.\n";
    std::cout << "   For 1D Poisson: optimal omega = 2/(1+sin(pi*h)).\n";
    std::cout << "4. omega > 2 always diverges (spectral radius > 1).\n";
    std::cout << "5. These methods are too slow as standalone solvers\n";
    std::cout << "   (convergence rate depends on mesh size h), but they\n";
    std::cout << "   effectively smooth high-frequency error components,\n";
    std::cout << "   making them ideal as smoothers in multigrid methods.\n";

    return EXIT_SUCCESS;
}
