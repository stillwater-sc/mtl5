// phase13b_multigrid_poisson.cpp — Multigrid for 1D Poisson
//
// This example demonstrates multigrid methods for solving -u'' = f on [0,1]:
//
//   1. Build a 3-level geometric multigrid hierarchy
//   2. Show V-cycle convergence as a stand-alone solver
//   3. Show multigrid as a preconditioner for CG
//   4. Compare convergence rates: CG alone vs. MG-preconditioned CG
//
// The 1D Poisson equation with Dirichlet BCs is a classic benchmark
// for multigrid. Multigrid achieves O(n) solution cost, independent
// of problem size — far superior to unpreconditioned Krylov methods.

#include <mtl/mtl.hpp>
#include <iostream>
#include <iomanip>
#include <vector>

using namespace mtl;

/// Build 1D Poisson matrix for n interior points.
static mat::compressed2D<double> make_poisson(std::size_t n) {
    double h2_inv = static_cast<double>((n + 1) * (n + 1));
    mat::compressed2D<double> A(n, n);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        for (std::size_t i = 0; i < n; ++i) {
            ins[i][i] << 2.0 * h2_inv;
            if (i > 0)     ins[i][i-1] << -1.0 * h2_inv;
            if (i < n - 1) ins[i][i+1] << -1.0 * h2_inv;
        }
    }
    return A;
}

int main() {
    std::cout << std::scientific << std::setprecision(6);
    std::cout << "=== Multigrid for 1D Poisson Equation ===\n\n";

    // ── 1. Build 3-level hierarchy: 63 → 31 → 15 ─────────────────────
    // Sizes follow the pattern n = 2^k - 1 for geometric coarsening
    const std::size_t n0 = 63;  // finest
    const std::size_t n1 = 31;
    const std::size_t n2 = 15;  // coarsest

    auto A0 = make_poisson(n0);
    auto A1 = make_poisson(n1);
    auto A2 = make_poisson(n2);

    auto R0 = itl::mg::make_restriction_1d(n0);
    auto R1 = itl::mg::make_restriction_1d(n1);
    auto P0 = itl::mg::make_prolongation_1d(n1);
    auto P1 = itl::mg::make_prolongation_1d(n2);

    std::vector<mat::compressed2D<double>> levels = {A0, A1, A2};
    std::vector<mat::compressed2D<double>> restrictors = {R0, R1};
    std::vector<mat::compressed2D<double>> prolongators = {P0, P1};

    std::cout << "Hierarchy: " << n0 << " -> " << n1 << " -> " << n2 << "\n\n";

    // Smoother: Gauss-Seidel
    auto smoother_factory = [](const mat::compressed2D<double>& A) {
        return itl::smoother::gauss_seidel<mat::compressed2D<double>>(A);
    };

    // Coarse solver: many GS iterations (effectively exact)
    auto coarse_solver = [&A2](vec::dense_vector<double>& x, const vec::dense_vector<double>& b) {
        itl::smoother::gauss_seidel<mat::compressed2D<double>> gs(A2);
        for (int i = 0; i < 100; ++i)
            gs(x, b);
    };

    itl::mg::multigrid<double> mg(levels, restrictors, prolongators,
                                   smoother_factory, coarse_solver, 2, 2);

    // RHS: constant forcing f(x) = 1
    vec::dense_vector<double> b(n0, 1.0);

    // ── 2. V-cycle as stand-alone solver ────────────────────────────────
    std::cout << "--- V-cycle as stand-alone solver ---\n";
    std::cout << std::setw(8) << "Cycle" << std::setw(16) << "||r|| / ||b||\n";
    std::cout << std::string(24, '-') << "\n";

    vec::dense_vector<double> x_mg(n0, 0.0);
    double norm_b = mtl::two_norm(b);

    for (int cycle = 1; cycle <= 15; ++cycle) {
        mg.vcycle(x_mg, b);

        auto Ax = A0 * x_mg;
        vec::dense_vector<double> r(n0);
        for (std::size_t i = 0; i < n0; ++i)
            r(i) = b(i) - Ax(i);
        double rel_resid = mtl::two_norm(r) / norm_b;

        std::cout << std::setw(8) << cycle << std::setw(16) << rel_resid << "\n";

        if (rel_resid < 1e-10) break;
    }

    // ── 3. CG without preconditioning ───────────────────────────────────
    std::cout << "\n--- CG without preconditioning ---\n";
    {
        vec::dense_vector<double> x(n0, 0.0);
        itl::pc::identity<mat::compressed2D<double>> pc(A0);
        itl::basic_iteration<double> iter(b, 500, 1e-10);
        itl::cg(A0, x, b, pc, iter);
        std::cout << "Converged in " << iter.iterations() << " iterations\n";

        auto Ax = A0 * x;
        vec::dense_vector<double> r(n0);
        for (std::size_t i = 0; i < n0; ++i)
            r(i) = b(i) - Ax(i);
        std::cout << "Relative residual: " << mtl::two_norm(r) / norm_b << "\n";
    }

    // ── 4. MG-preconditioned CG ─────────────────────────────────────────
    std::cout << "\n--- CG with multigrid preconditioner ---\n";
    {
        vec::dense_vector<double> x(n0, 0.0);
        itl::basic_iteration<double> iter(b, 500, 1e-10);
        itl::cg(A0, x, b, mg, iter);
        std::cout << "Converged in " << iter.iterations() << " iterations\n";

        auto Ax = A0 * x;
        vec::dense_vector<double> r(n0);
        for (std::size_t i = 0; i < n0; ++i)
            r(i) = b(i) - Ax(i);
        std::cout << "Relative residual: " << mtl::two_norm(r) / norm_b << "\n";
    }

    std::cout << "\nDone.\n";
    return 0;
}
