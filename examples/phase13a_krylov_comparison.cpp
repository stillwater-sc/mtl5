// phase13a_krylov_comparison.cpp — Krylov Solver Comparison
//
// This example compares the new Phase 13 Krylov solvers on sparse systems:
//
//   1. CGS (Conjugate Gradient Squared) on a non-symmetric system
//   2. BiCGSTAB(2) and BiCGSTAB(4) showing higher-order stabilization
//   3. MINRES on a symmetric indefinite system
//   4. Iteration count comparison across solvers
//
// Each solver is tested with identity and diagonal preconditioners to
// demonstrate the effect of preconditioning on convergence.

#include <mtl/mtl.hpp>
#include <iostream>
#include <iomanip>
#include <string>

using namespace mtl;

/// Build a non-symmetric sparse system: convection-diffusion
static mat::compressed2D<double> make_convection_diffusion(std::size_t n, double peclet) {
    mat::compressed2D<double> A(n, n);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        for (std::size_t i = 0; i < n; ++i) {
            ins[i][i] << 2.0;
            if (i > 0)     ins[i][i-1] << -1.0 - peclet;
            if (i < n - 1) ins[i][i+1] << -1.0 + peclet;
        }
    }
    return A;
}

/// Build a symmetric indefinite system
static mat::compressed2D<double> make_symmetric_indefinite(std::size_t n) {
    mat::compressed2D<double> A(n, n);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        for (std::size_t i = 0; i < n; ++i) {
            // Alternating positive/negative diagonal
            double diag = (i % 2 == 0) ? 4.0 : -3.0;
            ins[i][i] << diag;
            if (i > 0)     ins[i][i-1] << 1.0;
            if (i < n - 1) ins[i][i+1] << 1.0;
        }
    }
    return A;
}

/// Verify solution: compute ||b - A*x|| / ||b||
static double relative_residual(const mat::compressed2D<double>& A,
                                const vec::dense_vector<double>& x,
                                const vec::dense_vector<double>& b) {
    auto Ax = A * x;
    vec::dense_vector<double> r(b.size());
    for (std::size_t i = 0; i < b.size(); ++i)
        r(i) = b(i) - Ax(i);
    return mtl::two_norm(r) / mtl::two_norm(b);
}

int main() {
    std::cout << std::scientific << std::setprecision(4);
    const std::size_t n = 50;

    // ── 1. Non-symmetric system: CGS and BiCGSTAB variants ──────────────
    std::cout << "=== Krylov Solver Comparison ===\n\n";
    std::cout << "--- Non-symmetric system (convection-diffusion, n=" << n << ") ---\n";

    auto A_ns = make_convection_diffusion(n, 0.1);
    vec::dense_vector<double> b(n, 1.0);

    struct Result {
        std::string name;
        int iters;
        double rel_resid;
    };
    std::vector<Result> results;

    // CGS with identity PC
    {
        vec::dense_vector<double> x(n, 0.0);
        itl::pc::identity<mat::compressed2D<double>> pc(A_ns);
        itl::basic_iteration<double> iter(b, 500, 1e-10);
        itl::cgs(A_ns, x, b, pc, iter);
        results.push_back({"CGS (identity PC)", iter.iterations(),
                           relative_residual(A_ns, x, b)});
    }

    // CGS with diagonal PC
    {
        vec::dense_vector<double> x(n, 0.0);
        itl::pc::diagonal<mat::compressed2D<double>> pc(A_ns);
        itl::basic_iteration<double> iter(b, 500, 1e-10);
        itl::cgs(A_ns, x, b, pc, iter);
        results.push_back({"CGS (diagonal PC)", iter.iterations(),
                           relative_residual(A_ns, x, b)});
    }

    // BiCGSTAB with identity PC
    {
        vec::dense_vector<double> x(n, 0.0);
        itl::pc::identity<mat::compressed2D<double>> pc(A_ns);
        itl::basic_iteration<double> iter(b, 500, 1e-10);
        itl::bicgstab(A_ns, x, b, pc, iter);
        results.push_back({"BiCGSTAB (identity PC)", iter.iterations(),
                           relative_residual(A_ns, x, b)});
    }

    // BiCGSTAB(2) with identity PC
    {
        vec::dense_vector<double> x(n, 0.0);
        itl::pc::identity<mat::compressed2D<double>> pc(A_ns);
        itl::basic_iteration<double> iter(b, 500, 1e-10);
        itl::bicgstab_ell(A_ns, x, b, pc, iter, 2);
        results.push_back({"BiCGSTAB(2) (identity PC)", iter.iterations(),
                           relative_residual(A_ns, x, b)});
    }

    // BiCGSTAB(4) with identity PC
    {
        vec::dense_vector<double> x(n, 0.0);
        itl::pc::identity<mat::compressed2D<double>> pc(A_ns);
        itl::basic_iteration<double> iter(b, 500, 1e-10);
        itl::bicgstab_ell(A_ns, x, b, pc, iter, 4);
        results.push_back({"BiCGSTAB(4) (identity PC)", iter.iterations(),
                           relative_residual(A_ns, x, b)});
    }

    std::cout << std::left << std::setw(30) << "Solver"
              << std::right << std::setw(8) << "Iters"
              << std::setw(16) << "Rel.Resid" << "\n";
    std::cout << std::string(54, '-') << "\n";
    for (const auto& r : results) {
        std::cout << std::left << std::setw(30) << r.name
                  << std::right << std::setw(8) << r.iters
                  << std::setw(16) << r.rel_resid << "\n";
    }

    // ── 2. Symmetric indefinite system: MINRES ──────────────────────────
    std::cout << "\n--- Symmetric indefinite system (n=" << n << ") ---\n";

    auto A_si = make_symmetric_indefinite(n);
    vec::dense_vector<double> b2(n, 1.0);

    results.clear();

    // MINRES with identity PC
    {
        vec::dense_vector<double> x(n, 0.0);
        itl::pc::identity<mat::compressed2D<double>> pc(A_si);
        itl::basic_iteration<double> iter(b2, 500, 1e-10);
        itl::minres(A_si, x, b2, pc, iter);
        results.push_back({"MINRES (identity PC)", iter.iterations(),
                           relative_residual(A_si, x, b2)});
    }

    // MINRES with diagonal PC
    {
        vec::dense_vector<double> x(n, 0.0);
        itl::pc::diagonal<mat::compressed2D<double>> pc(A_si);
        itl::basic_iteration<double> iter(b2, 500, 1e-10);
        itl::minres(A_si, x, b2, pc, iter);
        results.push_back({"MINRES (diagonal PC)", iter.iterations(),
                           relative_residual(A_si, x, b2)});
    }

    std::cout << std::left << std::setw(30) << "Solver"
              << std::right << std::setw(8) << "Iters"
              << std::setw(16) << "Rel.Resid" << "\n";
    std::cout << std::string(54, '-') << "\n";
    for (const auto& r : results) {
        std::cout << std::left << std::setw(30) << r.name
                  << std::right << std::setw(8) << r.iters
                  << std::setw(16) << r.rel_resid << "\n";
    }

    std::cout << "\nDone.\n";
    return 0;
}
